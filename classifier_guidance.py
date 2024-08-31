import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
from experiment_helpers.training import pil_to_tensor_process
from typing import List

def hook_fn(module, input, output):
    module.output = output

#@title sampling function (with guidance and some extra tricks)
def classifier_sample(pipe: StableDiffusionPipeline, prompt:str, guidance_loss_scale:float,src_image_list: list,src_text_list:list, guidance_scale:int=10,
         negative_prompt:str = "zoomed in, blurry, oversaturated, warped", 
         num_inference_steps:int=30, start_latents:torch.Tensor = None,
         early_stop:int = 20, cfg_norm:bool=True, cfg_decay:bool=True)->Image.Image:
    device=pipe.unet.device
    # If no starting point is passed, create one
    latent_dim=int(src_image_list[0].size[0]*pipe.vae.config.scaling_factor)
    if start_latents is None:
        start_latents = torch.randn((1, 4,latent_dim , latent_dim), device=device)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    src_mid_block_output_list=[]
    for src_image,src_text in zip(src_image_list,src_text_list):
        src_tensor=pil_to_tensor_process(src_image).unsqueeze(0)
        src_latents=pipe.vae.encode(src_tensor).latent_dist.sample()
        small_t=pipe.scheduler.timesteps[-2]
        print("small_t",small_t)
        noise=torch.randn(src_latents.size())
        noisy_src_latents=pipe.scheduler.add_noise(src_latents, noise, small_t)
        text_embedding=pipe._encode_prompt(src_text, device, 1, True, negative_prompt)
        noise_pred = pipe.unet(noisy_src_latents, small_t, encoder_hidden_states=text_embedding).sample
        src_mid_block_output=pipe.unet.mid_block.output
        src_mid_block_output_list.append(src_mid_block_output)



    
    
    pipe.scheduler.set_timesteps(num_inference_steps)
    device=pipe.unet.device

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    latents = start_latents.clone()
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):

        if i > early_stop: guidance_loss_scale = 0 # Early stop (optional)

        sigma = pipe.scheduler.sigmas[i]

        # Set requires grad
        if guidance_loss_scale != 0: latents = latents.detach().requires_grad_()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual with the unet
        if guidance_loss_scale != 0:
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        else:
            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform CFG
        cfg_scale = guidance_scale
        if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
        if cfg_norm:
            noise_pred = noise_pred * (torch.linalg.norm(noise_pred_uncond) / torch.linalg.norm(noise_pred))

        if guidance_loss_scale != 0:
            mid_block_output=pipe.unet.mid_block.output
            for src_mid_block_output in src_mid_block_output_list:
                # Calculate our loss
                loss = torch.nn.functional.mse_loss(mid_block_output, src_mid_block_output_list)

                # Get gradient
                cond_grad = torch.autograd.grad(loss*guidance_loss_scale, latents)[0]

                # Modify the latents based on this gradient
                latents = latents.detach() - cond_grad  * sigma**2 

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())
    
    return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])