from PIL import Image,ImageDraw
from dift.src.models.dift_sd import SDFeaturizer,MyUNet2DConditionModel,OneStepSDPipeline
from diffusers import DiffusionPipeline,StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline,retrieve_timesteps
import torch

from diffusers.schedulers.scheduling_ddim import *
from openpose_better import OpenPoseDetectorProbs
from pose_helpers import get_poseresult,intermediate_points_body
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from controlnet_aux.open_pose.body import Keypoint
from tqdm.auto import tqdm
from torchvision import transforms as tfms
import argparse
import wandb
from experiment_helpers.training import train_unet

parser=argparse.ArgumentParser()


def assemble_grid(images, columns=4, bg_color=(255, 255, 255)):
    # Determine the dimensions of the individual images
    image_width, image_height = images[0].size

    # Calculate the number of rows needed
    rows = (len(images) + columns - 1) // columns  # Ceiling division

    # Create a new blank image for the grid
    grid_width = columns * image_width
    grid_height = rows * image_height
    grid_image = Image.new('RGB', (grid_width, grid_height), bg_color)

    # Paste the images into the grid
    for idx, img in enumerate(images):
        x_offset = (idx % columns) * image_width
        y_offset = (idx // columns) * image_height
        grid_image.paste(img, (x_offset, y_offset))

    return grid_image

def step(
    self:DDIMScheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[DDIMSchedulerOutput, Tuple]:
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    #print('\tprev,current',prev_timestep, timestep)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t
    #print('\t t,alpha_prod_t_prev,alpha_prod_t',timestep,alpha_prod_t_prev,alpha_prod_t)
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance

    if not return_dict:
        return (prev_sample,)

    return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator,init_sigma_prepare_latents, latents=None):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // self.vae_scale_factor,
        int(width) // self.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
    else:
        latents = latents.to(device)
        if init_sigma_prepare_latents:
            latents = latents * self.scheduler.init_noise_sigma

    
    return latents

@torch.no_grad()
def forward(
    self:UnsafeStableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    init_sigma_prepare_latents=True,
    **kwargs,
):

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    latent_list=[]
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, _ = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = prepare_latents(self,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        init_sigma_prepare_latents,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            print("inference t",t)
            latent_list.append(latents)
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = step(self.scheduler,noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0] #self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = image,None
    else:
        image = latents
        has_nsfw_concept = None

    do_denormalize = [True] * image.shape[0]
    

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),latent_list


## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    start_noise,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cpu"
):

    # Encode prompt
    positive,negative = pipe.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []
    intermediate_latent_dict={}

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    #timesteps = torch.cat((reversed(pipe.scheduler.timesteps),torch.tensor([1000]).to(pipe.device)))
    timesteps=(reversed(pipe.scheduler.timesteps))

    if do_classifier_free_guidance:
        text_embeddings=torch.cat([negative,positive])
    else:
        text_embeddings=positive

    if start_noise:
        noise=torch.randn(latents.size(),device=device)
        latents+=pipe.scheduler.alphas_cumprod[timesteps[0].item()]*noise

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1 ):

        t = timesteps[i]
        print("inversion loop t",t)
        

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred
        #print('invert t,alpha_t,alpha_t_next',t,alpha_t,alpha_t_next)
        # Store
        intermediate_latents.append(latents)
        intermediate_latent_dict[t.item()]=latents

    return torch.cat(intermediate_latents),intermediate_latent_dict

def mse(image1:Image.Image, image2:Image.Image)->float:
    # Ensure images have the same size
    image1=image1.resize(image2.size)

    # Convert images to NumPy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)

    # Calculate the MSE
    squared_diff = (arr1 - arr2) ** 2
    mse_value = np.mean(squared_diff)

    return mse_value

def draw_points(pose_src_keypoint_list:List[Keypoint],src_image:Image.Image,name="pose"):
    copy_image=src_image.copy()
    draw = ImageDraw.Draw(copy_image)
    for k in pose_src_keypoint_list:
        if k is not None:
            x=k.x*src_image.size[0]
            y=k.y*src_image.size[1]
            radius = 4
            draw.ellipse(
                    (x - radius, y - radius, 
                    x+ radius, y+ radius), 
                    fill='red', outline='red'
                )
    return copy_image

def keypoint_list_to_dict(keypoint_list:List[Keypoint])-> dict:
    d={}
    for k in keypoint_list:
        if k is not None:
            d[k.id]=k
    return d

parser.add_argument("--src_image_path",type=str,default="league5.jpg")
parser.add_argument("--steps",type=int,default=30)
parser.add_argument("--vae_config_coefficient",action="store_true")
parser.add_argument("--init_noise_config_coefficient",action="store_true")
parser.add_argument("--image_dir",type=str,default="graphs/")
parser.add_argument("--intervention_step",type=int,default=15)
parser.add_argument("--gen_image_path",type=str,default="league4.jpg")
parser.add_argument("--src_prompt",type=str,default="woman with blue hair holding gun")
parser.add_argument("--gen_prompt",type=str,default="woman standing and posing")
parser.add_argument("--train_unet",action="store_true")
parser.add_argument("--unet_epochs",type=int,default=20)
parser.add_argument("--project_name",type=str,default="keypoints")
parser.add_argument("--init_sigma_prepare_latents",action="store_true")
parser.add_argument("--intervention_end",type=int,default=25)
parser.add_argument("--src_fraction",type=float,default=1.0)
parser.add_argument("--src_fraction_stop",type=float,default=1.0)

#@torch.no_grad()
def swap_generate(args)-> Image.Image:
    steps=args.steps
    intervention_step=args.intervention_step
    try:
        accelerator=Accelerator(log_with="wandb",mixed_precision="no",gradient_accumulation_steps=8)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))
        
        pipe=UnsafeStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(accelerator.device)
    except:
        pipe=UnsafeStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cpu")


    src_image=Image.open(args.src_image_path).resize((512,512))
    gen_image=Image.open(args.gen_image_path).resize((512,512))
    if args.train_unet:
        unet=pipe.unet
        unet.train()
        optimizer=torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad])
        pipe=train_unet(pipe,args.unet_epochs,[src_image for _ in range(args.steps)]+[gen_image for _ in range(args.steps)],
                        [args.src_prompt for _ in range(args.steps)]+[args.gen_prompt for _ in range(args.steps)],
                        optimizer,
                        False,
                        "woman",
                        1,
                        1.0,
                        "",
                        accelerator,
                        steps,
                        0.0,
                        True
                        )

    
    #make timesteps thing
    height,width=src_image.size
    detector=OpenPoseDetectorProbs.from_pretrained('lllyasviel/Annotators')

    pose_result=get_poseresult(detector, src_image,height,False,True)
    src_keypoints=pose_result.body.keypoints
    intermediate_src_keypoints=intermediate_points_body(src_keypoints,4)
    all_src_keypoints=src_keypoints+intermediate_src_keypoints

    draw_points(all_src_keypoints,src_image).save("src.png")

    all_src_dict=keypoint_list_to_dict(all_src_keypoints)
    coefficient=1.0
    if args.vae_config_coefficient:
        coefficient*=pipe.vae.config.scaling_factor
    if args.init_noise_config_coefficient:
        coefficient*=pipe.scheduler.init_noise_sigma

    with torch.no_grad():
        src_start_latents = pipe.vae.encode(tfms.functional.to_tensor(src_image).unsqueeze(0).to(pipe.device) * 2 - 1)
    src_start_latents = coefficient * src_start_latents.latent_dist.sample()
    #_,inverted_src_dict=invert(src_start_latents,prompt,num_inference_steps=steps,device=pipe.device)
    #print([k for k in inverted_src_dict.keys()])


    generator=torch.Generator()
    timesteps,_=retrieve_timesteps(pipe.scheduler,steps)
    print(timesteps[:intervention_step])
    print(timesteps[intervention_step:])
    '''num_channels_latents = pipeline.unet.config.in_channels
    latents=pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            pipeline.dtype,
            pipeline.device,
            generator,
            None,
        )
    latents_clone=latents.clone()
    print('latesnts size',latents.size())
    
    print("normal image")
    gen_image=forward(pipeline,prompt,height,width,steps,latents=latents).images[0]'''
    #gen_image.save("gen1.png")

    coefficient=1.0
    if args.vae_config_coefficient:
        coefficient*=pipe.vae.config.scaling_factor
    if args.init_noise_config_coefficient:
        coefficient*=pipe.scheduler.init_noise_sigma

    
    with torch.no_grad():
        gen_start_latents = pipe.vae.encode(tfms.functional.to_tensor(gen_image).unsqueeze(0).to(pipe.device) * 2 - 1)
    gen_start_latents = coefficient * gen_start_latents.latent_dist.sample()
    gen_list,gen_latent_dict=invert(pipe,gen_start_latents,args.gen_prompt,args.init_sigma_prepare_latents,num_inference_steps=steps,device=pipe.device)

    def assemble(latent_list, name):
        inverted_image_list=[]
        for l in latent_list:
            with torch.no_grad():
                try:
                    image = pipe.vae.decode(l.unsqueeze(0) / coefficient, return_dict=False, generator=generator)[
                        0
                    ]
                except:
                    image = pipe.vae.decode(l / coefficient, return_dict=False, generator=generator)[
                        0
                    ]
            do_denormalize = [True] * image.shape[0]
        

            image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)[0]

            inverted_image_list.append(image)

        
        img=assemble_grid(inverted_image_list)
        return img #.save(name)

    gen_pose_result=get_poseresult(detector, gen_image,height,False,True)
    gen_keypoints=gen_pose_result.body.keypoints
    intermediate_gen_keypoints=intermediate_points_body(gen_keypoints,4)
    all_gen_keypoints=gen_keypoints+intermediate_gen_keypoints

    #draw_points(all_gen_keypoints, gen_image).save("gen1.png")

    all_gen_dict=keypoint_list_to_dict(all_gen_keypoints)
    #print(all_gen_dict)

    pre_timesteps=timesteps[:intervention_step]
    post_timesteps=timesteps[intervention_step:]

    print("pre")
    #latents_clone=gen_latent_dict[1000]
    #pre_latents=forward(pipeline,prompt,height,width,steps,timesteps=pre_timesteps,latents=latents_clone,output_type="latent").images[0]
    t=post_timesteps[0].item()
    pre_latents=gen_latent_dict[t].cpu().detach().numpy()[0]
    
    #inversion_latents=inverted_src_dict[t].cpu().detach().numpy()[0]
    noise = torch.randn_like(src_start_latents)
    src_latents=pipe.scheduler.add_noise(src_start_latents, noise, post_timesteps[0]).cpu().detach().numpy()[0]
    for k_id,point in all_src_dict.items():
        if k_id in all_gen_dict:
            print('matched for ',k_id)
            src_x=int(point.x*height)//8
            src_y=int(point.y*width)//8

            gen_point=all_gen_dict[k_id]
            gen_x=int(gen_point.x*height)//8
            gen_y=int(gen_point.y*width)//8

            

            #print(pre_latents.size())
            #print(inversion_latents.size())

            
            #print(pre_latents[:,gen_x,gen_y])
            #print(inversion_latents[:,src_x,src_y])
            pre_latents[:,gen_x,gen_y]=args.src_fraction*src_latents[:,src_x,src_y]+(1-args.src_fraction)*pre_latents[:,gen_x,gen_y]
    pre_latents=torch.tensor(pre_latents).unsqueeze(0)
    #print(pre_latents.size())
    print("post")
    post_image,latent_list=forward(pipe,args.gen_prompt,height,width,steps,timesteps=post_timesteps,init_sigma_prepare_latents=args.init_sigma_prepare_latents,latents=pre_latents)

    
    img=assemble(latent_list,"grid_forward.jpg")
    score=mse(img,gen_image)
    accelerator.log({
        "grid_forward":wandb.Image(img),
        "grid_forward_mse":score
    })

    post_image,latent_list=forward(pipe,args.gen_prompt,height,width,steps,
        init_sigma_prepare_latents=args.init_sigma_prepare_latents,latents=pre_latents)
    img=assemble(latent_list,"grid_forward_no_timesteps.jpg")
    score=mse(img,gen_image)
    accelerator.log({
        "grid_forward_no_time":wandb.Image(img),
        "grid_forward_no_time_mse":score
    })
    

    post_image,latent_list=forward(pipe,args.gen_prompt,height,width,steps,
                                   init_sigma_prepare_latents=args.init_sigma_prepare_latents,timesteps=post_timesteps)    
    img=assemble(latent_list,"grid_forward_no_latents.jpg")
    score=mse(img,gen_image)
    accelerator.log({
        "grid_forward_no_latents":wandb.Image(img),
        "grid_forward_no_latents_mse":score
    })

    post_image,latent_list=forward(pipe,args.gen_prompt,height,width,steps,init_sigma_prepare_latents=args.init_sigma_prepare_latents)    
    img=assemble(latent_list,"grid_forward_nothing.jpg")
    score=mse(img,gen_image)
    accelerator.log({
        "grid_forward_nothing":wandb.Image(img),
        "grid_forward_nothing_mse":score
    })


    post_image=post_image.images[0]
    total_width = gen_image.width + post_image.width
    max_height = max(gen_image.height, post_image.height)

    # Create a blank canvas to paste the images
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the two images onto the blank canvas
    new_image.paste(gen_image, (0, 0))
    new_image.paste(post_image, (gen_image.width, 0))

    # Save the concatenated image
    new_image.save("concatenated_image_horizontal.jpg")

    return

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    swap_generate(args)