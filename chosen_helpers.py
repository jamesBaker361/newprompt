from sklearn.cluster import KMeans
import torchvision.transforms as T
import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from transformers import ViTImageProcessor, ViTModel,CLIPProcessor, CLIPModel
import torch.nn.functional as F
from diffusers import  DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UniPCMultistepScheduler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from aesthetic_reward import get_aesthetic_scorer
import ImageReward as image_reward
import wandb
reward_cache="/scratch/jlb638/ImageReward"
from data_helpers import make_dataloader
from static_globals import *
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search,dot_score
from huggingface_hub import hf_hub_download,snapshot_download
from wang.ip_adapter import IPAdapter

#remove backgrounds and use faces?

#vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
#vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

checkpoint_dir=snapshot_download("h94/IP-Adapter")

image_encoder_path = f"{checkpoint_dir}/models/image_encoder"
ip_ckpt = f"{checkpoint_dir}/models/ip-adapter_sd15.bin"

def get_init_dist(last_hidden_states)->float:
    n=len(last_hidden_states)
    total=(n*(n+1)/2)-n
    init_dist=0.0
    for i in range(n):
        for j in range(i+1,n):
            init_dist+= np.linalg.norm(last_hidden_states[i] - last_hidden_states[j])/total
    return init_dist

    

def get_hidden_states(image_list:list, vit_processor: ViTImageProcessor, vit_model:ViTModel):
    vit_inputs = vit_processor(images=image_list, return_tensors="pt")
    #print("inputs :)")
    vit_inputs['pixel_values']=vit_inputs['pixel_values'].to(vit_model.device)
    vit_outputs=vit_model(**vit_inputs)
    #print("outputs :))")
    last_hidden_states = vit_outputs.last_hidden_state.detach()
    #print("last hidden :)))")
    last_hidden_states=last_hidden_states.cpu().numpy().reshape(len(image_list),-1)
    return last_hidden_states

def get_best_cluster_kmeans(
        image_list:list,
                            n_clusters:int,
                            min_cluster_size:int,
                            vit_processor: ViTImageProcessor, 
                            vit_model:ViTModel,*args):
    print(f"best cluster kmeans len(image_list) {len(image_list)}")
    last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
    try:
        print('last_hidden_states[0].shape()',last_hidden_states[0].shape())
        print('last_hidden_states[1].shape()',last_hidden_states[1].shape())
    except:
        pass
    try:
        print('last_hidden_states.shape()',last_hidden_states.shape())
    except:
        pass
    try:
        print('len(last_hidden_states)',len(last_hidden_states))
    except:
        pass
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(last_hidden_states)
    
    cluster_dict={}
    for label,embedding in zip(k_means.labels_, last_hidden_states):
        if label not in cluster_dict:
            cluster_dict[label]=[]
        cluster_dict[label].append(embedding)
    cluster_dict={label:v for label,v in cluster_dict.items() if len(v)>=min_cluster_size}

    dist_dict={}
    for label,v in cluster_dict.items():
        center=k_means.cluster_centers_[label]
        dist=np.mean([np.linalg.norm(center-embedding) for embedding in v])
        dist_dict[label]=dist
    print("dist_dict",dist_dict)
    min_label=[label for label in dist_dict.keys()][0]
    for label,dist in dist_dict.items():
        if dist<=dist_dict[min_label]:
            min_label=label
    valid_image_list=[]
    for label,image in zip(k_means.labels_,  image_list):
        if label==min_label:
            valid_image_list.append(image)
    return valid_image_list, dist_dict[min_label]

def get_ranked_images_list(image_list:list, text_prompt:str,clip_processor:CLIPProcessor, clip_model:CLIPModel)->list:
    clip_inputs=clip_processor(text=[text_prompt], images=image_list, return_tensors="pt", padding=True)
    clip_outputs = clip_model(**clip_inputs)
    logits_per_image=clip_outputs.logits_per_image.detach().numpy()
    try:
        print('clip_outputs.logits_per_image.detach().numpy().shape()',clip_outputs.logits_per_image.detach().numpy().shape())
    except:
        pass
    try:
        print('len(clip_outputs.logits_per_image.detach().numpy())',len(clip_outputs.logits_per_image.detach().numpy()))
    except:
        pass
    try:
        print('logits_per_image.shape()',logits_per_image.shape())
    except:
        pass
    try:
        print('len(logits_per_image)',len(logits_per_image))
    except:
        pass
    try:
        print('logits_per_image[0].shape()',logits_per_image[0].shape())
    except:
        pass
    try:
        print('len(logits_per_image[0])',len(logits_per_image[0]))
    except:
        pass
    pair_list=[ (logit,image) for logit,image in zip(logits_per_image, image_list)]
    pair_list.sort(key=lambda x: x[0])
    print(f"len(pair_list), {len(pair_list)}")
    print([logit for (logit,image) in pair_list])
    return [image for (logit,image) in pair_list]


def get_best_cluster_sorted(
        image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        negative:bool,clip_processor:CLIPProcessor, clip_model:CLIPModel,):
    ranked_image_list=get_ranked_images_list(image_list, text_prompt,clip_processor,clip_model)
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    if negative:
        ranked_image_list=ranked_image_list[:limit]
    else:
        ranked_image_list=ranked_image_list[-limit:]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)

def get_best_cluster_aesthetic(
        image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        *args):
    aesthetic_scorer=get_aesthetic_scorer()
    scored_ranked_image_list=[[ aesthetic_scorer(image).cpu().numpy()[0],image   ] for image in image_list]
    scored_ranked_image_list.sort(reverse=True, key=lambda x: x[0])
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    ranked_image_list=[image for [score,image] in scored_ranked_image_list][:limit]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)

def get_best_cluster_ir(
       image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        *args):
    ir_model=image_reward.load("ImageReward-v1.0",download_root=reward_cache)
    scored_ranked_image_list=[
        [ir_model.score(text_prompt, image),image] for image in image_list
    ]
    scored_ranked_image_list.sort(reverse=True, key=lambda x: x[0])
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    ranked_image_list=[image for [score,image] in scored_ranked_image_list][:limit]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)

def loop(images: list,
               text_prompt_list:list,
               pipeline:StableDiffusionPipeline,
               start_epoch:int,
               optimizer:object,
               accelerator:object,
               epochs:int,
                num_inference_steps:int,
                size:int,
                train_batch_size:int,
                noise_offset:float,
                max_grad_norm:float,
               )->StableDiffusionPipeline:
    '''
    given images generated from text prompt, and the src_image, trains the unet lora pipeline for epochs
    using the prompt and the src_image for conditioning and returns the trained pipeline
    
    images: PIL imgs to be used for training
    ip_adapter_image: 
    text_prompt: text prompts describing character
    pipeline: should already have ip adapter loaded
    start_epoch: epoch we're starting at
    epochs: how many epochs to do this training
    optimizer: optimizer
    acceelrator: accelerator object
    size: img dim 
    train_batch_size: batch size
    with_prior_preservation: whether to use prior preservation (for dreambooth)
    noise_offset: https://www.crosslabs.org//blog/diffusion-with-offset-noise
    '''
    torch.cuda.empty_cache()
    accelerator.free_memory()
    tracker=accelerator.get_tracker("wandb")
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    dataloader=make_dataloader(images,text_prompt_list, tokenizer,size, train_batch_size)
    '''print("len dataloader",len(dataloader))
    print("len images ",len(images))
    print("len text prompt list",len(text_prompt_list))'''
    unet=pipeline.unet
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters()) #optimizer should already be listening to whatever layers we're optimzing
    unet,text_encoder,vae,tokenizer, optimizer, dataloader= accelerator.prepare(
        unet,text_encoder,vae,tokenizer, optimizer, dataloader
    )
    added_cond_kwargs={}
    weight_dtype=pipeline.dtype
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps,clip_sample=False)
    global_step=0
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        for step,batch in enumerate(dataloader):
            batch_size=batch[IMAGES].shape[0]
            print(f"batch size {batch_size}")
            with accelerator.accumulate(unet,text_encoder):
                latents = vae.encode(batch[IMAGES].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if noise_offset:
                    noise += noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch[TEXT_INPUT_IDS])[0]
                #print('text_encoder(batch[TEXT_INPUT_IDS])',text_encoder(batch[TEXT_INPUT_IDS]))
                #print('encoder_hidden_states.size()',encoder_hidden_states.size())

                noise_pred = unet(noisy_latents, 
                                timesteps, 
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item()
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({f"train_loss": train_loss})
                train_loss = 0.0
    del dataloader
    return pipeline


def get_top_k(src_image:Image,image_list:list,vit_processor: ViTImageProcessor, vit_model:ViTModel,top_k):
    src_hidden_states=get_hidden_states([src_image],vit_processor, vit_model)
    hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
    hits = semantic_search(src_hidden_states, hidden_states, 
                                query_chunk_size=src_hidden_states.shape[0], 
                                top_k=top_k,
                                score_function=dot_score)[0]
    valid_images_list= [image_list[hit['corpus_id']] for hit in hits]
    return valid_images_list


def generate_with_style(pipeline:StableDiffusionPipeline,text_prompt:str,num_inference_steps:int,negative_prompt:str,src_image:Image):
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_vae_tiling()
    device=pipeline.unet.device
    ip_model = IPAdapter(pipeline, image_encoder_path, ip_ckpt, device, target_blocks=["block"],dtype=pipeline.dtype)
    return ip_model.generate(pil_image=src_image,
                           prompt=text_prompt,
                           negative_prompt= negative_prompt,
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=num_inference_steps, 
                           seed=42,
                           neg_content_emb=None,
                          )[0]