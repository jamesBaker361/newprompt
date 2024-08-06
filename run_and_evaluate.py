import sys
sys.path += ['.']
import sys
sys.path.append("/home/jlb638/Desktop/prompt")
import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
os.environ["WANDB_DIR"]="/scratch/jlb638/wandb"
os.environ["WANDB_CACHE_DIR"]="/scratch/jlb638/wandb_cache"
from numpy.linalg import norm
from tqdm.auto import tqdm
import re
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image,UnidentifiedImageError
import wandb
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from static_globals import *
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
import numpy as np
from peft import LoraConfig, get_peft_model
from aesthetic_reward import get_aesthetic_scorer
from chosen_helpers import get_hidden_states,get_best_cluster_kmeans,get_init_dist,loop,get_top_k,generate_with_style
import gc
from dvlab.rival.test_variation_sdv1 import make_eval_image
from instant.infer import instant_generate_one_sample
#from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
#from better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from text_embedding_helpers import prepare_textual_inversion
from trl import DDPOConfig
from pareto import get_dominant_list
import random
from dpok_pipeline import DPOKPipeline
from dpok_scheduler import DPOKDDIMScheduler
from dpok_reward import ValueMulti
from dpok_helpers import _get_batch, _collect_rollout,  _trim_buffer,_train_value_func,TrainPolicyFuncData, _train_policy_func
from facenet_pytorch import MTCNN
from experiment_helpers.elastic_face_iresnet import get_face_embedding,get_iresnet_model,rescale_around_zero,face_mask
from experiment_helpers.measuring import get_metric_dict,get_vit_embeddings
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.training import train_unet as train_unet_function
from experiment_helpers.lora_loading import save_pipeline_hf
from experiment_helpers.better_ddpo_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from experiment_helpers.clothing import clothes_segmentation, get_segmentation_model
from torchvision.transforms import PILToTensor
import torch.nn.functional as F
from huggingface_hub import HfApi

torch.autograd.set_detect_anomaly(True)

def cos_sim_rescaled(vector_i,vector_j,return_np=False):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    try:
        result= cos(vector_i,vector_j) *0.5 +0.5
    except TypeError:
        result= cos(torch.tensor(vector_i),torch.tensor(vector_j)) *0.5 +0.5
    if return_np:
        return result.cpu().numpy()
    return result

def center_crop_to_min_dimension_and_resize(image:Image)->Image:
    width, height = image.size
    min_dimension = min(width, height)
    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = (width + min_dimension) // 2
    bottom = (height + min_dimension) // 2
    return image.crop((left, top, right, bottom)).resize((512,512))

def evaluate_one_sample(
        method_name:str,
        src_image: Image,
        evaluation_prompt_list:list,
        accelerator:Accelerator,
        subject:str,
        num_inference_steps:int,
        n_img_chosen:int,
        target_cluster_size:int,
        min_cluster_size:int,
        convergence_scale:float,
        inf_config:str,
        is_half:bool,
        seed:int,
        inner_round:int,
        editing_early_steps:int,
        train_text_encoder:bool,
        train_text_encoder_embeddings:bool,
        train_unet:bool,
        use_lora_text_encoder:bool,
        use_lora:bool,
        train_gradient_accumulation_steps:int,
        batch_size:int,
        mixed_precision:str,
        reward_method:str,
        num_epochs:int,
        p_step,
        p_batch_size,
        v_flag,
        g_step,
        g_batch_size,
        reward_weight,
        kl_weight,
        kl_warmup,
        buffer_size,
        v_batch_size,
        v_lr,
        v_step,
        save_interval,
        num_samples,
        ratio_clip,
        samples_per_epoch,
        face_margin:int,
        use_face_distance:bool,
        initial_face_weight:float,
        final_face_weight:float,
        use_img_reward:bool,
        initial_img_reward_weight:float,
        final_img_reward_weight:float,
        use_vit_distance:bool,
        initial_vit_weight:float,
        final_vit_weight:float,
        use_vit_style:bool,
        initial_vit_style_weight:float,
        final_vit_style_weight:float,
        use_vit_content:bool,
        initial_vit_content_weight:float,
        final_vit_content_weight:float,
        image_dir:str,
        value_epochs:int,
        normalize_rewards:bool,
        normalize_rewards_individually:bool,
        n_normalization_images:int,
        use_value_function:bool,
        p_lr:bool,
        ddpo_lr:float,
        use_mse:bool,
        initial_mse_weight:float,
        final_mse_weight:float,
        use_mse_vae:bool,
        pretrain:bool,
        pretrain_epochs: int,
        pretrain_steps_per_epoch:int,
        per_prompt_stat_tracking:bool,
        label:str,
        ddpo_save_hf_tag:str,
        use_fashion_clip:bool,
        use_fashion_clip_segmented:bool,
        initial_fashion_clip_weight:float,
        final_fashion_clip_weight:float,
        multi_rewards:list)->dict:
    os.makedirs(image_dir,exist_ok=True)
    method_name=method_name.strip()
    src_image=center_crop_to_min_dimension_and_resize(src_image)
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")
    ir_model.requires_grad_(False)
    ir_model.eval()
    mtcnn=MTCNN(device="cpu")
    mtcnn.requires_grad_(False)
    mtcnn.eval()
    iresnet=get_iresnet_model("cpu")
    #mtcnn,iresnet=accelerator.prepare(mtcnn,iresnet)
    src_face_embedding=get_face_embedding([src_image],mtcnn,iresnet,10)[0]

    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model.eval()
    blip_model.requires_grad_(False)
    #blip_model.to(accelerator.device)

    #blip_processor,blip_model=accelerator.prepare(blip_processor,blip_model)

    caption_inputs = blip_processor(src_image, "", return_tensors="pt")
    caption_out=blip_model.generate(**caption_inputs)
    caption=blip_processor.decode(caption_out[0],skip_special_tokens=True).strip()
    print("blip caption ",caption)
    print("subject",subject)

    vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
    vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16')
    vit_model.eval()
    vit_model.requires_grad_(False)
    #vit_model.to(accelerator.device)
    #vit_model=accelerator.prepare(vit_model)

    normalization_image_list=[]

    max_train_steps=samples_per_epoch*num_epochs
    wandb_tracker=accelerator.get_tracker("wandb")

    src_image_tensor=PILToTensor()(src_image)
    src_image_tensor=rescale_around_zero(src_image_tensor)

    mse_vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    src_image_embedding=mse_vae.encode(src_image_tensor.unsqueeze(0))
    

    def get_mse_from_src(image:Image):
        image_tensor=PILToTensor()(image)
        image_tensor=rescale_around_zero(image_tensor)

        if use_mse_vae:
            image_latents=mse_vae.encode(image_tensor.unsqueeze(0)).latent_dist.sample()
            src_image_latents=src_image_embedding.latent_dist.sample()
            #print('src_image_latents.size(),image_latents.size()',src_image_latents.size(),image_latents.size() )
            loss= F.mse_loss(image_latents, src_image_latents,reduction="mean")
        else:
            #print('image_tensor.size(),src_image_tensor.size() ',image_tensor.size(),src_image_tensor.size() )
            loss= F.mse_loss(image_tensor, src_image_tensor,reduction="mean")
        torch.cuda.empty_cache()
        return loss



    def get_fashion_embedding(fashion_src:Image.Image,fashion_clip_processor:CLIPProcessor,fashion_clip_model:CLIPModel)-> np.ndarray:
        fashion_clip_inputs=fashion_clip_processor(text=[" "], images=[fashion_src], return_tensors="pt", padding=True)
        fashion_clip_inputs["input_ids"]=fashion_clip_inputs["input_ids"].to(fashion_clip_model.device)
        fashion_clip_inputs["pixel_values"]=fashion_clip_inputs["pixel_values"].to(fashion_clip_model.device)
        fashion_clip_inputs["attention_mask"]=fashion_clip_inputs["attention_mask"].to(fashion_clip_model.device)
        try:
            fashion_clip_inputs["position_ids"]= fashion_clip_inputs["position_ids"].to(fashion_clip_model.device)
        except:
            pass

        fashion_clip_outputs = fashion_clip_model(**fashion_clip_inputs)
        fashion_embedding=fashion_clip_outputs.image_embeds.detach().cpu().numpy()[0]
        return fashion_embedding

    fashion_src=src_image
    if use_fashion_clip_segmented:
        segmentation_model=get_segmentation_model(accelerator.device,weight_dtype)
        fashion_src=clothes_segmentation(src_image,segmentation_model,0)
    fashion_clip_processor=CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    fashion_clip_model=CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    fashion_clip_model.eval()
        
    fashion_src_embedding=get_fashion_embedding(fashion_src,fashion_clip_processor, fashion_clip_model)
    vit_src_image_embedding_list,vit_src_style_embedding_list,vit_src_content_embedding_list=get_vit_embeddings(
        vit_processor,vit_model,[src_image],False
    )
    vit_src_image_embedding=vit_src_image_embedding_list[0]
    vit_src_style_embedding=vit_src_style_embedding_list[0]
    vit_src_content_embedding=vit_src_content_embedding_list[0]
    


    def get_reward_fn():
        
        def _reward_fn(images, prompts, epoch,):
            print(images)
            vit_similarities=[0.0 for _ in images]
            face_similarities=[0.0 for _ in images]
            rewards=[0.0 for _ in images]
            scores=[0.0 for _ in images]
            style_similarities=[0.0 for _ in images]
            content_similarities=[0.0 for _ in images]
            mse_distances=[0.0 for _ in images]
            fashion_similarities=[0.0 for _ in images]
            time_factor=(float(epoch)/num_epochs)
            if method_name==DPOK:
                total_steps=max_train_steps//p_step
                time_factor=float(epoch)/float(total_steps)
            if use_vit_content or use_vit_style or use_vit_distance:
                vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(
                    vit_processor,vit_model,images,False
                )
            if use_vit_distance:
                vit_weight=initial_vit_weight+((final_vit_weight-initial_vit_weight)*time_factor)
                vit_similarities=[ cos_sim_rescaled(vit_src_image_embedding,embedding)
                        for embedding in vit_embedding_list]
                vit_similarities=[vit_weight*v for v in vit_similarities]
                wandb_tracker.log({
                    "vit_distance":np.mean(vit_similarities)
                })
            if use_vit_content:
                vit_content_weight=initial_vit_content_weight+((final_vit_content_weight-initial_vit_content_weight)*time_factor)
                content_similarities=[
                    cos_sim_rescaled(vit_src_content_embedding,content_embedding)
                    for content_embedding  in vit_content_embedding_list
                ]
                content_similarities=[
                    vit_content_weight * v for v in content_similarities
                ]
                wandb_tracker.log(
                    {"content_distance":np.mean(content_similarities)}
                )
            if use_vit_style:
                vit_style_weight=initial_vit_style_weight+((final_vit_style_weight-initial_vit_style_weight)*time_factor)
                style_similarities=[
                    cos_sim_rescaled(vit_src_style_embedding, style_embedding)
                    for style_embedding in vit_style_embedding_list
                ]
                style_similarities=[
                    vit_style_weight * v for v in style_similarities
                ]
                wandb_tracker.log({
                    "style_distance":np.mean(style_similarities)
                })
            if use_face_distance:
                face_weight=initial_face_weight+ ((final_face_weight-initial_face_weight)*time_factor)
                try:
                    image_face_embeddings=get_face_embedding(images,mtcnn,iresnet,face_margin)
                    face_similarities=[
                        cos_sim_rescaled(src_face_embedding,face_embedding)
                        for face_embedding in  image_face_embeddings
                    ]
                    face_similarities=[
                        face_weight * v for v in face_similarities
                    ]
                    wandb_tracker.log({
                        "face_distance":np.mean(face_similarities)
                    })
                except (RuntimeError,TypeError):
                    pass
            if use_img_reward:
                img_reward_weight=initial_img_reward_weight+((final_img_reward_weight-initial_img_reward_weight) * time_factor)
                scores=[ir_model.score( prompt.replace(PLACEHOLDER, subject),image) for prompt,image in zip(prompts,images)] #by default IR is normalized to N(0,1) so we rescale
                scores=[0.5 + v/4 for v in scores]
                scores=[s*img_reward_weight for s in scores]
                wandb_tracker.log({
                    "score":np.mean(scores)
                })
            if use_mse:
                mse_reward_weight=initial_mse_weight+((final_mse_weight-initial_mse_weight) *time_factor)
                mse_distances=[
                    -1.0* get_mse_from_src(image) for image in images
                ]
                mse_distances=[mse_reward_weight*m for m in mse_distances]
                try:
                    wandb_tracker.log({
                        "mse_distance":np.mean(mse_distances)
                    })
                except:
                    wandb_tracker.log({
                        "mse_distance":np.mean([m.detach().cpu().numpy() for m in mse_distances])
                    })
            
            if use_fashion_clip:
                fashion_similarities=[
                    cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(image,fashion_clip_processor,fashion_clip_model)) for image in images
                ]
            elif use_fashion_clip_segmented:
                segmented_images=[
                    clothes_segmentation(image,segmentation_model,0) for image in images
                ]
                fashion_similarities=[
                    cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(seg_image,fashion_clip_processor,fashion_clip_model)) for seg_image in segmented_images
                ]
            if use_fashion_clip or use_fashion_clip_segmented:
                fashion_clip_weight=initial_fashion_clip_weight+((final_fashion_clip_weight-initial_fashion_clip_weight) * time_factor)
                fashion_similarities=[fashion_clip_weight * f for f in fashion_similarities]
                wandb_tracker.log({
                    "fashion_distance":np.mean(fashion_similarities)
                })
                


            rewards=[
                d+f+s+vs+vc+m+fas for d,f,s,vs,vc,m,fas in zip(vit_similarities,face_similarities,
                                                       scores,style_similarities, content_similarities,mse_distances,fashion_similarities)
            ]
            try:
                wandb_tracker.log({
                    "reward_fn":np.mean(rewards)
                })
            except RuntimeError:
                wandb_tracker.log({
                    "reward_fn":np.mean([r.detach().cpu().numpy() for r in rewards])
                })
            if reward_method==REWARD_PARETO:
                dominant_list=get_dominant_list(vit_similarities,scores,face_similarities,style_similarities, content_similarities)
                for i in range(len(scores)):
                    if i not in dominant_list:
                        rewards[i]=0.0
            return rewards
        
        return _reward_fn

    prompt_list= [
        "a photo of {}",
        #"a rendering of {}",
        "a cropped photo of {}",
        #"the photo of {}",
        "a clean photo of {}",
        "a dirty photo of  {}",
        "a dark photo of {}",
        #"my photo of {}",
        "a cool photo of  {}",
        "a close-up photo of {}",
        "a bright photo of {}",
        #"a cropped photo of {}",
        #"a photo of {}",
        "a good photo of {}",
        #"one photo of {}",
       # "a close-up photo of {}",
        #"a rendition of {}",
        #"a clean photo of {}",
        #"a rendition of {}",
        "a nice photo of {}",
        #"a good photo of {}"
        ]
    weight_dtype={
            "no":torch.float32,
            "fp16":torch.float16,
            "bf16":torch.bfloat16
        }[accelerator.mixed_precision]
    if method_name == BLIP_DIFFUSION:
        blip_diffusion_pipe=BlipDiffusionPipeline.from_pretrained(
            "Salesforce/blipdiffusion", torch_dtype=torch.float32)
        blip_diffusion_pipe=accelerator.prepare(blip_diffusion_pipe)
        evaluation_image_list=[
            blip_diffusion_pipe(
                evaluation_prompt.format(subject),
                src_image,
                subject,
                subject,
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                neg_prompt=NEGATIVE,
                height=512,
                width=512,
                ).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==ELITE:
        pass
    elif method_name==RIVAL:
        evaluation_image_list=[
            make_eval_image(inf_config,accelerator,is_half,
                            "runwayml/stable-diffusion-v1-5",
                            evaluation_prompt.format(subject),
                            NEGATIVE,src_image,seed,inner_round,
                            num_inference_steps,editing_early_steps) for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==INSTANT:
        evaluation_image_list=[
            instant_generate_one_sample(src_image,evaluation_prompt.format(subject),
                                        NEGATIVE, num_inference_steps, 
                                        accelerator ) for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==IP_ADAPTER or method_name==FACE_IP_ADAPTER:
        pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None)
        unet=pipeline.unet
        vae=pipeline.vae
        tokenizer=pipeline.tokenizer
        text_encoder=pipeline.text_encoder
        pipeline("none",num_inference_steps=1) #things initialize weird if we dont do it once
        if method_name==IP_ADAPTER:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        elif method_name==FACE_IP_ADAPTER:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
        image_encoder=pipeline.image_encoder
        unet,text_encoder,vae,tokenizer,image_encoder = accelerator.prepare(
            unet,text_encoder,vae,tokenizer,image_encoder
        )
        for model in [vae,unet,text_encoder, image_encoder]:
            model.eval()
            model.requires_grad_(False)
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(subject),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None,
                    ip_adapter_image=src_image).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        '''elif method_name==DDPO:
        pipeline=BetterDefaultDDPOStableDiffusionPipeline(
            train_text_encoder,
            train_text_encoder_embeddings,
            train_unet,
            use_lora_text_encoder,
            use_lora=use_lora,
            pretrained_model_name="runwayml/stable-diffusion-v1-5"
        )

        print("len trainable parameters",len(pipeline.get_trainable_layers()))

        entity_name=subject
        if train_text_encoder_embeddings:
            entity_name=PLACEHOLDER
            pipeline.sd_pipeline.tokenizer, pipeline.sd_pipeline.text_encoder,placeholder_token_ids=prepare_textual_inversion(PLACEHOLDER,pipeline.sd_pipeline.tokenizer, pipeline.sd_pipeline.text_encoder)
        


        config=DDPOConfig(
            train_learning_rate=ddpo_lr,
            num_epochs=num_epochs,
            train_gradient_accumulation_steps=train_gradient_accumulation_steps,
            sample_num_steps=num_inference_steps,
            sample_batch_size=batch_size,
            train_batch_size=batch_size,
            sample_num_batches_per_epoch=samples_per_epoch,
            mixed_precision=mixed_precision,
            tracker_project_name="ddpo-personalization",
            log_with="wandb",
            per_prompt_stat_tracking=per_prompt_stat_tracking,
            accelerator_kwargs={
                #"project_dir":args.output_dir
            },
            #project_kwargs=project_kwargs
        )
        
        def prompt_fn():
            return entity_name,{}

        image_samples_hook=get_image_sample_hook(image_dir)
        _reward_fn=get_reward_fn()
        def reward_fn(images, prompts, epoch,prompt_metadata):
            return _reward_fn(images, prompts, epoch),{}
        subject_key=re.sub(r'\s+', '_', subject)
        trainer = BetterDDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook,
            subject_key
        )

        if pretrain:
            #pretrain_image_list=[src_image] *pretrain_steps_per_epoch
            pretrain_image_list=[]
            pretrain_prompt_list=[]
            for x in range(pretrain_steps_per_epoch):
                if x%2==0:
                    pretrain_image_list.append(src_image)
                else:
                    pretrain_image_list.append(src_image.transpose(Image.FLIP_LEFT_RIGHT))
                pretrain_prompt_list.append(prompt_list[x%len(prompt_list)])
            assert len(pretrain_image_list)==len(pretrain_prompt_list), f"error {len(pretrain_image_list)} != {len(pretrain_prompt_list)}"
            assert len(pretrain_image_list)==pretrain_steps_per_epoch, f"error {len(pretrain_image_list)} != {pretrain_steps_per_epoch}"
            
            pretrain_optimizer=trainer._setup_optimizer([p for p in pipeline.sd_pipeline.unet.parameters() if p.requires_grad])

            pipeline.sd_pipeline=train_unet_function(
                pipeline.sd_pipeline,
                pretrain_epochs,
                pretrain_image_list,
                pretrain_prompt_list,
                pretrain_optimizer,
                False,
                "prior",
                batch_size,
                1.0,
                subject,
                trainer.accelerator,
                num_inference_steps,
                0.0,
                True
            )
            torch.cuda.empty_cache()
            trainer.accelerator.free_memory()
        pipeline.sd_pipeline.scheduler.alphas_cumprod=pipeline.sd_pipeline.scheduler.alphas_cumprod.to("cpu")

        print(f"acceleerate device {trainer.accelerator.device}")
        tracker=trainer.accelerator.get_tracker("wandb").run
        with accelerator.autocast():
            trainer.train(retain_graph=False,normalize_rewards=normalize_rewards)
        evaluation_image_list=[
            pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        save_pipeline_hf(pipeline, f"jlbaker361/{ddpo_save_hf_tag}_{label}",f"/scratch/jlb638/{ddpo_save_hf_tag}_{label}")
        new_file=f"{entity_name}_{method_name}.txt"
        with open(new_file,"w+") as txt_file:
            txt_file.write(entity_name)
        api = HfApi()
        api.upload_file(
            path_or_fileobj= new_file,
            path_in_repo="entity_name.txt",
            repo_id=f"jlbaker361/{ddpo_save_hf_tag}_{label}",
            repo_type="model"
        )
        del pipeline'''
    elif method_name==DDPO_MULTI or DDPO:
        pipeline=BetterDefaultDDPOStableDiffusionPipeline(
            train_text_encoder,
            train_text_encoder_embeddings,
            train_unet,
            use_lora_text_encoder,
            use_lora=use_lora,
            pretrained_model_name="runwayml/stable-diffusion-v1-5"
        )
        print("len trainable parameters",len(pipeline.get_trainable_layers()))
        prompts=[]
        pretrain_image_list=[]
        face_key=subject
        fashion_key="clothes"
        content_key=f"{subject} wearing clothes"
        style_key=" league of legends style"
        if multi_rewards==None or len(multi_rewards)==0:
            prompts=[entity_name]
            pretrain_image_list=[src_image]
        for reward in multi_rewards:
            pretrain_img=src_image
            if reward==FACE_REWARD:
                pretrain_entity=face_key
                pretrain_img=face_mask(src_image,mtcnn,10)
            elif reward==FASHION_REWARD:
                pretrain_entity=fashion_key
                segmentation_model=get_segmentation_model(accelerator.device,weight_dtype)
                fashion_src=clothes_segmentation(src_image,segmentation_model,0)
                pretrain_img=fashion_src
            elif reward==CONTENT_REWARD or reward==BODY_REWARD:
                pretrain_entity=content_key
            elif reward==STYLE_REWARD:
                pretrain_entity=style_key
            prompts.append(pretrain_entity)
            #prompts.append(pretrain_entity)
            pretrain_image_list.append(pretrain_img)
            #pretrain_image_list.append(pretrain_img.transpose(Image.FLIP_LEFT_RIGHT))

        config=DDPOConfig(
            train_learning_rate=ddpo_lr,
            num_epochs=num_epochs,
            train_gradient_accumulation_steps=train_gradient_accumulation_steps,
            sample_num_steps=num_inference_steps,
            sample_batch_size=batch_size,
            train_batch_size=batch_size,
            sample_num_batches_per_epoch=samples_per_epoch,
            mixed_precision=mixed_precision,
            tracker_project_name="ddpo-personalization",
            log_with="wandb",
            per_prompt_stat_tracking=per_prompt_stat_tracking,
            accelerator_kwargs={
                #"project_dir":args.output_dir
            },
            #project_kwargs=project_kwargs
        )
        if method_name==DDPO_MULTI:

            def prompt_fn():
                return random.choice(prompts),{}
            
            def reward_fn(images, prompts, epoch,prompt_metadata):
                rewards=[]
                for image,prompt in zip(images,prompts):
                    if prompt==fashion_key:
                        if use_fashion_clip_segmented:
                            image=clothes_segmentation(image,segmentation_model,0)
                        reward=cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(image,fashion_clip_processor,fashion_clip_model))
                    elif prompt==face_key:
                        face_embedding=get_face_embedding([image],mtcnn,iresnet,face_margin)[0]
                        reward=cos_sim_rescaled(src_face_embedding,face_embedding)
                    elif prompt==content_key and CONTENT_REWARD in multi_rewards:
                        vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(
                        vit_processor,vit_model,[image],False)
                        content_embedding=vit_content_embedding_list[0]
                        reward=cos_sim_rescaled(content_embedding,vit_src_content_embedding)
                    elif prompt==content_key and BODY_REWARD in multi_rewards:
                        face_embedding=get_face_embedding([image],mtcnn,iresnet,face_margin)[0]
                        if use_fashion_clip_segmented:
                            image=clothes_segmentation(image,segmentation_model,0)

                        fashion_reward=0.5*cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(image,fashion_clip_processor,fashion_clip_model))
                        face_reward=0.5*cos_sim_rescaled(src_face_embedding,face_embedding)
                        reward=fashion_reward+face_reward
                    elif prompt==style_key:
                        vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(
                        vit_processor,vit_model,[image],False)
                        style_embedding=vit_style_embedding_list[0]
                        reward=cos_sim_rescaled(style_embedding,vit_src_style_embedding)
                    rewards.append(reward)
                return rewards,{}
        elif method_name==DDPO:
            def prompt_fn():
                return entity_name,{}

            _reward_fn=get_reward_fn()
            def reward_fn(images, prompts, epoch,prompt_metadata):
                return _reward_fn(images, prompts, epoch),{}
                    

        image_samples_hook=get_image_sample_hook(image_dir)
        subject_key=re.sub(r'\s+', '_', subject)
        trainer = BetterDDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook,
            subject_key
        )

        if pretrain:
            #pretrain_image_list=[src_image] *pretrain_steps_per_epoch
            _pretrain_image_list=[]
            _pretrain_prompt_list=[]
            for x in range(pretrain_steps_per_epoch):
                if x%2==0:
                    _pretrain_image_list.append(pretrain_image_list[x% len(pretrain_image_list)])
                else:
                    _pretrain_image_list.append(pretrain_image_list[x% len(pretrain_image_list)].transpose(Image.FLIP_LEFT_RIGHT))
                _pretrain_prompt_list.append(prompts[x%len(prompts)])
            pretrain_prompt_list=_pretrain_prompt_list
            pretrain_image_list=_pretrain_image_list
            assert len(pretrain_image_list)==len(pretrain_prompt_list), f"error {len(pretrain_image_list)} != {len(pretrain_prompt_list)}"
            assert len(pretrain_image_list)==pretrain_steps_per_epoch, f"error {len(pretrain_image_list)} != {pretrain_steps_per_epoch}"
            pretrain_optimizer=trainer._setup_optimizer([p for p in pipeline.sd_pipeline.unet.parameters() if p.requires_grad])
            pipeline.sd_pipeline=train_unet_function(
                pipeline.sd_pipeline,
                pretrain_epochs,
                pretrain_image_list,
                pretrain_prompt_list,
                pretrain_optimizer,
                False,
                "prior",
                batch_size,
                1.0,
                subject,
                trainer.accelerator,
                num_inference_steps,
                0.0,
                True
            )
            torch.cuda.empty_cache()
            trainer.accelerator.free_memory()
        
        pipeline.sd_pipeline.scheduler.alphas_cumprod=pipeline.sd_pipeline.scheduler.alphas_cumprod.to("cpu")

        print(f"acceleerate device {trainer.accelerator.device}")
        tracker=trainer.accelerator.get_tracker("wandb").run
        with accelerator.autocast():
            trainer.train(retain_graph=False,normalize_rewards=normalize_rewards)
        if method_name==DDPO_MULTI:
            entity_name=""
            if (FACE_REWARD in multi_rewards and FASHION_REWARD in multi_rewards) or CONTENT_REWARD in multi_rewards:
                entity_name = content_key
            elif FASHION_REWARD in multi_rewards:
                entity_name=face_key
            elif FASHION_REWARD in multi_rewards:
                entity_name=fashion_key
            
            if STYLE_REWARD in multi_rewards:
                entity_name+= style_key

        print(f"evaluation with entity_name {entity_name}")
        evaluation_image_list=[
            pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        save_pipeline_hf(pipeline, f"jlbaker361/{ddpo_save_hf_tag}_{label}",f"/scratch/jlb638/{ddpo_save_hf_tag}_{label}")
        new_file=f"{entity_name}_{method_name}.txt"
        with open(new_file,"w+") as txt_file:
            txt_file.write(entity_name)
        api = HfApi()
        api.upload_file(
            path_or_fileobj= new_file,
            path_in_repo="entity_name.txt",
            repo_id=f"jlbaker361/{ddpo_save_hf_tag}_{label}",
            repo_type="model"
        )
        del pipeline

        
    else:
        message=f"no support for {method_name} try one of "+" ".join(METHOD_LIST)
        raise Exception(message)

    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    print(evaluation_image_list)
    #METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD]
    metric_dict=get_metric_dict(evaluation_prompt_list, evaluation_image_list,[src_image],None,True)
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    return metric_dict,evaluation_image_list