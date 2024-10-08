import sys
sys.path += ['.']
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
from insightface.app import FaceAnalysis
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image,UnidentifiedImageError,ImageDraw
import wandb
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from functools import partial
from static_globals import *
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
import numpy as np
import gc
from dvlab.rival.test_variation_sdv1 import make_eval_image
from instant.infer import instant_generate_one_sample
#from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
#from better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from trl import DDPOConfig
from pareto import get_dominant_list
import random
from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face
from experiment_helpers.elastic_face_iresnet import get_face_embedding,get_iresnet_model,rescale_around_zero,face_mask
from experiment_helpers.measuring import get_metric_dict,get_vit_embeddings
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.training import train_unet as train_unet_function
from experiment_helpers.training import train_unet_single_prompt
from controlnet_aux.open_pose.body import Keypoint
from experiment_helpers.lora_loading import save_pipeline_hf
from experiment_helpers.better_ddpo_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from experiment_helpers.clothing import clothes_segmentation, get_segmentation_model
from experiment_helpers.cloth_process import generate_mask,load_seg_model,get_palette
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from torchvision.transforms import PILToTensor
import torch.nn.functional as F
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from openpose_better import OpenPoseDetectorProbs
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel,DDIMScheduler
import torchvision
from torchvision import transforms
import cv2
from experiment_helpers.background import remove_background,remove_background_birefnet
from transformers import AutoModelForImageSegmentation
from swin_mae import SwinMAE
from proto_gan_models import Discriminator
from controlnet_test import OpenposeDetectorResize
from classifier_guidance import classifier_call
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from einops import rearrange
from nearest_neighbors import nearest,cos_sim_rescaled
from dift.src.models.dift_sd import SDFeaturizer,MyUNet2DConditionModel,OneStepSDPipeline
from pose_helpers import get_poseresult,intermediate_points_body
from typing import List
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.autograd.set_detect_anomaly(True)

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
        samples_per_epoch,
        face_margin:int,
        use_face_distance:bool,
        initial_face_weight:float,
        final_face_weight:float,
        use_img_reward:bool,
        initial_img_reward_weight:float,
        final_img_reward_weight:float,
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
        multi_rewards:list,
        use_dream_sim:bool,
        initial_dream_sim_weight:float,
        final_dream_sim_weight:float,
        use_face_probs:bool,
        initial_face_probs_weight:float,
        final_face_probs_weight:float,
        use_pose_probs:bool,
        initial_pose_probs_weight:float,
        final_pose_probs_weight:float,
        remove_background_flag:bool,
        classifier_eta:float,
        use_clip_align:bool,
        initial_clip_align_weight:float,
        final_clip_align_weight:float,
        clip_align_prompt:str,
        semantic_matching:bool,
        semantic_matching_points:int,
        semantic_matching_strategy:str,
        use_dift:bool,
        initial_dift_weight:float,
        final_dift_weight:float,
        dift_t:int,
        dift_up_ft_index:int,
        dift_model:str,
        use_ip_adapter_ddpo:bool,
        custom_dift:bool,
        custom_dift_epochs:int,
        custom_dift_steps_per_epoch)->dict:
    set_seed(1234)
    os.makedirs(image_dir,exist_ok=True)
    detector=OpenPoseDetectorProbs.from_pretrained('lllyasviel/Annotators')
    method_name=method_name.strip()
    #src_image=center_crop_to_min_dimension_and_resize(src_image)
    H,W=src_image.size
    
    pose_result=get_poseresult(detector, src_image,H,False,True)
    interm_points=intermediate_points_body(pose_result.body.keypoints,2)
    pose_src_keypoint_list=interm_points+pose_result.body.keypoints
    def draw_points(pose_src_keypoint_list:List[Keypoint],src_image:Image.Image,name="pose"):
        copy_image=src_image.copy()
        draw = ImageDraw.Draw(copy_image)
        for k in pose_src_keypoint_list:
            if k is not None:
                x=k.x*H
                y=k.y*W
                radius = 4
                draw.ellipse(
                        (x - radius, y - radius, 
                        x+ radius, y+ radius), 
                        fill='red', outline='red'
                    )
        try:
            accelerator.log({
                name:copy_image
            })
        except:
            try:
                accelerator.log({
                name:wandb.Image(copy_image)
                })
            except:
                copy_image.save("temp.png")
                accelerator.log({
                    name:wandb.Image("temp.png")
                })
    draw_points(pose_src_keypoint_list, src_image)

    
    def get_keypoint_dict(keypoint_list:List[Keypoint],rescale=32)->dict:
        k_dict={}
        for k in keypoint_list:
            if k is not None:
                k_dict[k.id]=(int(k.x*H)//rescale, int(k.y*W)//rescale)
        return k_dict
    
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(accelerator.device)
    removed_src,mask=remove_background_birefnet(src_image,birefnet,return_mask=True)
    if remove_background_flag==False:
        removed_src=src_image
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")
    ir_model.requires_grad_(False)
    ir_model.eval()
    mtcnn=MTCNN(device="cpu")
    mtcnn.requires_grad_(False)
    mtcnn.eval()
    iresnet=get_iresnet_model("cpu")
    #mtcnn,iresnet=accelerator.prepare(mtcnn,iresnet)
    src_face_embedding=get_face_embedding([src_image],mtcnn,iresnet,10)[0]
    boxes, probs=mtcnn.detect(src_image)
    if boxes is not None:
        extracted_face_tensor=extract_face(src_image,boxes[0],112,10)
    else:
        extracted_face_tensor=torch.ones((3,112,112))
    face_image=torchvision.transforms.ToPILImage()(extracted_face_tensor)
    print("subject",subject)

    width,height=src_image.size
    print("width,height",src_image.size)
    
    transform_list = [
            transforms.Resize((width,height)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    composed_trans = transforms.Compose(transform_list)

    
        
    if use_dift:
        sd_featurizer=SDFeaturizer(dift_model)
        if custom_dift:
            my_pipeline=UnsafeStableDiffusionPipeline.from_pretrained(dift_model)
            my_pipeline.unet.train()
            my_pipeline=my_pipeline.to(accelerator.device)
            dift_training_image_list=[]
            while len(dift_training_image_list)< custom_dift_steps_per_epoch:
                dift_training_image_list.append(src_image)
                dift_training_image_list.append(src_image.transpose(Image.FLIP_LEFT_RIGHT))
            dift_optimizer=torch.optim.AdamW([p for p in my_pipeline.unet.parameters() if p.requires_grad],ddpo_lr)
            my_pipeline=train_unet_single_prompt(my_pipeline,
                                                 custom_dift_epochs,
                                                 dift_training_image_list,
                                                 "character",
                                                 dift_optimizer,
                                                 False,
                                                 "character",
                                                 batch_size,
                                                 1.0,
                                                 "character",
                                                 accelerator,
                                                 num_inference_steps,
                                                 0.0,
                                                 True,
                                                 log_images=2
                                                 )
            my_unet=MyUNet2DConditionModel.from_pretrained(dift_model,subfolder="unet")
            my_unet.load_state_dict(my_pipeline.unet.state_dict())
            one_step_pipeline=OneStepSDPipeline.from_pretrained(dift_model,unet=my_unet,safety_checker=None)
            sd_featurizer.pipe=one_step_pipeline.to(accelerator.device)
            
        src_image_tensor=PILToTensor()(removed_src.resize((1024,1024)))
        src_image_tensor=rescale_around_zero(src_image_tensor)
        src_dift_ft=sd_featurizer.forward(src_image_tensor,t=dift_t,up_ft_index=dift_up_ft_index,ensemble_size=1).squeeze(0).cpu()
        print("src_dift_ft size",src_dift_ft.size())
        dift_size=src_dift_ft.size()[-2:]
        print("dift_size ",dift_size)
        image_mask=np.array(src_image.resize(dift_size).convert("L"))
        valid_pixels = np.argwhere(image_mask != 0)
        #openpose_valid_pixels=[(int(k.x*H)//32, int(k.y*W)//32) for k in pose_src_keypoint_list if k is not None]
        rescale=width//dift_size[-1]
        keypoint_dict=get_keypoint_dict(pose_src_keypoint_list,rescale)
    

    max_train_steps=samples_per_epoch*num_epochs

    src_image_tensor=PILToTensor()(removed_src)
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
        return loss.detach().cpu().numpy()
    
    



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

    seg_model=load_seg_model("/scratch/jlb638/fashion_segmentation/cloth_segm.pth",device=accelerator.device)
    fashion_src=src_image
    if use_fashion_clip_segmented:
        fashion_src=generate_mask(src_image,seg_model,accelerator.device)
    fashion_clip_processor=CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    fashion_clip_model=CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    fashion_clip_model.eval()
        
    fashion_src_embedding=get_fashion_embedding(fashion_src,fashion_clip_processor, fashion_clip_model)

    torch.cuda.empty_cache()
    accelerator.free_memory()
    
    clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def semantic_reward(image:Image.Image,semantic_method:str):

        similarity=0.0
        
        if semantic_method=="dift":
            resize_dim=dift_size
            image_tensor=PILToTensor()(image.resize((1024,1024)))
            image_tensor=rescale_around_zero(image_tensor)
            image_ft=sd_featurizer.forward(image_tensor,t=dift_t,up_ft_index=dift_up_ft_index,ensemble_size=1).squeeze(0).cpu()
            src_image_ft=src_dift_ft

        if semantic_matching_strategy==NEAREST_NEIGHBORS:
            sampled_indices = random.sample(list(valid_pixels), min(semantic_matching_points,len(valid_pixels)))
            for (x,y) in sampled_indices:
                [_,sim]=nearest(src_image_ft, image_ft,x,y)
                similarity+=sim
        elif semantic_matching_strategy==OPENPOSE_POINTS:
            '''
                pose_result=get_poseresult(detector, src_image,H,False,True)
    interm_points=intermediate_points_body(pose_result.body.keypoints,2)
    pose_src_keypoint_list=interm_points+pose_result.body.keypoints
            '''
            try:
                gen_pose_result=get_poseresult(detector,image,H,False,True)
                gen_interm_points=intermediate_points_body(gen_pose_result.body.keypoints,2)
                gen_pose_src_keypoint_list=gen_interm_points+gen_pose_result.body.keypoints

                draw_points(gen_pose_src_keypoint_list,image,"pose_gen")

                gen_keypoint_dict=get_keypoint_dict(gen_pose_src_keypoint_list,rescale)

                for k in keypoint_dict.keys():
                    if k in gen_keypoint_dict:
                        (src_x,src_y)=keypoint_dict[k]
                        (target_x,target_y)=gen_keypoint_dict[k]
                        src_vector=src_image_ft[:,src_x,src_y]
                        target_vector=image_ft[:, target_x,target_y]
                        sim=cos_sim_rescaled(src_vector,target_vector).item()
                        similarity+=sim
            except:
                return 0
                    

        
        return similarity/semantic_matching_points

    

    def get_reward_fn():
        
        def _reward_fn(images, prompts, epoch,):
            print(images[0].size)
            vit_similarities=[0.0 for _ in images]
            face_similarities=[0.0 for _ in images]
            face_probabilities=[0.0 for _ in images]
            pose_probabilities=[0.0 for _ in images]
            rewards=[0.0 for _ in images]
            scores=[0.0 for _ in images]
            style_similarities=[0.0 for _ in images]
            content_similarities=[0.0 for _ in images]
            mse_distances=[0.0 for _ in images]
            fashion_similarities=[0.0 for _ in images]
            dream_similarities=[0.0 for _ in images]
            swin_similarities=[0.0 for _ in images]
            proto_gan_scores=[0.0 for _ in images]
            dift_scores=[0.0 for _ in images]
            time_factor=(float(epoch)/num_epochs)

            removed_images=images
            if  semantic_matching or remove_background_flag:
                removed_images=[remove_background_birefnet(image,birefnet) for image in images]
            print(removed_images)
            if use_face_probs:
                face_probs_weight=initial_face_probs_weight+((final_face_probs_weight-initial_face_probs_weight)*time_factor)
                face_probabilities=[]
                for image in images:
                    boxes,probs=mtcnn.detect(image)
                    if boxes is None or probs is None:
                        face_probabilities.append(0.0)
                    else:
                        face_probabilities.append(face_probs_weight*probs[0])
            if use_pose_probs:
                pose_probs_weight=initial_pose_probs_weight+ ((final_pose_probs_weight-initial_pose_probs_weight) * time_factor)
                pose_probabilities=[]
                for image in images:
                    probs=detector.probs(image)

                    pose_probabilities.append(pose_probs_weight* probs)
                try:
                    accelerator.log({
                        "pose_probs":np.mean(pose_probabilities)
                    })
                except:
                    pass

            if use_face_distance:
                face_weight=initial_face_weight+ ((final_face_weight-initial_face_weight)*time_factor)

                image_face_embeddings=get_face_embedding(images,mtcnn,iresnet,face_margin)
                face_similarities=[
                    cos_sim_rescaled(src_face_embedding,face_embedding)
                    for face_embedding in  image_face_embeddings
                ]
                face_similarities=[
                    face_weight * v for v in face_similarities
                ]
                face_similarities=[
                    v.detach().cpu().numpy() for v in face_similarities
                ]
                try:
                    accelerator.log({
                        "face_distance":np.mean(face_similarities)
                    })
                except:
                    pass
            if use_img_reward:
                img_reward_weight=initial_img_reward_weight+((final_img_reward_weight-initial_img_reward_weight) * time_factor)
                scores=[ir_model.score( prompt.replace(PLACEHOLDER, subject),image) for prompt,image in zip(prompts,images)] #by default IR is normalized to N(0,1) so we rescale
                scores=[0.5 + v/4 for v in scores]
                scores=[s*img_reward_weight for s in scores]
                try:
                    accelerator.log({
                        "score":np.mean(scores)
                    })
                except:
                    pass
            if use_mse:
                mse_reward_weight=initial_mse_weight+((final_mse_weight-initial_mse_weight) *time_factor)
                mse_distances=[
                    -1.0* get_mse_from_src(image) for image in removed_images
                ]
                mse_distances=[mse_reward_weight*m for m in mse_distances]
                try:
                    accelerator.log({
                        "mse_distance":np.mean(mse_distances)
                    })
                except:
                    try:
                        accelerator.log({
                            "mse_distance":np.mean([m.detach().cpu().numpy() for m in mse_distances])
                        })
                    except:
                        pass
            
            if use_fashion_clip:
                fashion_similarities=[
                    cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(image,fashion_clip_processor,fashion_clip_model)) for image in images
                ]
            elif use_fashion_clip_segmented:
                segmented_images=[
                    generate_mask(image,seg_model,accelerator.device) for image in images
                ]
                fashion_similarities=[
                    cos_sim_rescaled(fashion_src_embedding, get_fashion_embedding(seg_image,fashion_clip_processor,fashion_clip_model)) for seg_image in segmented_images
                ]
            if use_fashion_clip or use_fashion_clip_segmented:
                fashion_clip_weight=initial_fashion_clip_weight+((final_fashion_clip_weight-initial_fashion_clip_weight) * time_factor)
                fashion_similarities=[fashion_clip_weight * f for f in fashion_similarities]
                try:
                    fashion_similarities=[f.detach().cpu().numpy() for f in fashion_similarities]
                except:
                    pass
                try:
                    accelerator.log({
                        "fashion_distance":np.mean(fashion_similarities)
                    })
                except:
                    pass
            if use_dift:
                dift_weight=initial_dift_weight+((final_dift_weight-initial_dift_weight)* time_factor)
                dift_scores=[
                    dift_weight * semantic_reward(image, "dift") for image in removed_images
                ]
                try:
                    accelerator.log({
                        "dift_score": np.mean(dift_weight)
                    })
                except:
                    pass
                


            rewards=[
                d+f+s+vs+vc+m+fas+drm+fps+ppb+ssw+pgs+dfs for d,f,s,vs,vc,m,fas,drm,fps,ppb,ssw,pgs,dfs in zip(vit_similarities,face_similarities,
                                                       scores,style_similarities, content_similarities,
                                                       mse_distances,fashion_similarities,dream_similarities,face_probabilities
                                                       ,pose_probabilities,swin_similarities,proto_gan_scores,dift_scores)
            ]
            try:
                rewards=[r.detach().cpu().numpy() for r in rewards]
            except:
                pass
            try:
                accelerator.log({
                    "reward_fn":np.mean(rewards)
                })
            except:
                try:
                    accelerator.log({
                        "reward_fn":np.mean([r.detach().cpu().numpy() for r in rewards])
                    })
                except:
                    pass
            if reward_method==REWARD_PARETO:
                dominant_list=get_dominant_list(vit_similarities,scores,face_similarities,style_similarities, content_similarities)
                for i in range(len(scores)):
                    if i not in dominant_list:
                        rewards[i]=0.0
            print(rewards)
            return rewards
        
        return _reward_fn

    weight_dtype={
            "no":torch.float32,
            "fp16":torch.float16,
            "bf16":torch.bfloat16
        }[accelerator.mixed_precision]
    if method_name == BLIP_DIFFUSION:
        try:
            blip_diffusion_pipe=BlipDiffusionPipeline.from_pretrained(
                "Salesforce/blipdiffusion", torch_dtype=torch.float32)
        except:
            blip_diffusion_pipe=BlipDiffusionPipeline.from_pretrained(
                "Salesforce/blipdiffusion", torch_dtype=torch.float32,force_download=True)
        blip_diffusion_pipe=blip_diffusion_pipe.to(accelerator.device)
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
                height=height,
                width=width,
                ).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==ELITE:
        pass
    elif method_name==RIVAL:
        evaluation_image_list=[
            make_eval_image(inf_config,accelerator,is_half,
                            "CompVis/stable-diffusion-v1-4",
                            evaluation_prompt.format(subject),
                            NEGATIVE,src_image,seed,inner_round,
                            num_inference_steps,editing_early_steps) for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==INSTANT:
        '''evaluation_image_list=[
            instant_generate_one_sample(src_image,evaluation_prompt.format(subject),
                                        NEGATIVE, num_inference_steps, 
                                        accelerator ) for evaluation_prompt in evaluation_prompt_list
        ]'''
        app = FaceAnalysis(name='antelopev2', root='/scratch/jlb638/instant', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        try:
            download_path=snapshot_download("InstantX/InstantID")
        except:
            download_path=snapshot_download("InstantX/InstantID",force_download=True)
        # Path to InstantID models
        face_adapter = f'{download_path}/ip-adapter.bin'
        controlnet_path = f'{download_path}/ControlNetModel'

        # load IdentityNet
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe=pipe.to(accelerator.device)

        # load adapter
        pipe.load_ip_adapter_instantid(face_adapter)

        

        face_info = app.get(cv2.cvtColor(np.array(src_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        pipe.set_ip_adapter_scale(0.8)

        evaluation_image_list=[
            pipe(evaluation_prompt.format(subject),image_embeds=face_emb, image=face_kps, 
                 controlnet_conditioning_scale=0.8,
                 num_inference_steps=num_inference_steps,
                neg_prompt=NEGATIVE,
                height=height,
                width=width,).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==IP_ADAPTER or method_name==FACE_IP_ADAPTER:
        pipeline=StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",safety_checker=None)
        unet=pipeline.unet
        vae=pipeline.vae
        tokenizer=pipeline.tokenizer
        text_encoder=pipeline.text_encoder
        pipeline("none",num_inference_steps=1) #things initialize weird if we dont do it once
        try:
            if method_name==IP_ADAPTER:
                pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
            elif method_name==FACE_IP_ADAPTER:
                pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
        except:
            if method_name==IP_ADAPTER:
                pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin",force_download=True)
            elif method_name==FACE_IP_ADAPTER:
                pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin",force_download=True)

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
                    width=width,
                    height=height,
                    safety_checker=None,
                    ip_adapter_image=src_image).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==DDPO_MULTI or method_name== DDPO or method_name== CONTROL_HACK:
        pipeline=BetterDefaultDDPOStableDiffusionPipeline(
            train_text_encoder,
            train_text_encoder_embeddings,
            train_unet,
            use_lora_text_encoder,
            use_lora=use_lora,
            pretrained_model_name="CompVis/stable-diffusion-v1-4"
        )
        print("len trainable parameters",len(pipeline.get_trainable_layers()))
        prompts=[]
        pretrain_image_list=[]
        face_key=subject
        fashion_key="clothes"
        content_key=f"{subject} wearing clothes"
        style_key=" league of legends style"
        entity_name=subject
        if multi_rewards==None or len(multi_rewards)==0:
            prompts=[entity_name]
            pretrain_image_list=[src_image]
        else:
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
                elif reward==CONTENT_REWARD or reward==BODY_REWARD or reward==DREAM_REWARD:
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
        if method_name==DDPO or method_name==CONTROL_HACK:
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
            subject_key,
            height,
            use_ip_adapter=use_ip_adapter_ddpo,
            ip_adapter_src_image=removed_src
        )
        

        if pretrain:
            if use_ip_adapter_ddpo:
                pipeline.sd_pipeline.set_ip_adapter_scale(0.0)
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
            if use_ip_adapter_ddpo:
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
                    True,
                    ip_adapter_image=removed_src
                )
            else:
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
                    True,
                    ip_adapter_image=None
                )
            torch.cuda.empty_cache()
            trainer.accelerator.free_memory()
        
        if use_ip_adapter_ddpo:
            pipeline.sd_pipeline.set_ip_adapter_scale(0.5)
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

        if method_name ==CONTROL_HACK:
            untrained_pipeline=UnsafeStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
            untrained_pipeline.vae=untrained_pipeline.vae.to(accelerator.device)
            untrained_pipeline.text_encoder=untrained_pipeline.text_encoder.to(accelerator.device)
            untrained_pipeline.unet=untrained_pipeline.unet.to(accelerator.device)

            controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose"
            )

            detector=OpenposeDetectorResize.from_pretrained('lllyasviel/Annotators')
            pipe = StableDiffusionControlNetPipeline(vae=pipeline.sd_pipeline.vae,
                                                    tokenizer=pipeline.sd_pipeline.tokenizer,
                                                    text_encoder=pipeline.sd_pipeline.text_encoder,
                                                    unet=pipeline.sd_pipeline.unet,
                                                    controlnet=controlnet,
                                                    scheduler=pipeline.sd_pipeline.scheduler,safety_checker=None,
                                                    feature_extractor=None,requires_safety_checker=False)
            pipe.to(accelerator.device)
            evaluation_image_list=[]
            for evaluation_prompt in evaluation_prompt_list:
                pose_image=None
                while pose_image==None:
                    try:
                        untrained_image=untrained_pipeline(evaluation_prompt.format(entity_name),num_inference_steps=num_inference_steps,
                                negative_prompt=NEGATIVE,
                                width=width,
                                height=height,
                                safety_checker=None).images[0]
                        pose_image=detector(src_image,untrained_image)
                    except IndexError:
                        print(f"index error for {evaluation_prompt.format(entity_name)} ")
                evaluation_image=pipe(evaluation_prompt.format(entity_name),image=pose_image,num_inference_steps=num_inference_steps).images[0]

                evaluation_image_list.append(evaluation_image)

                
        else:
            if use_ip_adapter_ddpo:
                evaluation_image_list=[
                    pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                            num_inference_steps=num_inference_steps,
                            negative_prompt=NEGATIVE,
                            width=width,
                            height=height,
                            ip_adapter_image=removed_src,
                            safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
                ]
            else:
                evaluation_image_list=[
                    pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                            num_inference_steps=num_inference_steps,
                            negative_prompt=NEGATIVE,
                            width=width,
                            height=height,
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

    
        
        
    elif method_name==CLASSIFIER:
        pipe=UnsafeStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(num_inference_steps)
        evaluation_image_list=[]
        for evaluation_prompt in evaluation_prompt_list:
            evaluation_prompt=evaluation_prompt.format(subject)
            prompt_image=pipe(evaluation_prompt,num_inference_steps=num_inference_steps,
                        negative_prompt=NEGATIVE,
                        width=width,
                        height=height,
                        safety_checker=None).images[0]
            #evaluation_image=classifier_sample(pipe,evaluation_prompt.format(subject),0.1,[removed_src,prompt_image],[evaluation_prompt,subject],negative_prompt=NEGATIVE)
            evaluation_image=classifier_call(pipe, prompt=evaluation_prompt,src_image_list=[removed_src,prompt_image],
                                             src_text_list=[evaluation_prompt,subject], classifier_eta=classifier_eta,
                                             num_inference_steps=num_inference_steps,negative_prompt=NEGATIVE).images[0]
            evaluation_image_list.append(evaluation_image)
            
            



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