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
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from text_embedding_helpers import prepare_textual_inversion
from trl import DDPOConfig
from pareto import get_dominant_list
import random
from dpok_pipeline import DPOKPipeline
from dpok_scheduler import DPOKDDIMScheduler
from dpok_reward import ValueMulti
from dpok_helpers import _get_batch, _collect_rollout,  _trim_buffer,_train_value_func,TrainPolicyFuncData, _train_policy_func
from facenet_pytorch import MTCNN
from experiment_helpers.elastic_face_iresnet import get_face_embedding,get_iresnet_model
from experiment_helpers.measuring import get_metric_dict,get_vit_embeddings
from experiment_helpers.better_vit_model import BetterViTModel

def cos_sim_rescaled(vector_i,vector_j,return_np=False):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    try:
        result= cos(vector_i,vector_j) *0.5 +0.5
    except TypeError:
        result= cos(torch.tensor(vector_i),torch.tensor(vector_j)) *0.5 +0.5
    if return_np:
        return result.cpu().numpy()
    return result

def center_crop_to_min_dimension(image:Image)->Image:
    width, height = image.size
    min_dimension = min(width, height)
    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = (width + min_dimension) // 2
    bottom = (height + min_dimension) // 2
    return image.crop((left, top, right, bottom))

def evaluate_one_sample(
        method_name:str,
        src_image: Image,
        text_prompt:str,
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
        n_normalization_images:int
)->dict:
    os.makedirs(image_dir,exist_ok=True)
    method_name=method_name.strip()
    src_image=center_crop_to_min_dimension(src_image)
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
    def get_reward_fn(pipeline:StableDiffusionPipeline,entity_name:str):
        vit_src_image_embedding_list,vit_src_style_embedding_list,vit_src_content_embedding_list=get_vit_embeddings(
            vit_processor,vit_model,[src_image],False
        )
        vit_src_image_embedding=vit_src_image_embedding_list[0]
        vit_src_style_embedding=vit_src_style_embedding_list[0]
        vit_src_content_embedding=vit_src_content_embedding_list[0]
        normalization_image_list=[pipeline(entity_name,num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for _ in range(n_normalization_images)]
        normalization_vit_embedding_list,normalization_vit_style_embedding_list, normalization_vit_content_embedding_list=get_vit_embeddings(
                    vit_processor,vit_model,normalization_image_list,False
                )
        normalization_vit_similarities=[cos_sim_rescaled(vit_src_image_embedding,embedding)
                        for embedding in normalization_vit_embedding_list]
        vit_mean=np.mean(normalization_vit_similarities)
        vit_std=np.std(normalization_vit_similarities)
        normalization_style_similarities=[
                    cos_sim_rescaled(vit_src_style_embedding, style_embedding)
                    for style_embedding in normalization_vit_style_embedding_list
                ]
        vit_style_mean=np.mean(normalization_style_similarities)
        vit_style_std=np.std(normalization_style_similarities)
        normalization_content_similarities=[
                    cos_sim_rescaled(vit_src_content_embedding,content_embedding)
                    for content_embedding  in normalization_vit_content_embedding_list
                ]
        vit_content_mean=np.mean(normalization_content_similarities)
        vit_content_std=np.std(normalization_content_similarities)
        normalization_image_face_embeddings=get_face_embedding(normalization_image_list,mtcnn,iresnet,face_margin)
        normalization_face_similarities=[
                        cos_sim_rescaled(src_face_embedding,face_embedding)
                        for face_embedding in  normalization_image_face_embeddings
                    ]
        try:
            face_mean=np.mean(normalization_face_similarities)
            face_std=np.std(normalization_face_similarities)
        except RuntimeError:
            face_mean=np.mean([n.detach().cpu().numpy() for n in normalization_face_similarities])
            face_std=np.std([n.detach().cpu().numpy() for n in normalization_face_similarities])

        normalization_image_scores=[ir_model.score(entity_name, image) for image in normalization_image_list]
        image_score_mean=np.mean(normalization_image_scores)
        image_score_std=np.std(normalization_image_scores)
        def _reward_fn(images, prompts, epoch,):
            print(images)
            vit_similarities=[0.0 for _ in images]
            face_similarities=[0.0 for _ in images]
            rewards=[0.0 for _ in images]
            scores=[0.0 for _ in images]
            style_similarities=[0.0 for _ in images]
            content_similarities=[0.0 for _ in images]
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
                if normalize_rewards_individually:
                    vit_similarities=[
                        (v-vit_mean)/vit_std for v in vit_similarities
                    ]
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
                if normalize_rewards_individually:
                    content_similarities=[
                        (v -vit_content_mean)/vit_content_std for v in content_similarities
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
                if normalize_rewards_individually:
                    style_similarities=[
                        (v-vit_style_mean)/vit_style_std for v in style_similarities
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
                    if normalize_rewards_individually:
                        face_similarities=[
                            (v-face_mean)/face_std for v in face_similarities
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
                if normalize_rewards_individually:
                    scores=[
                        (v-image_score_mean)/image_score_std for v in scores
                    ]
                else:
                    scores=[0.5 + v/4 for v in scores]
                scores=[s*img_reward_weight for s in scores]
                wandb_tracker.log({
                    "score":np.mean(scores)
                })
            rewards=[
                d+f+s+vs+vc for d,f,s,vs,vc in zip(vit_similarities,face_similarities,scores,style_similarities, content_similarities)
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
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
        ]
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
    elif method_name==DDPO:
        pipeline=BetterDefaultDDPOStableDiffusionPipeline(
            train_text_encoder,
            train_text_encoder_embeddings,
            train_unet,
            use_lora_text_encoder,
            use_lora=use_lora,
            pretrained_model_name="runwayml/stable-diffusion-v1-5"
        )
        entity_name=subject
        if train_text_encoder_embeddings:
            entity_name=PLACEHOLDER
            pipeline.sd_pipeline.tokenizer, pipeline.sd_pipeline.text_encoder,placeholder_token_ids=prepare_textual_inversion(PLACEHOLDER,pipeline.sd_pipeline.tokenizer, pipeline.sd_pipeline.text_encoder)
        config=DDPOConfig(
            num_epochs=num_epochs,
            train_gradient_accumulation_steps=train_gradient_accumulation_steps,
            sample_num_steps=num_inference_steps,
            sample_batch_size=batch_size,
            train_batch_size=batch_size,
            sample_num_batches_per_epoch=samples_per_epoch,
            mixed_precision=mixed_precision,
            tracker_project_name="ddpo-personalization",
            log_with="wandb",
            accelerator_kwargs={
                #"project_dir":args.output_dir
            },
            #project_kwargs=project_kwargs
        )
        
        def prompt_fn():
            return random.choice(prompt_list).format(entity_name),{}

        image_samples_hook=get_image_sample_hook(image_dir)
        _reward_fn=get_reward_fn(pipeline.sd_pipeline,entity_name)
        def reward_fn(images, prompts, epoch,prompt_metadata):
            return _reward_fn(images, prompts, epoch),{}
        trainer = BetterDDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook
        )
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
        del pipeline
    elif method_name==DPOK:
        reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        reward_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        reward_tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        weight_dtype={
            "no":torch.float32,
            "fp16":torch.float16,
            "bf16":torch.bfloat16
        }[accelerator.mixed_precision]
        pipeline=DPOKPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", #torch_dtype=weight_dtype
        )
        pipeline("do this to help it instantiate things",num_inference_steps=1)
        pipeline.safety_checker=None
        unet=pipeline.unet
        pipeline.scheduler = DPOKDDIMScheduler.from_config(pipeline.scheduler.config)
        print('pipeline.scheduler.config',pipeline.scheduler.config)
        unet_copy = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="unet",)
        text_encoder=pipeline.text_encoder
        tokenizer=pipeline.tokenizer
        pipeline.text_encoder.to(accelerator.device) #, dtype=weight_dtype)
        pipeline.vae.to(accelerator.device) #, dtype=weight_dtype)
        pipeline.unet.to(accelerator.device) #, dtype=weight_dtype)
        unet_copy.to(accelerator.device) #, dtype=weight_dtype)
        pipeline.scheduler.to(accelerator.device) #, weight_dtype)
        reward_clip_model.to(accelerator.device)
        #vit_model.to(accelerator.device)
        #vit_processor.to(accelerator.device)
        #reward_clip_model.to(accelerator.device)
        for model in [reward_clip_model, vit_model, unet_copy]:
            model.requires_grad_(False)

        pipeline.setup_parameters(
            train_text_encoder,
                 train_text_encoder_embeddings,
                 train_unet,
                  use_lora_text_encoder,
                  use_lora
        )
        entity_name=subject
        if train_text_encoder_embeddings:
            entity_name=PLACEHOLDER
        trainable_parameters=pipeline.get_trainable_layers()
        #print(trainable_parameters)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=0.00001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=0.00000001)
        
        def _my_data_iterator(data,batch_size):
            random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                yield batch

        data_iterator=_my_data_iterator([p.format(entity_name) for p in prompt_list], g_batch_size)
        data_iter_loader = iter(data_iterator)


        '''lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps,
        )'''
        value_function = ValueMulti(num_inference_steps, (4, 64, 64))
        value_optimizer = torch.optim.AdamW(value_function.parameters(), lr=v_lr)
        value_function, value_optimizer = accelerator.prepare(
        value_function, value_optimizer
        )
        trainable_parameters, optimizer, data_iter_loader = accelerator.prepare(
        trainable_parameters, optimizer, data_iter_loader )

        total_batch_size = (
        batch_size
        * accelerator.num_processes
            * train_gradient_accumulation_steps)
        


          # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(0, max_train_steps),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        def _map_cpu(x):
            return x.cpu()
        
        state_dict = {}
        state_dict["prompt"] = []
        state_dict["state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
        state_dict["next_state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
        state_dict["timestep"] = _map_cpu(torch.LongTensor())
        state_dict["final_reward"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
        state_dict["unconditional_prompt_embeds"] = _map_cpu(
            torch.FloatTensor().to(weight_dtype)
        )
        state_dict["guided_prompt_embeds"] = _map_cpu(
            torch.FloatTensor().to(weight_dtype)
        )
        state_dict["txt_emb"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
        state_dict["log_prob"] = _map_cpu(torch.FloatTensor().to(weight_dtype))

        def get_text_emb(prompts):
            inputs = reward_tokenizer(
                    prompts,
                    max_length=tokenizer.model_max_length,
                    padding="do_not_pad",
                    truncation=True,
            )
            input_ids = inputs.input_ids
            padded_tokens = reward_tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            )

            txt_emb = reward_clip_model.get_text_features(
                input_ids=padded_tokens.input_ids.to(accelerator.device).unsqueeze(0)
            )
            return txt_emb.squeeze(0)
        
        _reward_fn=get_reward_fn(pipeline,entity_name)
        def reward_fn(images, prompts, step):
            txt_emb=get_text_emb(prompts)
            return torch.tensor(_reward_fn(images, prompts, step)), txt_emb
        
        trainable_list=[]
        if train_text_encoder or train_text_encoder_embeddings:
            trainable_list.append(text_encoder)
        if train_unet:
            trainable_list.append(unet)
        policy_steps=train_gradient_accumulation_steps*p_step
        #with accelerator.autocast():
        def _single_value_epoch(step,v_batch_size=v_batch_size,
                                v_step=v_step,
                                g_batch_size=g_batch_size,
                                num_samples=num_samples):
            unet.eval()
            batch=_get_batch(
                data_iter_loader,
                _my_data_iterator,
                prompt_list,
                g_batch_size,
                num_samples,
                accelerator
            )
            _collect_rollout(g_step, pipeline,False,batch, reward_fn,state_dict,step,num_inference_steps) #def _collect_rollout(g_step, pipe, is_ddp, batch, calculate_reward, state_dict):
            _trim_buffer(buffer_size, state_dict)

            #value learning
            value_optimizer.zero_grad()
            total_val_loss=0.0
            v_batch_size=min(v_batch_size,num_inference_steps) #otherwise we will have to sample_size > population
            for _v in range(v_step):
                if _v< v_step-1:
                    with accelerator.no_sync(value_function):
                        total_val_loss+=_train_value_func(value_function, state_dict, accelerator, v_batch_size,v_step)
                else:
                    total_val_loss+=_train_value_func(value_function, state_dict, accelerator, v_batch_size,v_step)
            value_optimizer.step()
            value_optimizer.zero_grad()
            if accelerator.is_main_process:
                print("value_loss", total_val_loss)
                accelerator.log({"value_loss": total_val_loss}, )
            del total_val_loss
            torch.cuda.empty_cache()
        for v_epoch in range(value_epochs):
            _single_value_epoch(0)
        for count in range(0, max_train_steps//p_step):
            _single_value_epoch(count)

            #poloucy learning
            tpfdata = TrainPolicyFuncData()
            p_batch_size=min(p_batch_size,num_inference_steps) #otherwise we will have to sample_size > population
            for _p in range(p_step):
                optimizer.zero_grad()
                for accum_step in range(train_gradient_accumulation_steps):
                    if accum_step<train_gradient_accumulation_steps-1:
                        with accelerator.no_sync(unet):
                            with accelerator.no_sync(text_encoder):
                                _train_policy_func(
                                    p_batch_size,
                                    ratio_clip,
                                    reward_weight,
                                    kl_warmup,
                                    kl_weight,
                                    train_gradient_accumulation_steps,
                                    state_dict,
                                    pipeline,
                                    unet_copy,
                                    False,
                                    count,
                                    policy_steps,
                                    accelerator,
                                    tpfdata,
                                    value_function,
                                    num_inference_steps
                                )
                    else:
                        _train_policy_func(
                                p_batch_size,
                                ratio_clip,
                                reward_weight,
                                kl_warmup,
                                kl_weight,
                                train_gradient_accumulation_steps,
                                state_dict,
                                pipeline,
                                unet_copy,
                                False,
                                count,
                                policy_steps,
                                accelerator,
                                tpfdata,
                                value_function,
                                num_inference_steps
                        )
                    if accelerator.sync_gradients:
                        norm = accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                    tpfdata.tot_grad_norm += norm.item() / p_step
                    optimizer.step()
                    if accelerator.is_main_process:
                        print(f"count: [{count} / {max_train_steps // p_step}]")
                        print("train_reward", torch.mean(state_dict["final_reward"]).item())
                        accelerator.log(
                        {"train_reward": torch.mean(state_dict["final_reward"]).item()})
                        print("grad norm", tpfdata.tot_grad_norm, "ratio", tpfdata.tot_ratio)
                        print("kl", tpfdata.tot_kl, "p_loss", tpfdata.tot_p_loss)
                        accelerator.log({"grad norm": tpfdata.tot_grad_norm}, )
                        accelerator.log({"ratio": tpfdata.tot_ratio}, )
                        accelerator.log({"kl": tpfdata.tot_kl}, )
                        accelerator.log({"p_loss": tpfdata.tot_p_loss}, )
                    torch.cuda.empty_cache()
            validation_prompt_list=[entity_name]
            generator=torch.Generator(device=accelerator.device).manual_seed(123)
            validation_image_list=[pipeline(validation_prompt,num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    generator=generator,
                    safety_checker=None).images[0] for validation_prompt in validation_prompt_list ]
            for i,image in enumerate(validation_image_list):
                path=f"{image_dir}/{i}.png"
                image.save(path)
                try:
                    accelerator.log({
                        f"validation_img_dpok":wandb.Image(path)
                    })
                except:
                    print(f"couldnt find {path}")
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        del unet, unet_copy, value_function, value_optimizer, optimizer, state_dict,reward_clip_model, vit_model, data_iterator,pipeline

    
    elif method_name in [CHOSEN,CHOSEN_K,CHOSEN_K_STYLE,CHOSEN_STYLE]:
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
        pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None)
        unet=pipeline.unet
        vae=pipeline.vae
        tokenizer=pipeline.tokenizer
        text_encoder=pipeline.text_encoder
        for model in [vae,unet,text_encoder]:
            model.requires_grad_(False)
        config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none")
        unet = get_peft_model(unet, config)
        unet.train()
        unet.print_trainable_parameters()
        trainable_parameters=[]
        for model in [vae,unet,text_encoder]:
            trainable_parameters+=[p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=0.00000001)
        unet,text_encoder,vae,tokenizer = accelerator.prepare(
            unet,text_encoder,vae,tokenizer
        )
        n_clusters=n_img_chosen // target_cluster_size
        if method_name in [CHOSEN_K, CHOSEN]:
            image_list=[
                pipeline(text_prompt,negative_prompt=NEGATIVE,num_inference_steps=num_inference_steps,safety_checker=None).images[0] for _ in range(n_img_chosen)]
        elif method_name in [CHOSEN_K_STYLE, CHOSEN_STYLE]:
            image_list=[
                generate_with_style(pipeline,text_prompt=text_prompt,negative_prompt=NEGATIVE,num_inference_steps=num_inference_steps,src_image=src_image) for _ in range(n_img_chosen)
            ]
        print("generated initial sets of images")
        last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
        init_dist=get_init_dist(last_hidden_states)
        pairwise_distances=init_dist
        iteration=0
        while pairwise_distances>=convergence_scale*init_dist and iteration<10:
            iteration+=1
            if method_name in [CHOSEN, CHOSEN_STYLE]:
                valid_image_list, centroid_distances=get_best_cluster_kmeans(image_list, n_clusters, min_cluster_size, vit_processor, vit_model)
            elif method_name in [CHOSEN_K, CHOSEN_K_STYLE]:
                valid_image_list=get_top_k(src_image, image_list, vit_processor, vit_model,target_cluster_size)
            text_prompt_list=[text_prompt]*len(valid_image_list)
            pipeline=loop(
                valid_image_list,
                text_prompt_list,
                pipeline,
                0,
                optimizer,
                accelerator,
                1,
                num_inference_steps,
                size=512,
                train_batch_size=2,
                noise_offset=0.0,
                max_grad_norm=1.0
            )
            if method_name in [CHOSEN_K, CHOSEN]:
                image_list=[
                    pipeline(text_prompt,negative_prompt=NEGATIVE,num_inference_steps=num_inference_steps,safety_checker=None).images[0] for _ in range(n_img_chosen)]
            elif method_name in [CHOSEN_K_STYLE, CHOSEN_STYLE]:
                image_list=[
                    generate_with_style(pipeline,text_prompt=text_prompt,negative_prompt=NEGATIVE,num_inference_steps=num_inference_steps,src_image=src_image) for _ in range(n_img_chosen)
                ]
            last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
            init_dist=get_init_dist(last_hidden_states)
            pairwise_distances=init_dist
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(text_prompt),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
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