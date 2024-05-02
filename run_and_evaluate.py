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
from PIL import Image
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
from better_ddpo_trainer import BetterDDPOTrainer
from text_embedding_helpers import prepare_textual_inversion
from trl import DDPOConfig
from pareto import get_dominant_list
import random
from dpok_pipeline import DPOKPipeline
from dpok_scheduler import DPOKDDIMScheduler
from dpok_reward import ValueMulti
from dpok_helpers import _get_batch, _collect_rollout,  _trim_buffer,_train_value_func,TrainPolicyFuncData, _train_policy_func
    
def cos_sim(vector_i,vector_j)->float:
    return np.dot(vector_i,vector_j)/(norm(vector_i)*norm(vector_j))

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
        samples_per_epoch
)->dict:
    method_name=method_name.strip()
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")
    ir_model.requires_grad_(False)
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
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
        vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model)[0]

        if reward_method==REWARD_NORMAL:
            def reward_fn(images, prompts, epoch,prompt_metadata):
                print("vit_src_image_embedding")
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model)
                print("image_vit_embeddings")
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model,False)
                print('vit_src_image_embedding',vit_src_image_embedding)
                print('image_vit_embeddings ',image_vit_embeddings)
                print('embedding',image_vit_embeddings[0])
                try:
                    print('vit_src_image_embedding.size()',vit_src_image_embedding.size())
                except:
                    pass
                try:
                    print('image_vit_embeddings size',image_vit_embeddings.size())
                except:
                    pass
                try:
                    print('embedding size',image_vit_embeddings[0].size())
                except:
                    pass
                distances=[ cos_sim(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                return distances, {}
        elif reward_method==REWARD_TIME:
            def reward_fn(images, prompts, epoch,prompt_metadata):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model)[0]
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model)
                distances=[ cos_sim(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                images=pipeline.sd_pipeline.image_processor.postprocess(images)
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                score_weight=float(epoch)/num_epochs
                distance_weight=1.0-score_weight
                distances=[distance_weight*d for d in distances]
                scores=[score_weight*s for s in scores]
                return [d+s for d,s in zip(distances,scores)],{}
        elif reward_method==REWARD_PARETO: #todo
            def reward_fn(images, prompts, epoch,prompt_metadata):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model)[0]
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model)
                distances=[ cos_sim(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                images=pipeline.sd_pipeline.image_processor.postprocess(images)
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                dominant_list=get_dominant_list(distances,scores)
                #score_weight=float(epoch)/num_epochs
                #distance_weight=1.0-score_weight
                #distances=distance_weight*distances
                #scores=score_weight*scores
                rewards=[]
                for i in range(len(scores)):
                    if i in dominant_list:
                        rewards.append(scores[i]+distances[i])
                    else:
                        rewards.append(0.0)
                return rewards,{}
        elif reward_method==REWARD_PARETO_TIME:
            def reward_fn(images, prompts, epoch,prompt_metadata):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model)[0]
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model)
                distances=[ cos_sim(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                images=pipeline.sd_pipeline.image_processor.postprocess(images)
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                dominant_list=get_dominant_list(distances,scores)
                score_weight=float(epoch)/num_epochs
                distance_weight=1.0-score_weight
                distances=[distance_weight*d for d in distances]
                scores=[score_weight*s for s in scores]
                rewards=[]
                for i in range(len(scores)):
                    if i in dominant_list:
                        rewards.append(scores[i]+distances[i])
                    else:
                        rewards.append(0.0)
                return rewards,{}


        def image_samples_hook(*args):
            return

        trainer = BetterDDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook
        )
        print(f"acceleerate device {trainer.accelerator.device}")
        tracker=trainer.accelerator.get_tracker("wandb").run
        trainer.train(retain_graph=True)
        evaluation_image_list=[
            pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==DPOK: #TODO
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
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
        "runwayml/stable-diffusion-v1-5", torch_dtype=weight_dtype
        )
        unet=pipeline.unet
        pipeline.scheduler = DPOKDDIMScheduler.from_config(pipeline.scheduler.config)
        unet_copy = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="unet",)
        text_encoder=pipeline.text_encoder
        tokenizer=pipeline.tokenizer
        pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
        pipeline.vae.to(accelerator.device, dtype=weight_dtype)
        pipeline.unet.to(accelerator.device, dtype=weight_dtype)
        unet_copy.to(accelerator.device, dtype=weight_dtype)
        pipeline.scheduler.to(accelerator.device, weight_dtype)
        vit_model.to(accelerator.device)
        #vit_processor.to(accelerator.device)
        reward_clip_model.to(accelerator.device)
        for model in [reward_clip_model, vit_model, unet_copy]:
            model.requires_grad_(False)
        vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model,False)[0].to(accelerator.device)

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
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=0.00000001)
        
        def _my_data_iterator(data,batch_size):
            random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                yield batch

        data_iterator=_my_data_iterator([p.format(PLACEHOLDER) for p in prompt_list], g_batch_size)
        data_iter_loader = iter(data_iterator)


        '''lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps,
        )'''
        value_function = ValueMulti(50, (4, 64, 64))
        value_optimizer = torch.optim.AdamW(value_function.parameters(), lr=v_lr)
        value_function, value_optimizer = accelerator.prepare(
        value_function, value_optimizer
        )
        value_function, value_optimizer = accelerator.prepare(
            value_function, value_optimizer)
        trainable_parameters, optimizer = accelerator.prepare(
        trainable_parameters, optimizer)

        total_batch_size = (
        batch_size
        * accelerator.num_processes
            * train_gradient_accumulation_steps)
        max_train_steps=samples_per_epoch*num_epochs


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

        if reward_method==REWARD_NORMAL:
            def reward_fn(images, prompts, step):
                print("vit_src_image_embedding")
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model,False)
                print("image_vit_embeddings")
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model,False)
                print('vit_src_image_embedding',vit_src_image_embedding)
                print('image_vit_embeddings ',image_vit_embeddings)
                print('embedding',image_vit_embeddings[0])
                try:
                    print('vit_src_image_embedding.size()',vit_src_image_embedding.size())
                except:
                    pass
                try:
                    print('image_vit_embeddings size',image_vit_embeddings.size())
                except:
                    pass
                try:
                    print('embedding size',image_vit_embeddings[0].size())
                except:
                    pass
                distances=[ cos(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                txt_emb=get_text_emb(prompts)
                return torch.stack(distances), txt_emb
        elif reward_method==REWARD_TIME:
            def reward_fn(images, prompts, step):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model,False)
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model,False)
                distances=[ cos(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                #images=pipeline.image_processor.postprocess(torch.tensor(images))
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                score_weight=float(step)/max_train_steps
                distance_weight=1.0-score_weight
                distances=[distance_weight*d for d in distances]
                scores=[score_weight*s for s in scores]
                txt_emb=get_text_emb(prompts)
                return torch.stack([d+s for d,s in zip(distances,scores)]),txt_emb
        elif reward_method==REWARD_PARETO: #todo
            def reward_fn(images, prompts, step):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model,False)
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model,False)
                distances=[ cos(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                #images=pipeline.image_processor.postprocess(torch.tensor(images))
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                dominant_list=get_dominant_list(distances,scores)
                #score_weight=float(epoch)/num_epochs
                #distance_weight=1.0-score_weight
                #distances=distance_weight*distances
                #scores=score_weight*scores
                rewards=[]
                for i in range(len(scores)):
                    if i in dominant_list:
                        rewards.append(scores[i]+distances[i])
                    else:
                        rewards.append(0.0)
                txt_emb=get_text_emb(prompts)
                return torch.stack(rewards),txt_emb
        elif reward_method==REWARD_PARETO_TIME:
            def reward_fn(images, prompts, step):
                vit_src_image_embedding=get_hidden_states([src_image],vit_processor, vit_model,False)
                image_vit_embeddings=get_hidden_states(images,vit_processor, vit_model,False)
                distances=[ cos(vit_src_image_embedding,embedding)
                           for embedding in image_vit_embeddings]
                #images=pipeline.image_processor.postprocess(torch.tensor(images))
                scores=[
                    0.5+ ir_model.score( prompt.replace(PLACEHOLDER, subject),image)/2.0 for prompt,image in zip(prompts,images)
                ] #by default its normalized to have mean=0, std dev=1
                dominant_list=get_dominant_list(distances,scores)
                score_weight=score_weight=float(step)/max_train_steps
                distance_weight=1.0-score_weight
                distances=[distance_weight*d for d in distances]
                scores=[score_weight*s for s in scores]
                rewards=[]
                for i in range(len(scores)):
                    if i in dominant_list:
                        rewards.append(scores[i]+distances[i])
                    else:
                        rewards.append(0.0)
                txt_emb=get_text_emb(prompts)
                return torch.stack(rewards),txt_emb
        trainable_list=[]
        if train_text_encoder or train_text_encoder_embeddings:
            trainable_list.append(text_encoder)
        if train_unet:
            trainable_list.append(unet)
        policy_steps=train_gradient_accumulation_steps*p_step
        for count in range(0, max_train_steps//p_step):
            unet.eval()
            batch=_get_batch(
                data_iter_loader,
                _my_data_iterator,
                prompt_list,
                g_batch_size,
                num_samples,
                accelerator
            )
            _collect_rollout(g_step, pipeline,False,batch, reward_fn,state_dict,count) #def _collect_rollout(g_step, pipe, is_ddp, batch, calculate_reward, state_dict):
            _trim_buffer(buffer_size, state_dict)

            #value learning
            value_optimizer.zero_grad()
            total_val_loss=0.0
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
                accelerator.log({"value_loss": total_val_loss}, step=count)
            del total_val_loss
            torch.cuda.empty_cache()

            #poloucy learning
            tpfdata = TrainPolicyFuncData()
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
                                    value_function
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
                                value_function
                        )
                    if accelerator.sync_gradients:
                        norm = accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                    tpfdata.tot_grad_norm += norm.item() / p_step
                    optimizer.step()
                    if accelerator.is_main_process:
                        print(f"count: [{count} / {max_train_steps // p_step}]")
                        print("train_reward", torch.mean(state_dict["final_reward"]).item())
                        accelerator.log(
                        {"train_reward": torch.mean(state_dict["final_reward"]).item()},
                        step=count,
                        )
                        print("grad norm", tpfdata.tot_grad_norm, "ratio", tpfdata.tot_ratio)
                        print("kl", tpfdata.tot_kl, "p_loss", tpfdata.tot_p_loss)
                        accelerator.log({"grad norm": tpfdata.tot_grad_norm}, step=count)
                        accelerator.log({"ratio": tpfdata.tot_ratio}, step=count)
                        accelerator.log({"kl": tpfdata.tot_kl}, step=count)
                        accelerator.log({"p_loss": tpfdata.tot_p_loss}, step=count)
                    torch.cuda.empty_cache()
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]            

    
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

    print(evaluation_image_list)
    #METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD]
    metric_dict={}
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_inputs=clip_processor(text=evaluation_prompt_list, images=evaluation_image_list+[src_image], return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    text_embed_list=outputs.text_embeds.detach().numpy()
    image_embed_list=outputs.image_embeds.detach().numpy()[:-1]
    src_image_embed=outputs.image_embeds.detach().numpy()[-1]
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")
    ir_model.requires_grad_(False)

    identity_consistency_list=[]
    target_similarity_list=[]
    prompt_similarity_list=[]
    for i in range(len(image_embed_list)):
        image_embed=image_embed_list[i]
        text_embed=text_embed_list[i]
        target_similarity_list.append(cos_sim(image_embed,src_image_embed))
        prompt_similarity_list.append(cos_sim(image_embed, text_embed))
        for j in range(i+1, len(image_embed_list)):
            #print(i,j)
            vector_j=image_embed_list[j]
            sim=cos_sim(image_embed,vector_j)
            identity_consistency_list.append(sim)


    metric_dict[IDENTITY_CONSISTENCY]=np.mean(identity_consistency_list)
    metric_dict[TARGET_SIMILARITY]=np.mean(target_similarity_list)
    metric_dict[PROMPT_SIMILARITY]=np.mean(prompt_similarity_list)
    #for evaluation_image,evaluation_prompt in zip(evaluation_image_list, evaluation_prompt_list):
    metric_dict[IMAGE_REWARD]=np.mean(
        [ir_model.score(evaluation_prompt.format(subject),evaluation_image) for evaluation_prompt,evaluation_image in zip(evaluation_prompt_list, evaluation_image_list) ]
    )
    aesthetic_scorer=get_aesthetic_scorer()
    metric_dict[AESTHETIC_SCORE]=np.mean(
        [aesthetic_scorer(evaluation_image).cpu().numpy()[0] for evaluation_image in evaluation_image_list]
    )
    for metric in METRIC_LIST:
        if metric not in metric_dict:
            metric_dict[metric]=0.0
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    unet=vae=tokenizer=text_encoder=image_encoder=blip_diffusion_pipe=pipeline=clip_model=optimizer=None
    return metric_dict,evaluation_image_list