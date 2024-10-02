import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import time
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model,save_model
from experiment_helpers.training import train_unet,train_unet_single_prompt
from datasets import load_dataset
import torch
import shutil
from static_globals import *
from peft.utils import get_peft_model_state_dict
import wandb
from peft import LoraConfig
from PIL import Image
import random

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="unet")
parser.add_argument("--save_dir",type=str,default="/scratch/jlb638/unet/")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/unet_images/")
parser.add_argument("--load_saved",action="store_true")
parser.add_argument("--use_lora",action="store_true")
parser.add_argument("--pretrained_src",type=str,default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--epochs",type=int,default=5)
parser.add_argument("--hf_repo",type=str,default="jlbaker361/league_unet")
parser.add_argument("--pretrained_vae",action="store_true")
parser.add_argument("--vae_id",type=str,default="jlbaker361/league_vae_25")
parser.add_argument("--resize",type=int,default=512)
parser.add_argument("--lora_r", default=16, type=int)
parser.add_argument("--lora_alpha", default=32, type=int)
parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")

def flip_images_horizontally(image_list):
    # List to store flipped images
    flipped_images = []
    
    for image in image_list:
        # Flip the image horizontally
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_images.append(flipped_image)
    
    return flipped_images

def main(args):
    os.makedirs(args.save_dir,exist_ok=True)
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.batch_size)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_src)
    if args.pretrained_vae:
        vae_weight_path = hf_hub_download(repo_id=args.vae_id, filename="vae/diffusion_pytorch_model.safetensors")
        load_model(pipeline.vae,vae_weight_path)

    #pipeline=pipeline.to(accelerator.device)
    for model in [pipeline.vae, pipeline.text_encoder]:
        model.eval()
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        pipeline.unet.add_adapter(lora_config)
    pipeline.unet=pipeline.unet.to(accelerator.device)
    pipeline.vae=pipeline.vae.to(accelerator.device)
    pipeline.unet.train()
    pipeline=pipeline.to(accelerator.device)

    evaluation_prompt_list=[
        " {} ",
        " happy {} ",
        " {} eating a burger ",
        " {} dancing"
    ]

    evaluation_images=[
        pipeline(prompt.format("character"),num_inference_steps=30,
                 negative_prompt=NEGATIVE,height=args.resize,width=args.resize).images[0] for prompt in evaluation_prompt_list
    ]
    for image in evaluation_images:
        try:
            accelerator.log({
                "image_before":wandb.Image(image)
            })
        except:
            tmp="temp.png"
            image.save(tmp)
            accelerator.log({
                "image_before":wandb.Image(tmp)
            })

    param_groups = [p for p in pipeline.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=0.00001, weight_decay=5e-2, betas=(0.9, 0.95))
    training_image_list=[row["splash"].resize((args.resize,args.resize)) for row in load_dataset(args.dataset,split="train")]
    training_image_list=flip_images_horizontally(training_image_list)
    random.shuffle(training_image_list)
    training_prompt_list=["character" for _ in training_image_list]
    pipeline=train_unet_single_prompt(pipeline,
                        args.epochs,
                        training_image_list,
                        "character",
                        optimizer,
                        False,
                        " ",
                        1,
                        1.0,
                        "character",
                        accelerator,
                        30,
                        0.0,
                        True
                        )
    
    if args.use_lora:
        unet_lora_layers = get_peft_model_state_dict(pipeline.unet)
        pipeline.save_lora_weights(args.save_dir,unet_lora_layers)
    pipeline.save_pretrained(args.save_dir, push_to_hub=True, repo_id=args.hf_repo)
    evaluation_prompt_list=[
        " {} ",
        " happy {} ",
        " {} eating a burger ",
        " {} dancing"
    ]
    evaluation_images=[
        pipeline(prompt.format("character"),num_inference_steps=30,
                 negative_prompt=NEGATIVE,height=args.resize,width=args.resize).images[0] for prompt in evaluation_prompt_list
    ]
    for image in evaluation_images:
        try:
            accelerator.log({
                "image_after":wandb.Image(image)
            })
        except:
            tmp="temp.png"
            image.save(tmp)
            accelerator.log({
                "image_after":wandb.Image(tmp)
            })
    '''pair_dict={
      #  "tokenizer":pipeline.tokenizer,
        "vae":pipeline.vae,
        "unet":pipeline.unet,
        "text_encoder":pipeline.text_encoder
    }
    for name,model in pair_dict.items():
        new_dir=os.makedirs(os.path.join(args.save_dir,name),exist_ok=True)
        file_path=os.path.join(args.save_dir,name,"diffusion_pytorch_model.safetensors")
        save_model(model,file_path)

        original_config_file=hf_hub_download(repo_id=args.pretrained_src, filename=f"{name}/config.json")
        new_config_file_path=os.path.join(args.save_dir,name,"config.json")

        shutil.copy(original_config_file,new_config_file_path)
        
        
    
    #copy tokenizer files
    for txt_file in ["merges.txt","special_tokens_map.json","tokenizer_config.json","vocab.json"]:
        new_dir=os.makedirs(os.path.join(args.save_dir,"tokenizer"),exist_ok=True)
        original_path=hf_hub_download(repo_id=args.pretrained_src, filename=f"tokenizer/{txt_file}")
        new_path=os.path.join(args.save_dir,"tokenizer",txt_file)
        shutil.copy(original_path, new_path)
    #copy scheduler files
    new_dir=os.makedirs(os.path.join(args.save_dir,"scheduler"),exist_ok=True)
    scheduler_config_src=hf_hub_download(repo_id=args.pretrained_src, filename=f"scheduler/scheduler_config.json")
    new_scheduler_config=os.path.join(args.save_dir,"scheduler","scheduler_config.json")
    shutil.copy(scheduler_config_src,new_scheduler_config)
                                      
    model_index_src=hf_hub_download(repo_id=args.pretrained_src, filename=f"model_index.json")
    new_model_index=os.path.join(args.save_dir,"model_index.json")'''
    return

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")