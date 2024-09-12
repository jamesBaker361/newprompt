import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import time
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model,save_model
from experiment_helpers.training import train_unet
from datasets import load_dataset
import torch

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--save_dir",type=str,default="/scratch/jlb638/unet/")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/unet_images/")
parser.add_argument("--load_saved",action="store_true")
parser.add_argument("--use_lora",action="store_true")
parser.add_argument("--pretrained_src",type=str,default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--epochs",type=int,default=5)
parser.add_argument("--hf_repo",type=str,default="jlbaker361/league_unet")
parser.add_argument("--pretrained_vae",action="store_true")
parser.add_argument("--vae_id",type=str,default="jlbaker361/league_vae_25")




def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_src)
    if args.pretrained_vae:
        vae_weight_path = hf_hub_download(repo_id=args.vae_id, filename="vae/diffusion_pytorch_model.bin")
        load_model(pipeline.vae,vae_weight_path)

    pipeline=pipeline.to(accelerator.device)
    for model in [pipeline.vae, pipeline.text_encoder]:
        model.eval()
    pipeline.unet.train()
    param_groups = [p for p in pipeline.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=0.0001, weight_decay=5e-2, betas=(0.9, 0.95))
    training_image_list=[row["splash"].resize((args.resize,args.resize)) for row in load_dataset(args.dataset,split="train")]
    training_prompt_list=[" " for _ in training_image_list]
    pipeline=train_unet(pipeline,
                        args.epochs,
                        training_image_list,
                        training_prompt_list,
                        optimizer,
                        False,
                        " ",
                        args.batch_size,
                        1.0,
                        "character",
                        accelerator,
                        30,
                        0.0,
                        True
                        )
    
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