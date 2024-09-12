#this will finetune the vae used for stable diffusion ON the league data
import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from experiment_helpers.training import pil_to_tensor_process
from datasets import load_dataset
import torch
from huggingface_hub import hf_hub_download, HfApi,create_repo
from safetensors.torch import load_file,save_file
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
import wandb

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="pretrain_vae")
parser.add_argument("--save_dir",type=str,default="/scratch/jlb638/vae/")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/vae_images/")
parser.add_argument("--hf_repo",type=str,default="jlbaker361/league_vae")
parser.add_argument("--save_interval",type=int,default=5)
parser.add_argument("--load_saved",action="store_true")
parser.add_argument("--pretrained_src",type=str,default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--resize",type=int,default=768)

def tensor_to_pil(tensor:torch.Tensor)->Image.Image:
    tensor = (tensor + 1) / 2

    # Step 2: Scale the tensor from [0, 1] to [0, 255] and convert to uint8
    tensor = tensor.clamp(0, 1)  # Ensure the values are in [0, 1] range
    tensor = tensor.mul(255).byte()  # Convert to [0, 255] and type uint8

    # Step 3: Convert to PIL image
    to_pil = ToPILImage()
    image = to_pil(tensor)
    return image


def main(args):
    os.makedirs(args.save_dir,exist_ok=True)
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    image_list=[row["splash"].resize((args.resize,args.resize)) for row in load_dataset(args.dataset,split="train")]
    data=[pil_to_tensor_process(img) for img in image_list]
    batched_data=[]
    for j in range(0,len(data),args.batch_size):
        batched_data.append(data[j:j+args.batch_size])
    batched_data=[torch.stack(batch) for batch in batched_data]
    fixed_images=batched_data[0].to(accelerator.device)
    batched_data=batched_data[1:]
    print(f"each epoch has {len(batched_data)} batches")
    for fixed in fixed_images:
        print("fixed range",torch.max(fixed),torch.min(fixed))
    
    vae=StableDiffusionPipeline.from_pretrained(args.pretrained_src).vae
    '''weight_path=hf_hub_download(repo_id=args.pretrained_src, filename="vae/diffusion_pytorch_model.safetensors")
    state_dict=load_file(weight_path)

    vae.load_state_dict(state_dict)'''

    vae.train()

    vae=vae.to(accelerator.device)

    param_groups = [p for p in vae.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=0.0001, weight_decay=5e-2, betas=(0.9, 0.95))  # 原来是5E-2

    for e in range(args.epochs):
        loss_list=[]
        for step,batch in enumerate(batched_data):
            optimizer.zero_grad()
            batch=batch.to(vae.device)
            encoded=vae.encode(batch).latent_dist.sample()

            decoded=vae.decode(encoded,return_dict=False)

            loss=F.mse_loss(encoded,decoded)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        accelerator.log({
            "loss":np.mean(loss_list)
        })
        encoded_fixed=vae.encode(fixed_images).latent_dist.sample()
        decoded_fixed=vae.decode(encoded_fixed,return_dict=False)

        images=[tensor_to_pil(t) for t in decoded_fixed]

        for x,image in enumerate(images):
            try:
                accelerator.log({
                    f"image_{x}":wandb.Image(image)
                })
            except:
                path="temp.png"
                image.save(path)
                accelerator.log({
                    f"image_{x}":wandb.Image(path)
                })

        if e+1 % args.save_interval==0:
            state_dict=vae.state_dict()

            save_file(state_dict,os.path.join(args.save_dir,f"checkpoint_{e}/diffusion_pytorch_model.bin"))
    
    state_dict=vae.state_dict()

    save_file(state_dict,os.path.join(args.save_dir,f"diffusion_pytorch_model.bin"))

    api = HfApi()
    create_repo(args.hf_repo,repo_type="model",exist_ok=True)
    api.upload_folder(
        folder_path=args.save_dir,
        repo_id=args.hf_repo,
        repo_type="model",
    )

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