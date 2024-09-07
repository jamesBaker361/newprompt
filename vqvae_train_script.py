import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
from vqvae_model import VQVAE
import torch.optim as optim
import numpy as np

import wandb
from torchvision.utils import save_image



parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="vqvae")
parser.add_argument("--hf_dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--save_interval",type=int,default=50)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/swin_images/")
parser.add_argument("--checkpoint_dir",type=str,default="/scratch/jlb638/swin_checkpoints/")
parser.add_argument("--repo_id",type=str,default="jlbaker361/swin-512")
parser.add_argument("--test_data",action="store_true")
parser.add_argument("--load_saved",action="store_true")
parser.add_argument("--train_contrastive",action="store_true")
parser.add_argument("--contrastive_weight",type=float,default=1.0)
parser.add_argument("--contrastive_cluster_size",type=int,default=4)
parser.add_argument("--contrastive_n_clusters",type=int,default=4)
parser.add_argument("--contrastive_steps_per_epoch",type=int,default=16)
parser.add_argument("--contrastive_margin",type=float,default=2.0)
parser.add_argument("--contrastive_start_epoch",type=int,default=150)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    transform_list = [
            transforms.Resize((int(args.img_size),int(args.img_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    device=accelerator.device

    data=[row["splash"] for row in load_dataset(args.hf_dataset,split="train")]
    if args.test_data:
        data=[Image.open("boot.jpg") for _ in range(32)]
    i=0
    while len(data) %args.batch_size!=0:
        data.append(data[i])
        i+=1
    data=[trans(image) for image in data]
    batched_data=[]
    for j in range(0,len(data),args.batch_size):
        batched_data.append(data[j:j+args.batch_size])
    batched_data=[torch.stack(batch) for batch in batched_data]

    fixed_images=batched_data[0].to(device)
    batched_data=batched_data[1:]
    print(f"each epoch has {len(batched_data)} batches")
    for fixed in fixed_images:
        print("fixed range",torch.max(fixed),torch.min(fixed))

    model= VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    for e in range(args.epochs):
        start_time=time.time()
        recon_list=[]
        perplexity_list=[]
        embedding_list=[]
        loss_list=[]
        for i,batch in enumerate(batched_data):
            optimizer.zero_grad()
            batch=batch.to(device)
            embedding_loss, predicted, perplexity = model(batch)

            recon_loss = torch.mean((predicted - batch)**2) #/ x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()
            recon_list.append(recon_loss.item())
            perplexity_list.append(perplexity.item())
            embedding_list.append(embedding_loss.item())
            loss_list.append(loss.item())

        metrics={
            "recon_loss":np.mean(recon_list),
            "perplexity":np.mean(perplexity_list),
            "embedding_loss":np.mean(embedding_list),
            "loss":np.mean(loss_list)
        }
        end_time=time.time()
        print(f"epoch {e} elapsed {end_time-start_time} seconds")
        for k,v in metrics.items():
            print("\t",k,v)
        accelerator.log(metrics)
        with torch.no_grad():
            _, predicted, __ = model(fixed_images)
            print("pred range",torch.max(pred),torch.min(pred))
            pred=pred.add(1).mul(0.5)
            for index,image in enumerate(pred):
                src_image=fixed_images[index].add(1).mul(0.5)
                
                path=f"{args.image_dir}pred_{index}.jpg"
                save_image([src_image,image],path)
                try:
                    accelerator.log({
                        f"pred_{index}":wandb.Image(path)
                    })
                except:
                    print("couldnt upload ",path)

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