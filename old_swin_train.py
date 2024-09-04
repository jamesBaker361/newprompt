import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from torchvision import transforms
import torch
from datasets import load_dataset
from PIL import Image
from swin_mae import SwinMAE
from swin_utils.misc import NativeScalerWithGradNormCount,load_model
from swin_utils.lr_sched import adjust_learning_rate
from torchvision.utils import save_image
import math
import numpy as np
import wandb
from functools import partial
from torch import nn
from huggingface_hub import HfApi
from experiment_helpers.checkpoint import find_latest_checkpoint
import re
from swin_contrast_utils import generate_random_crops, sample_subsets, ContrastiveLoss
api = HfApi()

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="swin")
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--checkpoint_encoder', default='', type=str)
parser.add_argument('--checkpoint_decoder', default='', type=str)
parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='Masking ratio (percentage of removed patches).')
parser.add_argument("--img_size",type=int,default=512)
parser.add_argument("--patch_size",type=int,default=4)
parser.add_argument("--window_size",type=int,default=8)

# model parameters
parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)
parser.add_argument("--embed_dim",type=int,default=64)
parser.add_argument("--decoder_embed_dim",type=int,default=512)

# optimizer parameters
parser.add_argument('--accum_iter', default=1, type=int)
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR')

# other parameters
parser.add_argument('--output_dir', default='./output_dir',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')



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



def main(args):
    api.create_repo(args.repo_id,exist_ok=True)
    for folder in [args.image_dir, args.checkpoint_dir]:
        os.makedirs(folder,exist_ok=True)
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

    if args.train_contrastive:
        contrastive_loss_module=ContrastiveLoss(args.contrastive_margin)
        contrastive_data=[row["splash"] for row in load_dataset(args.hf_dataset,split="train")]
        if args.test_data:
            contrastive_data=[Image.open("boot.jpg") for _ in range(32)]
        contrastive_batches=[]
        for image in contrastive_data:
            random_crops=generate_random_crops(image,args.contrastive_cluster_size)
            random_crops=[trans(crop) for crop in random_crops]
            contrastive_batches.append(random_crops)
        contrastive_batches=[torch.stack(contrast_batch) for contrast_batch in contrastive_batches]

    

    # Set model
    model = SwinMAE(norm_pix_loss=args.norm_pix_loss, 
                                          mask_ratio=args.mask_ratio,
                                          embed_dim=args.embed_dim,
                                          decoder_embed_dim=args.decoder_embed_dim,
                                          img_size=args.img_size,
                                          patch_size=args.patch_size,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          window_size=args.window_size)
    
    if args.load_saved:
        pattern= re.compile(r"model_(\d+)\.pt")
        path,current_epoch=find_latest_checkpoint(args.checkpoint_dir,pattern)
        if path is not None:
            args.start_epoch=current_epoch
            path=os.path.join(args.checkpoint_dir,path)
            print(f"loadinf from {path}")
            state_dict=torch.load(path)
            model.load_state_dict(state_dict)
    model.to(device)
    model_without_ddp = model

    #fixed_noise = torch.FloatTensor(args.batch_size, args.img_size//8,args.img_size//8,args.decoder_embed_dim).normal_(0, 1).to(device)
    fixed_images=batched_data[0].to(device)
    batched_data=batched_data[1:]
    print(f"each epoch has {len(batched_data)} batches")
    for fixed in fixed_images:
        print("fixed range",torch.max(fixed),torch.min(fixed))
    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))  # 原来是5E-2
    loss_scaler = NativeScalerWithGradNormCount()

    # Create model
    #load_model(args=args, model_without_ddp=model_without_ddp)
    model.train(True)

    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for e in range(args.start_epoch,args.epochs+1):
        contrastive_weight=args.contrastive_weight*(e/args.epochs)
        loss_list=[]
        start_time=time.time()
        for data_iter_step,batch in enumerate(batched_data):
            optimizer.zero_grad()
            batch=batch.to(device)
            if data_iter_step % args.accum_iter == 0:
                adjust_learning_rate(optimizer, data_iter_step / len(batched_data) + e, args)
            
            loss, _, _ = model(batch)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                break
            loss_list.append(loss_value)
            loss.backward()
            optimizer.step()
        end_time=time.time()
        print(f"epoch {e} elapsed {end_time-start_time} seconds")
        metrics={
            "loss":np.mean(loss_list)
        }
        if args.train_contrastive and contrastive_weight>0:
            start_time=time.time()
            similarity_contrastive_loss_list=[]
            different_contrastive_loss_list=[]
            subsets=sample_subsets(contrastive_batches,args.contrastive_n_clusters)
            
            for subset in subsets:
                optimizer.zero_grad()
                clusters=[]
                similarity_contrastive_loss=0.0
                for batch in subset:
                    #each batch is a bunch of images of the SAME thing
                    batch=batch.to(device)
                    embeddings,_=model.forward_encoder(batch)
                    #clusters.append(embeddings)
                    '''for i in range(len(embeddings)):
                        for j in range(i+1,len(embeddings)):
                            contrastive_loss+=contrastive_loss_module(embeddings[i],embeddings[j],0)'''
                    similarity_contrastive_loss=contrastive_weight* sum([sum([contrastive_loss_module(embeddings[i],embeddings[j],1) for j in range(i+1,len(embeddings)) ]) for i in range(len(embeddings)) ])
                    similarity_contrastive_loss=similarity_contrastive_loss/(len(embeddings)* (len(embeddings)-1)/2)
                    similarity_contrastive_loss.backward()
                    optimizer.step()
                    similarity_contrastive_loss_list.append(similarity_contrastive_loss.item())
                optimizer.zero_grad()
                for batch in subset:
                    #each batch is a bunch of images of the SAME thing
                    batch=batch.to(device)
                    embeddings,_=model.forward_encoder(batch)
                    clusters.append(embeddings)
                different_contrastive_loss=contrastive_weight*sum([sum([contrastive_loss_module(clusters[i],clusters[j],0) for j in range(i+1,len(clusters))]) for i in range(len(clusters)) ])
                different_contrastive_loss=different_contrastive_loss/(len(embeddings)* len(clusters) *(len(clusters)-1)/2 )
                different_contrastive_loss.backward()
                optimizer.step()
                different_contrastive_loss_list.append(different_contrastive_loss.item())
            end_time=time.time()
            print(f"contrastive epoch {e} elapsed {end_time-start_time} seconds")
            metrics["similarity_contrastive_loss"]=np.mean(similarity_contrastive_loss_list)
            metrics["different_contrastive_loss"]=np.mean(different_contrastive_loss_list)
        for k,v in metrics.items():
            print("\t",k,v)
        accelerator.log(metrics)
        with torch.no_grad():
            _,pred,_=model(fixed_images)
            #print('pred size',pred.size())
            pred=model.unpatchify(pred)
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
            '''fake_images=model.forward_decoder(fixed_noise)
            fake_images=fake_images.add(1).mul(0.5)
            for index,image in enumerate(fake_images):
                path=f"{args.image_dir}fake_{index}.jpg"
                save_image(image,path)
                accelerator.log({
                    f"fake_{index}":wandb.Image(path)
                })'''
        if e%args.save_interval==0 or e==args.epochs:
            torch.save(model.state_dict(),f"{args.checkpoint_dir}model_{e}.pt")
            api.upload_folder(repo_id=args.repo_id,
                              repo_type="model",
                              folder_path=args.checkpoint_dir)



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