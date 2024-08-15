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

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="swin")
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--save_freq', default=400, type=int)
parser.add_argument('--checkpoint_encoder', default='', type=str)
parser.add_argument('--checkpoint_decoder', default='', type=str)
parser.add_argument('--data_path', default=r'C:\文件\数据集\腮腺对比学习数据集\三通道合并\concat\train', type=str)  # fill in the dataset path here
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
parser.add_argument('--log_dir', default='./output_dir',
                    help='path where to tensorboard log')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument("--hf_dataset",type=str,default="jlbaker361/new_league_data_256")
parser.add_argument("--save_interval",type=int,default=10)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/swin_images/")
parser.add_argument("--checkpoint_dir",type=str,default="/scratch/jlb638/swin_checkpoints/")
parser.add_argument("--repo_id",type=str,default="jlbaker361/swin-512")
parser.add_argument("--test_data",action="store_true")



def main(args):
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

    # Set model
    model = SwinMAE(norm_pix_loss=args.norm_pix_loss, 
                                          mask_ratio=args.mask_ratio,
                                          embed_dim=args.embed_dim,
                                          decoder_embed_dim=args.decoder_embed_dim,
                                          img_size=args.img_size,
                                          patch_size=args.patch_size,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          window_size=args.window_size)
    model.to(device)
    model_without_ddp = model

    fixed_noise = torch.FloatTensor(args.batch_size, args.img_size//8,args.img_size//8,args.decoder_embed_dim).normal_(0, 1).to(device)
    fixed_images=batched_data[0]
    for fixed in fixed_images:
        print("fixed range",torch.max(fixed),torch.min(fixed))
    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))  # 原来是5E-2
    loss_scaler = NativeScalerWithGradNormCount()

    # Create model
    load_model(args=args, model_without_ddp=model_without_ddp)
    model.train(True)

    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for e in range(args.start_epoch,args.epochs):
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
                accelerator.log({
                    f"pred_{index}":wandb.Image(path)
                })
            '''fake_images=model.forward_decoder(fixed_noise)
            fake_images=fake_images.add(1).mul(0.5)
            for index,image in enumerate(fake_images):
                path=f"{args.image_dir}fake_{index}.jpg"
                save_image(image,path)
                accelerator.log({
                    f"fake_{index}":wandb.Image(path)
                })'''
        if e%args.save_interval==0:
            torch.save(model.state_dict(),f"{args.checkpoint_dir}model_{e}.pt")



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