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
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from diffusers.pipelines import BlipDiffusionPipeline
from PIL import Image
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from static_globals import *
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from aesthetic_reward import get_aesthetic_scorer
import argparse
from run_and_evaluate import evaluate_one_sample
from datasets import load_dataset
import wandb
from gpu import print_details
import gc
from experiment_helpers.static_globals import METRIC_LIST
import datetime

parser=argparse.ArgumentParser()

parser.add_argument("--limit",type=int,default=50)
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--method_name",type=str,default=BLIP_DIFFUSION)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/league-hard-prompt")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--convergence_scale",type=float,default=0.75)
parser.add_argument("--n_img_chosen",type=int,default=64)
parser.add_argument("--target_cluster_size",type=int,default=10)
parser.add_argument("--min_cluster_size",type=int,default=5)
parser.add_argument("--inf_config", type=str, default="dvlab/configs/rival_variation.json")
parser.add_argument("--inner_round", type=int, default=1, help="number of images per reference")
parser.add_argument("--is_half", type=bool, default=False)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--editing_early_steps", type=int, default=1000)
parser.add_argument("--train_text_encoder",action="store_true",help="ddpo train text encoder")
parser.add_argument("--train_text_encoder_embeddings",action="store_true",help="ddpo train text encoder like for text inversion")
parser.add_argument("--train_unet",action="store_true",help="ddpo whether to train unet")
parser.add_argument("--use_lora_text_encoder",action="store_true",help="ddpo use lora if train_text_encoder=True")
parser.add_argument("--use_lora",action="store_true",help="unet ddpo use lora")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--train_gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--reward_method",type=str,default=REWARD_NORMAL, help=f"one of {' '.join(REWARD_TYPE_LIST)}")
parser.add_argument("--num_epochs",type=int,default=10)
parser.add_argument("--samples_per_epoch",type=int,default=64)
parser.add_argument(
    "--p_step",
    type=int,
    default=5,
    help="The number of steps to update the policy per sampling step",
)
parser.add_argument(
    "--p_batch_size",
    type=int,
    default=4,
    help=(
        "batch size for policy update per gpu, before gradient accumulation;"
        " total batch size per gpu = gradient_accumulation_steps *"
        " p_batch_size"
    ),
)
parser.add_argument(
    "--v_flag",
    type=int,
    default=1,
)
parser.add_argument(
    "--g_step", type=int, default=1, help="The number of sampling steps"
)
parser.add_argument(
    "--g_batch_size",
    type=int,
    default=10,
    help="batch size of prompts for sampling per gpu",
)
parser.add_argument("--p_lr",type=float,default=0.000001)
parser.add_argument(
    "--reward_weight", type=float, default=10, help="weight of reward loss"
)
parser.add_argument(
    "--kl_weight", type=float, default=0.01, help="weight of kl loss"
)
parser.add_argument(
    "--kl_warmup", type=int, default=-1, help="warm up for kl weight"
)
parser.add_argument(
    "--buffer_size", type=int, default=1000, help="size of replay buffer"
)
parser.add_argument(
    "--v_batch_size", type=int, default=16, 
    help="batch size for value function update per gpu, no gradient accumulation"  # pylint: disable=line-too-long
)
parser.add_argument(
    "--v_lr", type=float, default=1e-4, help="learning rate for value fn"
)
parser.add_argument(
    "--v_step", type=int, default=5,
    help="The number of steps to update the value function per sampling step"
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=100,
    help="save model every save_interval steps",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
    help="number of samples to generate per prompt",
)
parser.add_argument(
    "--clip_norm", type=float, default=0.1, help="norm for gradient clipping"
)
parser.add_argument("--ratio_clip",type=int,default=0.0001)
parser.add_argument("--face_margin",type=int,default=10,help="pixel margin for extracted face")
metrics=["face","img_reward","vit","vit_style","vit_content","mse"]
for metric in metrics:
    parser.add_argument(f"--use_{metric}",action="store_true")
    parser.add_argument(f"--initial_{metric}_weight",type=float,default=0.0)
    parser.add_argument(f"--final_{metric}_weight",type=float,default=0.0)
parser.add_argument("--project_name",type=str,default="one_shot")
parser.add_argument("--subject_key",type=str,default="subject")
parser.add_argument("--label_key",type=str,default="label")
parser.add_argument("--image_key",type=str,default="splash")
parser.add_argument("--prompt_key",type=str,default="optimal_prompt")
parser.add_argument("--keyword",type=str,default="default")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/oneshot")
parser.add_argument("--value_epochs",type=int,default=0,help="epochs to train alue function before training unet for dpok")
parser.add_argument("--normalize_rewards",action="store_true",help="whether to normalize rewards for ddpo")
parser.add_argument("--normalize_rewards_individually",action="store_true",help="whether to normalize each individual reward in reward function")
parser.add_argument("--n_normalization_images",type=int,default=2)
parser.add_argument("--use_value_function",action="store_true")
parser.add_argument("--ddpo_lr",type=float,default=3e-4)
parser.add_argument("--use_mse_vae",action="store_true")
parser.add_argument("--pretrain",action="store_true")
parser.add_argument("--pretrain_epochs",type=int,default=10)
parser.add_argument("--pretrain_steps_per_epoch",type=int,default=64)
parser.add_argument("--use_default_text",action="store_true")
parser.add_argument("--default_text",type=str,default="League_of_legends_character")
'''  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=12,
      help=(
          "Number of updates steps to accumulate before performing a"
          " backward/update pass for policy"
      ),
  )
  parser.add_argument("--lora_rank", type=int, default=4, help="rank for LoRA")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=1e-5,
      help="Learning rate for policy",
  )'''

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    dataset=load_dataset(args.src_dataset,split="train")
    print('dataset.column_names',dataset.column_names)
    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    evaluation_prompt_list=[
        "a photo of {} at the beach",
        "a photo of  {} in the jungle",
        "a photo of  {} in the snow",
        "a photo of  {} in the street",
        "a photo of  {} with a city in the background",
        "a photo of  {} with a mountain in the background",
        "a photo of  {} with the Eiffel Tower in the background",
        "a photo of  {} near the Statue of Liberty",
        "a photo of  {} near the Sydney Opera House",
        "a photo of  {} floating on top of water",
        "a photo of  {} eating a burger",
        "a photo of  {} drinking a beer",
        "a photo of  {} wearing a blue hat",
        "a photo of  {} wearing sunglasses",
        "a photo of  {} playing with a ball",
        "a photo of  {} as a police officer"
    ]
    len_dataset=len([r for r in dataset])
    print("len",len_dataset)
    for j,row in enumerate(dataset):
        if j<args.start:
            continue
        gc.collect()
        accelerator.free_memory()
        torch.cuda.empty_cache()
        if j>args.limit:
            print("reached limit")
            break
        subject=row[args.subject_key]
        label=row[args.label_key]
        src_image=row[args.image_key]
        text_prompt=row[args.prompt_key]
        if args.use_default_text:
            text_prompt=args.default_text.replace("_"," ")
        metric_dict,evaluation_image_list=evaluate_one_sample(args.method_name,
                                                              src_image,
                                                              text_prompt,
                                                              evaluation_prompt_list,
                                                              accelerator,subject,
                                                              args.num_inference_steps,
                                                              args.n_img_chosen,
                                                              args.target_cluster_size,
                                                              args.min_cluster_size,
                                                              args.convergence_scale,
                                                              args.inf_config,
                                                                args.is_half,
                                                                args.seed,
                                                                args.inner_round,
                                                                args.editing_early_steps,
                                                                args.train_text_encoder,
                                                                args.train_text_encoder_embeddings,
                                                                args.train_unet,
                                                                args.use_lora_text_encoder,
                                                                args.use_lora,
                                                                args.train_gradient_accumulation_steps,
                                                                args.batch_size,
                                                                args.mixed_precision,
                                                                args.reward_method,
                                                                args.num_epochs,
                                                                args.p_step,
                                                                args.p_batch_size,
                                                                args.v_flag,
                                                                args.g_step,
                                                                args.g_batch_size,
                                                                args.reward_weight,
                                                                args.kl_weight,
                                                                args.kl_warmup,
                                                                args.buffer_size,
                                                                args.v_batch_size,
                                                                args.v_lr,
                                                                args.v_step,
                                                                args.save_interval,
                                                                args.num_samples,
                                                                args.ratio_clip,
                                                                args.samples_per_epoch,
                                                                args.face_margin,
                                                                args.use_face,
                                                                args.initial_face_weight,
                                                                args.final_face_weight,
                                                                args.use_img_reward,
                                                                args.initial_img_reward_weight,
                                                                args.final_img_reward_weight,
                                                                args.use_vit,
                                                                args.initial_vit_weight,
                                                                args.final_vit_weight,
                                                                args.use_vit_style,
                                                                args.initial_vit_style_weight,
                                                                args.final_vit_style_weight,
                                                                args.use_vit_content,
                                                                args.initial_vit_content_weight,
                                                                args.final_vit_content_weight,
                                                                args.image_dir,
                                                                args.value_epochs,
                                                                args.normalize_rewards,
                                                                args.normalize_rewards_individually,
                                                                args.n_normalization_images,
                                                                args.use_value_function,
                                                                args.p_lr,
                                                                args.ddpo_lr,
                                                                args.use_mse,
                                                                args.initial_mse_weight,
                                                                args.final_mse_weight,
                                                                args.use_mse_vae,
                                                                args.pretrain,
                                                                args.pretrain_epochs,
                                                                args.pretrain_steps_per_epoch
                                                                )
        os.makedirs(f"{args.image_dir}/{label}/",exist_ok=True)
        for i,image in enumerate(evaluation_image_list):
            path=f"{args.image_dir}/{label}/{args.method_name}_{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}/{args.method_name}_{i}":wandb.Image(path)
            })
        print(f"after {j} samples:")
        for metric,value in metric_dict.items():
            aggregate_dict[metric].append(value)
            print(f"\t{metric} : {value}")
    print(f"after {j} samples:")
    for metric,value_list in aggregate_dict.items():
        print(f"\t{metric} {np.mean(value_list)}")
        accelerator.log({
            f"{metric}":np.mean(value_list)
        })
        accelerator.log({
            f"{args.method_name}_{metric}":np.mean(value_list)
        })
if __name__=='__main__':
    # Get current date and time
    current_datetime = datetime.datetime.now()

    # Print the current date and time
    print("Current date and time:", current_datetime)
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!")