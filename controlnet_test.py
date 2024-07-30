import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.lora_loading import load_lora_weights,get_pipeline_from_hf
import datetime
import time
from PIL import Image
from accelerate import Accelerator
from datasets import load_dataset
from experiment_helpers.static_globals import METRIC_LIST
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import gc
import wandb
from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3
import numpy as np

parser=argparse.ArgumentParser()

class OpenposeDetectorResize(OpenposeDetector):
    def __call__(
            self,
            proportion_image:np.ndarray,
            pose_image:np.ndarray,
            detect_resolution:int=512,
              image_resolution:int=512, 
              include_body:bool=True, 
              include_hand:bool=False, 
              include_face:bool=False,
               output_type="pil"):
        if not isinstance(proportion_image, np.ndarray):
            proportion_image = np.array(proportion_image, dtype=np.uint8)
        if not isinstance(pose_image, np.ndarray):
            pose_image = np.array(pose_image, dtype=np.uint8)

parser.add_argument("--limit",type=int,default=1)
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/league-hard-prompt")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--project_name",type=str,default="oneshot_control")
parser.add_argument("--subject_key",type=str,default="subject")
parser.add_argument("--label_key",type=str,default="label")
parser.add_argument("--image_key",type=str,default="splash")
parser.add_argument("--prompt_key",type=str,default="optimal_prompt")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/oneshot_control")
parser.add_argument("--ddpo_save_hf_tag",type=str,default="vanilla")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--use_default_text",action="store_true")
parser.add_argument("--default_text",type=str,default="League_of_legends_character")

def main(args):
    pose_directory="poses"
    pose_image_list=[]
    for file in os.listdir(pose_directory):
        if file.endswith("png"):
            pose_image_list.append(Image.open(os.path.join(pose_directory,file)))
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    dataset=load_dataset(args.src_dataset,split="train")
    print('dataset.column_names',dataset.column_names)
    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    len_dataset=len([r for r in dataset])
    print("len",len_dataset)
    for j,row in enumerate(dataset):
        if j<args.start:
            continue
        gc.collect()
        accelerator.free_memory()
        torch.cuda.empty_cache()
        if j>=args.limit:
            print("reached limit")
            break
        subject=row[args.subject_key]
        label=row[args.label_key]
        src_image=row[args.image_key]
        text_prompt=row[args.prompt_key]
        if args.use_default_text:
            text_prompt=args.default_text.replace("_"," ")
            subject=text_prompt
        hub_model_id=f"jlbaker361/{args.ddpo_save_hf_tag}_{label}"
        sd_pipeline=get_pipeline_from_hf(hub_model_id,False,False,True,False,use_lora=True,pretrained_model_name="runwayml/stable-diffusion-v1-5").sd_pipeline
        torch_dtype={
            "no":torch.float16,
            "fp16":torch.float16
        }[args.mixed_precision]
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch_dtype
        )
        pipe = StableDiffusionControlNetPipeline(vae=sd_pipeline.vae,
                                                 tokenizer=sd_pipeline.tokenizer,
                                                 text_encoder=sd_pipeline.text_encoder,
                                                 unet=sd_pipeline.unet,
                                                 controlnet=controlnet,
                                                 scheduler=sd_pipeline.scheduler,safety_checker=None,
                                                 feature_extractor=None,requires_safety_checker=False)
        pipe.to(accelerator.device)
        pipe.to(torch_dtype)
        pipe.unet,pipe.text_encoder,pipe.vae,pipe.controlnet=accelerator.prepare(pipe.unet,pipe.text_encoder,pipe.vae,pipe.controlnet)
        for p,pose_image in enumerate(pose_image_list):
            image=pipe(subject,image=pose_image,num_inference_steps=args.num_inference_steps).images[0]
            accelerator.log({
                f"image_test_{p}":wandb.Image(image)
            })

if __name__=='__main__':
    current_datetime = datetime.datetime.now()

    # Print the current date and time
    print("Current date and time:", current_datetime)
    start = time.time()
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")