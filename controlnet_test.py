import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.lora_loading import load_lora_weights,get_pipeline_from_hf
from experiment_helpers.measuring import get_metric_dict
import datetime
import time
from PIL import Image
import cv2
from accelerate import Accelerator
from datasets import load_dataset
from experiment_helpers.static_globals import METRIC_LIST
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import gc
import wandb
from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3,resize_image
from controlnet_aux.open_pose.util import draw_bodypose
from controlnet_aux.open_pose.body import Keypoint
from typing import List
import numpy as np

parser=argparse.ArgumentParser()

def normalize_keypoints(keypoints: List[Keypoint]) -> List[Keypoint]:
    xs = [kp.x for kp in keypoints]
    ys = [kp.y for kp in keypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y

    normalized_keypoints = [
        Keypoint((kp.x - min_x) / width, (kp.y - min_y) / height, kp.score, kp.id)
        for kp in keypoints
    ]
    
    return normalized_keypoints

def calculate_transformation(source: List[Keypoint], target: List[Keypoint]):
    source_xs = np.array([kp.x for kp in source])
    source_ys = np.array([kp.y for kp in source])
    target_xs = np.array([kp.x for kp in target])
    target_ys = np.array([kp.y for kp in target])
    
    scale_x = (target_xs.max() - target_xs.min()) / (source_xs.max() - source_xs.min())
    scale_y = (target_ys.max() - target_ys.min()) / (source_ys.max() - source_ys.min())
    
    scale = (scale_x + scale_y) / 2
    
    trans_x = target_xs.mean() - source_xs.mean() * scale
    trans_y = target_ys.mean() - source_ys.mean() * scale
    
    return scale, trans_x, trans_y

def transform_keypoints(keypoints: List[Keypoint], scale: float, trans_x: float, trans_y: float) -> List[Keypoint]:
    transformed_keypoints = [
        Keypoint(kp.x * scale + trans_x, kp.y * scale + trans_y, kp.score, kp.id)
        for kp in keypoints
    ]
    
    return transformed_keypoints

def adjust_keypoints(shape_keypoints: List[Keypoint], proportion_keypoints: List[Keypoint]) -> List[Keypoint]:
    normalized_shape = normalize_keypoints(shape_keypoints)
    normalized_proportion = normalize_keypoints(proportion_keypoints)
    
    scale, trans_x, trans_y = calculate_transformation(normalized_proportion, normalized_shape)
    
    adjusted_keypoints = transform_keypoints(proportion_keypoints, scale, trans_x, trans_y)
    
    return adjusted_keypoints

class OpenposeDetectorResize(OpenposeDetector):
    def __call__(
            self,
            proportion_image:np.ndarray,
            shape_image:np.ndarray,
            detect_resolution:int=512,
              image_resolution:int=512, 
              include_body:bool=True, 
              include_hand:bool=False, 
              include_face:bool=False,
               output_type="pil"):
        if not isinstance(proportion_image, np.ndarray):
            proportion_image = np.array(proportion_image, dtype=np.uint8)
        if not isinstance(shape_image, np.ndarray):
            shape_image = np.array(shape_image, dtype=np.uint8)

        proportion_image = HWC3(proportion_image)
        proportion_image = resize_image(proportion_image, detect_resolution)
        H, W, C = proportion_image.shape

        proportion_poses = self.detect_poses(proportion_image, include_hand, include_face)

        proportion_keypoints=proportion_poses[0].body.keypoints

        shape_image = HWC3(shape_image)
        shape_image = resize_image(shape_image, detect_resolution)
        H, W, C = shape_image.shape

        shape_poses = self.detect_poses(shape_image, include_hand, include_face)

        shape_keypoints=shape_poses[0].body.keypoints

        adjusted=adjust_keypoints(shape_keypoints, proportion_keypoints)

        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, adjusted)

        detected_map = canvas
        detected_map = HWC3(detected_map)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map

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

aggregate_dict={}

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
    detector=OpenposeDetectorResize.from_pretrained('lllyasviel/Annotators')
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
        sd_pipeline=get_pipeline_from_hf(hub_model_id,False,False,True,False,use_lora=True,pretrained_model_name="runwayml/stable-diffusion-v1-5",swap_pair=["weight","default.weight"]).sd_pipeline
        '''for k,v in sd_pipeline.unet.named_parameters():
            print(k)'''
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
        eval_images=[]
        for p,pose_image in enumerate(pose_image_list):
            try:
                pose_image=detector(src_image,pose_image)
            except:
                pass
            image=pipe(subject,image=pose_image,num_inference_steps=args.num_inference_steps).images[0]
            accelerator.log({
                f"{label}/image_{p}":wandb.Image(image)
            })
            eval_images.append(image)
        metric_dict=get_metric_dict(["" for _ in eval_images],eval_images, [src_image])
        for k,v in metric_dict.items():
            if k not in aggregate_dict:
                aggregate_dict[k]=[]
            aggregate_dict[k].append(v)
    accelerator.log({
        k:np.mean(v) for k,v in aggregate_dict.items()
    })
    for k,v in aggregate_dict.items():
        print(k,np.mean(v))

        

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