import sys
from datasets import Dataset,load_dataset
import argparse
from optim_utils_hf import *
from accelerate import Accelerator
from transformers import Blip2Processor, Blip2ForConditionalGeneration

'''
{
    "prompt_len": 16,
    "iter": 3000,
    "lr": 0.1,
    "weight_decay": 0.1,
    "prompt_bs": 1,
    "loss_weight": 1.0,
    "print_step": 100,
    "batch_size": 1,
    "clip_model": "ViT-H-14",
    "clip_pretrain": "laion2b_s32b_b79k"
}
'''

parser=argparse.ArgumentParser()
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/new_league_data_hard")
parser.add_argument("--src_dataset",type=str,default="jlbaker361/new_league_data")
parser.add_argument("--limit",type=int,default=20)
parser.add_argument("--clip_model",type=str,default="openai/clip-vit-base-patch32")
parser.add_argument("--clip_pretrain",type=str,default="laion2b_s32b_b79k")
parser.add_argument("--prompt_len",type=int,default=8)
parser.add_argument("--iter",type=int,default=3000)
parser.add_argument("--lr",type=float,default=0.1)
parser.add_argument("--weight_decay",type=float,default=0.1)
parser.add_argument("--prompt_bs",type=int,default=1)
parser.add_argument("--loss_weight",type=float,default=1.0)
parser.add_argument("--print_step",type=int,default=500)
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=1)

def main(args):
    accelerator=Accelerator()
    device=accelerator.device
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    clip_processor=CLIPProcessor.from_pretrained(args.clip_model)
    tokenizer=AutoTokenizer.from_pretrained(args.clip_model)
    clip_model=CLIPModel.from_pretrained(args.clip_model).to(device)
    #model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device,cache_dir=cache_dir)
    try:
        prior_dataset=load_dataset(args.dest_dataset,split="train")
        src_dict={
            column: prior_dataset[column] for column in prior_dataset.column_names
        }
    except:
        src_dict={
            "label":[],
            "optimal_prompt":[],
            "splash":[],
            "subject":[]
        }
    label_set=set([])
    src_dataset=load_dataset(args.src_dataset,split="train")
    for i,row in enumerate(src_dataset):
        if i>args.limit:
            break
        label=row["label"]
        print(f"label {label}")
        if label in label_set:
            print(f"already did {label}")
            continue
        splash=row["splash"]
        #optimal_prompt=optimize_prompt(model, preprocess, args, device, target_images=[splash])
        optimal_prompt=optimize_prompt(args,clip_processor,clip_model,tokenizer,splash,device)
        """
        for question in ["a picture of"]:
                    inputs = processor(splash_img, question, return_tensors="pt")

                    out = model.generate(**inputs)
                    caption+=" "+processor.decode(out[0], skip_special_tokens=True).strip()
        """
        blip_inputs=blip_processor(splash,"a picture of",return_tensors="pt")
        blip_outputs=blip_model.generate(**blip_inputs)
        subject=blip_processor.decode(blip_outputs[0],skip_special_tokens=True).strip()
        src_dict["label"].append(label)
        src_dict["splash"].append(splash)
        src_dict["optimal_prompt"].append(optimal_prompt)
        src_dict["subject"].append(subject)
        Dataset.from_dict(src_dict).push_to_hub(args.dest_dataset)
if __name__=='__main__':
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    try:
        print('torch.cuda.get_device_name()',torch.cuda.get_device_name())
        print('torch.cuda.get_device_capability()',torch.cuda.get_device_capability())
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    except:
        print("couldnt print cuda details")
    args=parser.parse_args()
    print(args)
    main(args)
