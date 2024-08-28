#https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/illaoi/skins/skin10/images/illaoi_splash_uncentered_10.jpg

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from controlnet_aux import OpenposeDetector
from facenet_pytorch import MTCNN
from datasets import Dataset,load_dataset
import numpy as np
from controlnet_aux.util import HWC3,resize_image
from experiment_helpers.measuring import get_caption
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from experiment_helpers.measuring import get_face_caption,get_fashion_caption
from experiment_helpers.elastic_face_iresnet import MTCNN
from experiment_helpers.clothing import get_segmentation_model
from experiment_helpers.background import remove_background_birefnet
import os
import time
from gpu import print_details
# Option 1: use with transformers

from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)


def extract_squares(image):
    # Ensure the image is rectangular
    width, height = image.size
    if width == height:
        raise ValueError("The image is already square.")
    if height > width:
        raise ValueError("The image should be wider than it is tall.")

    # Determine the size of the square to extract
    square_size = height  # Since we're extracting squares, we use the height

    # Crop the center square
    center_x = (width - square_size) // 2
    center_square = image.crop((center_x, 0, center_x + square_size, square_size))

    # Crop the leftmost square
    left_square = image.crop((0, 0, square_size, square_size))

    # Crop the rightmost square
    right_square = image.crop((width - square_size, 0, width, square_size))

    return left_square, center_square, right_square

print_details()

limit=10000

# URL of the webpage containing the links to the images
url = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/"

# Create a directory to save the downloaded files
os.makedirs("lol_characters", exist_ok=True)

# Fetch the content of the webpage
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links on the page
links = soup.find_all('a')

# Base URL to append to the links
base_url = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/"
detector=OpenposeDetector.from_pretrained("lllyasviel/Annotators")
#mtcnn = MTCNN( margin=15)
#blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#blip_conditional_gen = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").eval()
device="cpu"
if torch.cuda.is_available():
    device="cuda"



blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_conditional_gen = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").eval()

def is_more_than_90_black(image):
    # Convert image to grayscale
    grayscale_image = image.convert("L")
    
    # Convert the grayscale image to a NumPy array
    image_array = np.array(grayscale_image)
    
    # Count the number of black pixels (value == 0)
    black_pixels = np.sum(image_array == 0)
    
    # Get the total number of pixels
    total_pixels = image_array.size
    
    # Calculate the percentage of black pixels
    black_pixel_percentage = black_pixels / total_pixels
    
    # Check if more than 90% of the image is black
    return black_pixel_percentage > 0.90

segmentation_model=get_segmentation_model(device,torch.float32)
mtcnn=MTCNN(device=device)
birefnet=birefnet.to(device)

src_dataset=load_dataset("jschoormans/humanpose_densepose",split="train")

limit=1000
src_dict={
    "label":[],
    "splash":[],
    "subject":[],}
hf_path=f"jlbaker361/humans_90_plus_noback_{limit}"
suffix="_no_back" 
for i,row in enumerate(src_dataset):
    img=row["file_name"]
    boxes,probs=mtcnn.detect(img)
    if boxes is not None and  probs[0]>=0.99:
        array_img=np.array(img, dtype=np.uint8)
        array_img = HWC3(array_img)
        array_img = resize_image(array_img, 512)
        H, W, C = array_img.shape

        #subject=get_caption(img,blip_processor,blip_conditional_gen)

        proportion_poses = detector.detect_poses(array_img)
        if len(proportion_poses)==1:
            img=remove_background_birefnet(img,birefnet)
            if is_more_than_90_black(img)==False:
                src_dict["label"].append(f"human_{limit}")
                src_dict["splash"].append(img)
                src_dict["subject"].append("character")
                #src_dict['blip_caption'].append(get_caption(img,blip_processor,blip_conditional_gen).replace("stock photo","").replace("stock image",""))
                #src_dict["subject"].append(remove_numbers(os.path.splitext(filename)[0]))
                #src_dict["face_caption"].append(get_face_caption(img,blip_processor,blip_conditional_gen,mtcnn,10))
                #src_dict["fashion_caption"].append(get_fashion_caption(img,blip_processor,blip_conditional_gen,segmentation_model,0))
                limit-=1
                if limit %50==0:
                    try:
                        Dataset.from_dict(src_dict).push_to_hub(hf_path)
                    except:
                        pass
                elif limit <=0:
                    try:
                        Dataset.from_dict(src_dict).push_to_hub(hf_path)
                    except:
                        time.sleep(10)
                        Dataset.from_dict(src_dict).push_to_hub(hf_path)