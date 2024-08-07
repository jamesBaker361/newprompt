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
import os

limit=256

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



segmentation_model=get_segmentation_model(device,torch.float32)
mtcnn=MTCNN(device=device)

src_dict={
    "label":[],
    "splash":[],
    "subject":[],
  #  "blip_caption":[],
   # "face_caption":[],
    #"fashion_caption":[],
}

cursed_labels=[
    "annie08",
    "garen04",
    "ezreal02",
    "ezreal06",
    "braum33"
]
for link in links:
    #print(link)
    href=None
    title=None
    try:
        href = link.get('href')
        title=link.get('title')
    except:
        pass
    if href!=None and title !=None and title+"/"==href and title not in ["cassiopeia","drmundo"]:  # Add other extensions if needed
        
        #skins/skin30/images/aatrox_splash_uncentered_30.jpg
        for num in range(50):
            #print(title,num)
            formatted_num=str(num)
            if num<10:
                formatted_num="0"+formatted_num
            label=title+formatted_num
            if label not in cursed_labels:
                file_url = base_url + href +f"skins/skin{formatted_num}/images/{title}_splash_uncentered_{num}.jpg"
                file_name = os.path.join("lol_characters", title+f"_{num}.jpg")
                head_response = requests.head(file_url)
                if head_response.status_code == 200:
                    print(file_url)
                    img_response = requests.get(file_url)
                    img_response.raise_for_status()
                    img_data = BytesIO(img_response.content)


                    
                    # Open the image with Pillow and save it
                    with Image.open(img_data) as img:
                        width, height = img.size
                        square_size = min(width, height)
                        left = (width - square_size) / 2
                        top = (height - square_size) / 2
                        right = (width + square_size) / 2
                        bottom = (height + square_size) / 2
                        img = img.crop((left, top, right, bottom))
                        boxes,probs=mtcnn.detect(img)
                        if boxes is not None and  probs[0]>=0.99:
                            array_img=np.array(img, dtype=np.uint8)
                            array_img = HWC3(array_img)
                            array_img = resize_image(array_img, 512)
                            H, W, C = array_img.shape

                            subject=get_caption(img,blip_processor,blip_conditional_gen)

                            proportion_poses = detector.detect_poses(array_img)
                            if len(proportion_poses)==1:
                                src_dict["label"].append(label)
                                src_dict["splash"].append(img)
                                src_dict["subject"].append("character")
                                #src_dict['blip_caption'].append(get_caption(img,blip_processor,blip_conditional_gen).replace("stock photo","").replace("stock image",""))
                                #src_dict["subject"].append(remove_numbers(os.path.splitext(filename)[0]))
                                #src_dict["face_caption"].append(get_face_caption(img,blip_processor,blip_conditional_gen,mtcnn,10))
                                #src_dict["fashion_caption"].append(get_fashion_caption(img,blip_processor,blip_conditional_gen,segmentation_model,0))
                            #img.save(file_name)
                    #print(f"Downloaded {file_name}")
                            limit-=1
                            if limit %10==0:
                                Dataset.from_dict(src_dict).push_to_hub("jlbaker361/new_league_data_256")
                                load_dataset("jlbaker361/new_league_data_256")
                            if limit<=0:
                                exit()
