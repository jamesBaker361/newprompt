import torch
from experiment_helpers.measuring import cos_sim
from typing import Union
import numpy as np
from PIL import Image
from scipy.linalg import norm

def cos_sim_rescaled(vector_i,vector_j,return_np=False):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    try:
        result= cos(vector_i,vector_j) *0.5 +0.5
    except TypeError:
        result= cos(torch.tensor(vector_i),torch.tensor(vector_j)) *0.5 +0.5
    if return_np:
        return result.detach().cpu().numpy()
    return result

def nearest(ft_src:Union[torch.Tensor, np.ndarray,Image.Image],ft_target:Union[torch.Tensor, np.ndarray,Image.Image],x:int,y:int)->list:
    '''
    ft_src= (c,h,w)
    ft_target (c,h,w)
    x,y coordinates of point in ft_src, and we want to find the nearest x,y in ft_target and the distance

    returns [[x,y],dist]
    '''
    if type(ft_src)==Image.Image:
        ft_src=np.array(ft_src)
        ft_src = np.transpose(ft_src, (2, 1, 0))
    if type(ft_target)==Image.Image:
        ft_target=np.array(ft_target)
        ft_target = np.transpose(ft_target, (2, 1, 0))
    src_feature_vector=ft_src[:,x,y]
    if type(ft_target)==torch.Tensor:
        (C,W,H)=ft_target.size()
    elif type(ft_target)==np.ndarray:
        
        (C,W,H)=ft_target.shape
    max_sim=-99999
    max_x=0
    max_y=0
    for i in range(W):
        for j in range(H):
            target_feature_vector=ft_target[:,i,j]
            sim=cos_sim_rescaled(src_feature_vector,target_feature_vector).item()
            if sim>max_sim:
                max_sim=sim
                max_x=i
                max_y=j
    return [[max_x,max_y], max_sim]

if __name__=='__main__':
    print("starting")
    # Create black 4x4 images
    image1 = Image.new('RGB', (4, 4), color='black')
    image2 = Image.new('RGB', (4, 4), color='black')

    # Access pixel data for the images
    pixels1 = image1.load()
    pixels2 = image2.load()

    # Set red dot in the top-left corner for image1
    pixels1[0, 0] = (255, 0, 0)

    # Set red dot in the bottom-right corner for image2
    pixels2[3, 3] = (255, 0, 0)

    print(nearest(image1,image2,0,0))
    print("ending!?!?")

