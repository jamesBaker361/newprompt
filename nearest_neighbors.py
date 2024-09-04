import torch
from experiment_helpers.measuring import cos_sim
from typing import Union
import numpy as np
from PIL import Image

def nearest(ft_src:Union[torch.Tensor, np.ndarray,Image.Image],ft_target:Union[torch.Tensor, np.ndarray,Image.Image],x:int,y:int)->list:
    '''
    ft_src= (c,h,w)
    ft_target (c,h,w)
    x,y coordinates of point in ft_src, and we want to find the nearest x,y in ft_target and the distance

    returns [[x,y],dist]
    '''
    if type(ft_src)==Image.Image:
        ft_src=np.array(ft_src)
    if type(ft_target)==Image.Image:
        ft_target=np.array(ft_target)
    src_feature_vector=ft_src[:,x,y]
    if type(ft_target)==torch.tensor:
        (C,W,H)=ft_target.size()
    elif type(ft_target)==np.ndarray:
        (C,W,H)=ft_target.shape
    max_sim=-99999
    max_x=0
    max_y=0
    for i in range(W):
        for j in range(H):
            target_feature_vector=ft_target[:,i,j]
            sim=cos_sim(target_feature_vector, src_feature_vector)
            if sim>max_sim:
                sim=max_sim
                max_x=i
                max_y=j
    return [[max_x,max_y], max_sim]

if __name__=='__main__':
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

