import torch
from torch import nn
import random

class ContrastiveLossNormalized(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLossNormalized, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        #label 1= same, 0=different
        # Normalize embeddings across all dimensions (batch-wise normalization)
        all_outputs = torch.cat((output1, output2), dim=0)
        mean = all_outputs.mean()
        std = all_outputs.std()
        
        # Normalize using batch statistics
        output1 = (output1 - mean) / (std + 1e-8)
        output2 = (output2 - mean) / (std + 1e-8)
        
        # Compute Euclidean distance between normalized embeddings
        euclidean_distance = torch.norm(output1 - output2, dim=1)
        print("euclid dist", euclidean_distance)
        
        # Contrastive Loss calculation
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        # Return the mean loss over the batch
        loss =  torch.mean(loss_similar + loss_dissimilar)
        return loss
    

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance between embeddings
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        # Loss for similar pairs
        loss_similar = label * torch.pow(euclidean_distance, 2)
        # Loss for dissimilar pairs
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        # Final loss is the sum of both
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss

def generate_random_crops(image, n):
    width, height = image.size
    ratio=random.uniform(0.8, 1)
    crop_width = int(0.8 * width)
    crop_height = int(0.8 * height)
    
    crops = []
    
    for _ in range(n):
        # Randomly choose the top-left corner of the crop
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        # Define the box for cropping (left, upper, right, lower)
        crop_box = (left, top, left + crop_width, top + crop_height)
        
        # Crop the image
        crop = image.crop(crop_box).resize((image.size))
        
        # Add the crop to the list of crops
        crops.append(crop)
    
    return crops

def sample_subsets(tensor_list, k):
    remaining_elements = tensor_list.copy()  # Make a copy of the original tensor list
    sampled_elements = []  # Track previously sampled elements
    subsets = []  # List to store the sampled subsets

    while len(remaining_elements) >= k:
        # If less than k elements remain, refill the pool
        if len(remaining_elements) < k:
            remaining_elements.extend(sampled_elements)
            sampled_elements.clear()

        # Randomly sample k tensors from the remaining elements
        subset_indices = random.sample(range(len(remaining_elements)), k)
        subset = [remaining_elements[i] for i in subset_indices]

        # Remove the sampled tensors from the remaining pool
        for index in sorted(subset_indices, reverse=True):
            del remaining_elements[index]

        # Add the subset to the list of subsets
        subsets.append(subset)

        # Keep track of the sampled elements
        sampled_elements.extend(subset)

    if len(remaining_elements)>0:
        extras=k-len(remaining_elements)
        subset_indices = random.sample(range(len(sampled_elements)), extras)
        subset = [sampled_elements[i] for i in subset_indices]
        subset.extend(remaining_elements)
        

    return subsets