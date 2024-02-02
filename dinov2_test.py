#=============Libraries=======================
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
#pca analysis
from sklearn.decomposition import PCA

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

dinov2_vits14.eval()
dinov2_vits14.cuda()

image = '/mnt/1TBNVME/vivek/2024/Stanford_Cars/imgs_train/5/00151.jpg'

img = Image.open(image)

transform_pth = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_vectors(image, model, transform):
    """
    Get the feature vectors for all images in the path.
    """
    img = Image.open(image).convert('RGB')
    # get the length and width of the image
    #width, height = img.size
    #print(width, height)
    #aspect_ratio = width/height
    #transform = transforms.Compose([transforms.Resize((224, int(224*aspect_ratio))),
                                    #transforms.ToTensor(),
                                    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = transform(img)
    # add batch dimension
    img = img.unsqueeze(0)
    # Get the feature vector for the image
    #GPU:
    vector = model(img.cuda(non_blocking=True))
    #CPU:
    #vector = model(img)
    #vector = model(img.cuda(non_blocking=True))
    #making rank 0
    vector = nn.functional.normalize(vector, dim=1, p=2)
    #get image name
    target = image
    #print(target)
    #print(vector)S
    return vector, target

vec2, target2 = get_vectors(image, dinov2_vits14, transform_pth)

pca = PCA(n_components=2)
pca.fit(vec2.detach().cpu().numpy())
pca_vec = pca.transform(vec2.detach().cpu().numpy())
print(pca_vec)

