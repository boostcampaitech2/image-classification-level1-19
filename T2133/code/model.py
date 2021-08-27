import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

def get_imagenet_resnet18():
    MASK_CLASS_NUM = 18
    imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
    imagenet_resnet18.fc = torch.nn.Linear(in_features=512,     out_features=MASK_CLASS_NUM, bias=True)


    # classifier initialize
    nn.init.xavier_uniform_(imagenet_resnet18.fc.weight)
    stdv = 1/math.sqrt(512)
    imagenet_resnet18.fc.bias.data.uniform_(-stdv, stdv)
    return imagenet_resnet18