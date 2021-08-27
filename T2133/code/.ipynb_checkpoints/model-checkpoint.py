import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
