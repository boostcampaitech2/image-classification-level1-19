import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset import MaskDataset
import warnings
warnings.filterwarnings('ignore')

import torch
from skimage import io, transform

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import math

class MaskDataLoader:
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self):
        transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomResizedCrop(224),
                transforms.RandomAffine(
                    degrees=15, translate=(0.2, 0.2),
                    scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
            ]),
            transforms.ToTensor(),
        ])

        transforms_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            
        ])  
        train_path = '/opt/ml/input/data/train/'
        train_idx,test_idx = torch.utils.data.random_split(range(2700), [2140,560], generator=torch.Generator().manual_seed(42))
        self.dataset = MaskDataset(train_path,list(train_idx),transforms_train)
        self.test_dataset = MaskDataset(train_path,list(test_idx),transforms_test)
        self.data_loader = DataLoader(self.dataset, batch_size = 64, shuffle= True, num_workers=4,drop_last = True) 
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=4,drop_last = True)


