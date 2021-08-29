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
from albumentations import *
from albumentations.pytorch import ToTensorV2


class MaskDataLoader:
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self):
        
        transform = self.get_transforms()
        transforms_train = transform['train']
        transforms_test = transform['val']
        
        train_path = '/opt/ml/input/data/train/'         # range(2700), [2140,560] 
        train_idx,test_idx = torch.utils.data.random_split(range(5400), [4280,1120], generator=torch.Generator().manual_seed(42))
        self.dataset = MaskDataset(train_path,list(train_idx),transforms_train)
        self.test_dataset = MaskDataset(train_path,list(test_idx),transforms_test)
        self.data_loader = DataLoader(self.dataset, batch_size = 64, shuffle= True, num_workers=4,drop_last = True) 
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=4,drop_last = True)




    def get_transforms(self,need=('train', 'val'), img_size=(224, 224), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        """
        train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.

        Args:
            need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
            img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
            mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
            std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

        Returns:
            transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
        """
        transformations = {}
        if 'train' in need:
            transformations['train'] = Compose([
                Resize(img_size[0], img_size[1], p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                GaussNoise(p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        if 'val' in need:
            transformations['val'] = Compose([
                Resize(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        return transformations
    
    
#     transforms_train = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomChoice([
#                 transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomAffine(
#                     degrees=15, translate=(0.2, 0.2),
#                     scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
#             ]),
#             transforms.ToTensor(),
#         ])

#         transforms_test = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
            
#         ])  