from torch.utils.data.dataset import TensorDataset
from PIL import Image

import torchvision.transforms as transforms
import pandas as pd
import os
import torch

class CustomDataset(TensorDataset):
    '''
    Load images and info dataset from specified folder

    Args:
        path(str): A common directory path of train or eval dataset
        transform: A composer of transformer to modify images
        category(str): A flag for mask / age / gender
    '''
    def __init__(self, root, transform, category='mask') -> None:
        super(CustomDataset, self).__init__()
        self.root = os.path.join(root, 'train')
        self.category = category
        self.transform = transform
        self.classes = [i for i in range(0, 18)]
        self.info_df = pd.read_csv(self.root + '/new_train.csv', delimiter=',', encoding='utf-8-sig')

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, index):
        if self.category == 'mask':
            label = self.info_df.iloc[index]['mask_label']
        elif self.category == 'age':
            label = self.info_df.iloc[index]['age_label']
        else:
            label = self.info_df.iloc[index]['gender_label']
        image_path = '.' + self.info_df.iloc[index]['image_path'] # (caution) relative path for jupyter
        image = Image.open(image_path)
        return self.transform(image), torch.tensor(label)
    
    def get_label(self):
        return self.info_df.iloc[:]['age_label']


class MaskDataset(TensorDataset):
    '''
    Load images and info dataset from specified folder

    Args:
        path(str): A common directory path of train or eval dataset
        transform: A composer of transformer to modify images
        train(boolean): A flag whether the dataset is train or eval
        age_flag(boolean): A flag of considering augmentation for age >= 60
    '''
    def __init__(self, root, transform, train=True, age_flag=False) -> None:
        super(MaskDataset, self).__init__()
        self.root = root
        self.train = train
        self.age_flag = age_flag
        self.transform = transform
        self.classes = [i for i in range(0, 18)]

        X, y = [], []
        if self.train:
            self.root = os.path.join(self.root, 'train')
            self.info_df = pd.read_csv(self.root + '/new_train.csv', delimiter=',', encoding='utf-8-sig')
        else:
            self.root = os.path.join(self.root, 'eval')
            self.info_df = pd.read_csv(self.root + '/info.csv', delimiter=',', encoding='utf-8-sig')

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, index):
        if not self.train:
            path = os.path.join(self.root, 'images/')
            file = self.info_df.iloc[index]['ImageID']
            image = Image.open(path + file)
            return self.transform(image), self.info_df.iloc[index]['ans']

        label = self.info_df.iloc[index]['label']
        image_path = self.info_df.iloc[index]['image_path']
        image = Image.open(image_path)
        return self.transform(image), torch.tensor(label)


# root = '././input/data'
# transform = transforms.Compose([transforms.ToTensor()])
# ds = CustomDataset(root, transform)