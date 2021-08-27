from torch.utils.data.dataset import TensorDataset
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
import pandas as pd
import os
import torch

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

# root = './input/data'
# transform = transforms.Compose([transforms.ToTensor()])
# ds = MaskDataset(root, transform)

# vision transformer
class ViTDataset(TensorDataset):
    '''
    For creating image dataset for ViT model

    Args:

    '''    
    def __init__(self, root, transform, train=True) -> None:
        super(ViTDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.classes = [i for i in range(0, 18)]

        X, y = [], []
        if train:
            self.root = os.path.join(self.root, 'train')
            self.info_df = pd.read_csv(self.root + '/new_train.csv', delimiter=',', encoding='utf-8-sig')

            for i, image_path in tqdm(enumerate(self.info_df['image_path'])):
                if i == 100:
                    break
                image = Image.open(image_path)
                label = self.info_df.iloc[i]['label']
                X.append(self.transform(image))
                y.append(label)
                
                # age 60 data increase
                if label == 2 or label == 14:
                    for i in range(0, 8):
                        X.append(self.transform(image))
                        y.append(label)

            self.X = X
            self.y = y

    def __getitem__(self, index):
        return {'img': self.X[index], 'label': self.y[index]}

    def __len__(self):
        return len(self.X)

# ============================= mean / std of images =================================

# img_info = get_img_stats(img_dir, df.path.values)

# print(f'RGB Mean: {np.mean(img_info["means"], axis=0) / 255.}')
# print(f'RGB Standard Deviation: {np.mean(img_info["stds"], axis=0) / 255.}')

# def get_img_stats(img_dir, img_ids):
#     img_info = dict(heights=[], widths=[], means=[], stds=[])
#     for img_id in tqdm(img_ids):
#         for path in glob(os.path.join(img_dir, img_id, '*')):
#             img = np.array(Image.open(path))
#             h, w, _ = img.shape
#             img_info['heights'].append(h)
#             img_info['widths'].append(w)
#             img_info['means'].append(img.mean(axis=(0,1)))
#             img_info['stds'].append(img.std(axis=(0,1)))
#     return img_info