from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2
import torch
import os 
from tqdm import tqdm
import numpy as np

data_path = '/opt/ml/input/data/eval/'

class EvalDataset(Dataset):
    
    
    def __init__(self, data_path,transform=None):
        
        self.main_path = data_path
        self.df_csv = pd.read_csv(data_path+'info.csv')
        self.transform = transform  
        self.length = self.__len__()
        
        
    def __getitem__(self, index):

            
            sub_path = self.df_csv.iloc[index]['ImageID']
            file_path = os.path.join(self.main_path, 'new_imgs', sub_path) # 폴더 이름 1. images(원본) / 2. new_imgs (크랍)
            image = Image.open(file_path)
            
            if self.transform:
                image = self.transform(image=np.array(image))['image']

            return image

    
    def __len__(self):
        return len(self.df_csv)
MODEL_PATH= '/opt/ml/code/saved/models/' 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

img_size=(224, 224)
mean=(0.548, 0.504, 0.479)
std=(0.237, 0.247, 0.246)    
transforms_eval = Compose([Resize(img_size[0], img_size[1]),
        Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),], p=1.0)

eval_dataset = EvalDataset(data_path,transforms_eval)
eval_dataloaders = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0,drop_last = False)
answer = []
model = torch.load(os.path.join(MODEL_PATH, f"resnet18_model_0.89_0.001_best.pt")).to(device)

for ind, images in enumerate(tqdm(eval_dataloaders)):
            
    images = images.to(device)
    logits = model(images)
    _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
    answer.extend(preds.cpu().numpy())
    
submission = pd.read_csv(data_path+'info.csv')
submission['ans']= answer
submission.to_csv(os.path.join('/opt/ml/code/saved/result/'+f'resnet18_model_submission_0.89.csv'),index=False)

