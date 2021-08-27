import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')



# train_data_path = '/opt/ml/input/data/train/train.csv'
# eval_data_path = '/opt/ml/input/data/eval/info.csv'
class MaskDataset(Dataset):
    
    
    def __init__(self, data_path,idx,transform=None):
        self.main_path = data_path
        self.df_csv = pd.read_csv(data_path+'train.csv').iloc[idx]
        self.df_csv['gender'] = self.df_csv['gender'].map({'male':0, 'female':1})
        self.transform = transform  
        self.length = self.__len__()
        
        
    def __getitem__(self, index):

            main_index, sub_index = index//7, index%7
            sub_path = self.df_csv.iloc[main_index]['path']
            file_path = os.path.join(self.main_path, 'images', sub_path)
            files = [file_name for file_name in os.listdir(file_path) if file_name[0] != '.']
            image = Image.open(os.path.join(file_path, files[sub_index]))
            label = self.data_classification(self.mask_classification(files[sub_index]), 
                                self.df_csv.iloc[main_index]['gender'],self.age(self.df_csv.iloc[main_index]['age'])) 

            if self.transform:
                image = self.transform(image)



            return image, label

    
    def __len__(self):
        return len(self.df_csv)*7
    
    
    def age(self,x):
        if int(x) < 30: return 0
        elif int(x) < 60: return 1
        else: return 2
    
    
    def mask_classification(self,path):

        if path.startswith('m'):return 0
        elif path.startswith('i'):return 1
        else:return 2


    def data_classification(self,mask,gender,age):
        
        if mask == 0 and gender == 0 and age == 0:
            return 0
        elif mask == 0 and gender == 0 and age == 1:
            return 1
        elif mask == 0 and gender == 0 and age == 2:
            return 2
        elif mask == 0 and gender == 1 and age == 0:
            return 3
        elif mask == 0 and gender == 1 and age == 1:
            return 4
        elif mask == 0 and gender == 1 and age == 2:
            return 5
        elif mask == 1 and gender == 0 and age == 0:
            return 6
        elif mask == 1 and gender == 0 and age == 1:
            return 7
        elif mask == 1 and gender == 0 and age == 2:
            return 8
        elif mask == 1 and gender == 1 and age == 0:
            return 9
        elif mask == 1 and gender == 1 and age == 1:
            return 10
        elif mask == 1 and gender == 1 and age == 2:
            return 11
        elif mask == 2 and gender == 0 and age == 0:
            return 12
        elif mask == 2 and gender == 0 and age == 1:
            return 13
        elif mask == 2 and gender == 0 and age == 2:
            return 14
        elif mask == 2 and gender == 1 and age == 0:
            return 15
        elif mask == 2 and gender == 1 and age == 1:
            return 16
        elif mask == 2 and gender == 1 and age == 2:
            return 17
    





