import os
import pandas as pd
import tqdm
import re

class Make_Label():
    def __init__(self, BASE_PATH, CSV_FILE, COLS):
        self.path= BASE_PATH
        self.img_path= os.path.join(self.path, 'images')
        self.csv_file= CSV_FILE
        self.df= pd.DataFrame(columns= COLS)

    def labeling(self):
        idx_df = 0
        for idx in tqdm.tqdm(range(train_csv.shape[0])):
            IMG_PATH = os.path.join(self.img_path, train_csv.iloc[idx]['path']) # image path
            file_list = os.listdir(IMG_PATH)

            for file in file_list:
                if file.rstrip().startswith('._'): # server 내에서 ._{파일명} 제거
                    continue
                
                file_path = os.path.join(IMG_PATH, file)
                self.df.loc[idx_df] = train_csv.loc[idx][['gender', 'age']] # train.csv 파일에서 gender, age 정보를 가져옴
                self.df.loc[idx_df]['path'] = file_path # image 전체 경로
                self.df.loc[idx_df]['name'] = file # image 파일 이름
                self.check_label(self.df, idx_df)
                idx_df += 1

        self.df = self.df.label
        self.df.to_csv('label.csv') # csv 파일 형식으로 저장

    def check_label(self, df, idx_df):
        # DataFrame의 mask, gender, age 정보를 가져옴 
        mask = df.loc[idx_df]['name'][:4] 
        gender = df.loc[idx_df]['gender'] 
        age = df.loc[idx_df]['age'] 

        # 데이터 tab의 class description 기준으로 분류
        if mask == 'mask':
            if gender == 'male':
                if age < 30:
                    df.loc[idx_df]['label'] = 0
                elif age < 60:
                    df.loc[idx_df]['label'] = 1
                else:
                    df.loc[idx_df]['label'] = 2
            else:
                if age < 30:
                    df.loc[idx_df]['label'] = 3
                elif age < 60:
                    df.loc[idx_df]['label'] = 4
                else:
                    df.loc[idx_df]['label'] = 5
        elif mask == 'inco':  # incorrct_mask
            if gender == 'male':
                if age < 30:
                    df.loc[idx_df]['label'] = 6
                elif age < 60:
                    df.loc[idx_df]['label'] = 7
                else:
                    df.loc[idx_df]['label'] = 8
            else:
                if age < 30:
                    df.loc[idx_df]['label'] = 9
                elif age < 60:
                    df.loc[idx_df]['label'] = 10
                else:
                    df.loc[idx_df]['label'] = 11
        elif mask == 'norm':  # not wear
            if gender == 'male':
                if age < 30:
                    df.loc[idx_df]['label'] = 12
                elif age < 60:
                    df.loc[idx_df]['label'] = 13
                else:
                    df.loc[idx_df]['label'] = 14
            else:
                if age < 30:
                    df.loc[idx_df]['label'] = 15
                elif age < 60:
                    df.loc[idx_df]['label'] = 16
                else:
                    df.loc[idx_df]['label'] = 17


if __name__== '__main__':

    BASE_TRAIN_PATH= './input/data/train' # train data path
    train_csv= pd.read_csv(os.path.join(BASE_TRAIN_PATH, 'train.csv'))
    # print(train_csv.shape) # (2700, 5)

    make_label= Make_Label(BASE_TRAIN_PATH, train_csv, ['gender', 'age', 'path', 'name', 'label'])
    make_label.labeling()