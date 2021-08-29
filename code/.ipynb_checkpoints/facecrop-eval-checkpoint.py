import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = '/opt/ml/input/data/eval/new_imgs'
img_path = '/opt/ml/input/data/eval/images'
os.mkdir(new_img_dir)
cnt = 0

for paths in os.listdir(img_path):
    if paths[0] == '.': continue
    
    imgs = os.path.join(img_path,paths)
    
    
    if paths[0] == '.': continue

   
    img = cv2.imread(imgs)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #mtcnn 적용
    boxes,probs = mtcnn.detect(img)

    # boxes 확인
    if len(probs) > 1: 
        print(boxes)
    if not isinstance(boxes, np.ndarray):
        print('Nope!')
        # 직접 crop
        img=img[100:400, 50:350, :]

    # boexes size 확인
    else:
        xmin = int(boxes[0, 0])-30
        ymin = int(boxes[0, 1])-30
        xmax = int(boxes[0, 2])+30
        ymax = int(boxes[0, 3])+30

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512

        img = img[ymin:ymax, xmin:xmax, :]
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)   
    tmp = os.path.join(new_img_dir, paths)
    cnt += 1
    plt.imsave(tmp, img)
#################### 슬랙에 메세지 남기기 ####################
url = 'https://hooks.slack.com/services/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'# 웹후크 URL 입력
message = "Finsh" # 메세지 입력
title = (f"data crop") # 타이틀 입력
slack_data = {
    "username": "NotificationBot", # 보내는 사람 이름
    "icon_emoji": ":satellite:",
    #"channel" : "#somerandomcahnnel",
    "attachments": [
        {
            "color": "#9733EE",
            "fields": [
                {
                    "title": title,
                    "value": message,
                    "short": "false",
                }
            ]
        }
    ]
}
byte_length = str(sys.getsizeof(slack_data))
headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
response = requests.post(url, data=json.dumps(slack_data), headers=headers)
if response.status_code != 200:
    raise Exception(response.status_code, response.text)
         #############################################################        
print(cnt)