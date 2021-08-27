import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
import os
from tqdm import tqdm
import sys
import requests
import json
from pandas.io.json import json_normalize

class Trainer:
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, device,
                 data_loader, LEARNING_RATE, test_data_loader=None, scheduler=None):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.test_data_loader = test_data_loader
        self.LEARNING_RATE = LEARNING_RATE
        self.scheduler = scheduler
        if not scheduler:  
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min')


       

    def train(self,NUM_EPOCH):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        

        # metric
        best_test_accuracy = 0.
        best_test_f1 = 0.
        best_test_loss = 9999.
        best_prediction = []

        # path
        MODEL_PATH = '/opt/ml/code/saved/models/'

        # 데이터 로더
        dataloaders = {"train" : self.data_loader, "test" : self.test_data_loader }

        # early stop 
        early_stop_count = 0                                
        early_stop = 3

        for epoch in range(NUM_EPOCH): # NUM_EPOCH
            for phase in ["train", "test"]:
                running_loss = 0.
                running_acc = 0.
                epoch_f1 = 0.
                
                if phase == "train":
                    self.model.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
                elif phase == "test":
                    self.model.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함
                    prediction = [] # 그래프 뽑을 때 필요한  prediction
                    
                for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])):
                    # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
                    # hint - with, pbar
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                    with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                        logits = self.model(images)
                        _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
                        loss = self.criterion(logits, labels)
                        
                        
                    if phase == "train":
                        loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                        self.optimizer.step() # 계산된 gradient를 가지고 모델 업데이트
                    if phase == 'test':
                        self.scheduler.step(loss) 
                        prediction.extend(preds.cpu().numpy()) # 그래프 뽑을 때 필요한  prediction
                        
                    running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                    running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

                # 한 epoch이 모두 종료되었을 때,
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_acc / len(dataloaders[phase].dataset)
                epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                
                # 각 epoch 기록
                print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, 평균 F1 Scrore: {epoch_f1:.2f} lr : {self.optimizer.state_dict().get('param_groups')[0].get('lr')} \n\n")
                if phase == "test" and best_test_accuracy < epoch_acc: # phase가 test일 때, best accuracy 계산
                    best_test_accuracy = epoch_acc

                if phase == "test" and best_test_loss > epoch_loss: # phase가 test일 때, best loss 계산
                    best_test_loss = epoch_loss
                
                if phase == "test" and best_test_f1 < epoch_f1: # phase가 test일 때, best accuracy 계산
                    best_test_f1 = epoch_f1
                    torch.save(self.model, os.path.join(MODEL_PATH, f"resnet18_model_{self.LEARNING_RATE}_best.pt"))
                    early_stop_count = 0
                    best_prediction = prediction
                    
                    #################### 슬랙에 메세지 남기기 ####################
                    url = 'https://hooks.slack.com/services/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'# 웹후크 URL 입력
                    message = f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, 평균 F1 Scrore: {epoch_f1:.2f} lr : {self.optimizer.state_dict().get('param_groups')[0].get('lr')}" # 메세지 입력
                    title = (f"New Incoming Message :zap:") # 타이틀 입력
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
                    
                else:
                    early_stop_count += 1
                
            # early stop
            if early_stop_count == early_stop:
                print("="*10 + "early stopped." + "="*10)
                break
        print("학습 종료!")
        file_newname_newfile = os.path.join(MODEL_PATH,f"resnet18_model_{best_test_f1:.2f}_{self.LEARNING_RATE}_best.pt")
        file_oldname = os.path.join(MODEL_PATH, f"resnet18_model_{self.LEARNING_RATE}_best.pt")
        os.rename(file_oldname, file_newname_newfile)
        print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}, 최고 높은 f1 : {best_test_f1}")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)