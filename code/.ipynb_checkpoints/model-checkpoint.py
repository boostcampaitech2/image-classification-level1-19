import torch
import torchvision
import math
import torch.nn as nn

class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        # print("네트워크 필요 입력 채널 개수", self.resnet18.conv1.weight.shape[1])
        # print("네트워크 출력 채널 개수 (예측 class type 개수)", self.resnet18.fc.weight.shape[0])
        # print("네트워크 구조", self.resnet18)
        CLASS_NUM = 18
        self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=CLASS_NUM, bias=True)
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

        
   
   
    def forward(self, x):
       
        return self.resnet18(x)





