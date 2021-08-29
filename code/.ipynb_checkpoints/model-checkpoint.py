import torch
import torchvision
import math
import torch.nn as nn

class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        CLASS_NUM = 18
        self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=CLASS_NUM, bias=True)
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)
        self.resnet18.fc.weight.requires_grad = True
        self.resnet18.fc.bias.requires_grad = True
        
    def forward(self, x):
       
        return self.resnet18(x)





