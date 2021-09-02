import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import gc
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Resnet18
class Resnet18T2142(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)

        resnet18.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

        torch.nn.init.xavier_uniform_(resnet18.fc.weight)
        stdv = 1/math.sqrt(512)
        resnet18.fc.bias.data.uniform_(-stdv, stdv)
        self.net = resnet18
    def forward(self, x):
        out = self.net(x)
        return out

class Resnext50T2142(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnext50 = torch.hub.load('pytorch/vision:v0.8.0', 'resnext50_32x4d', pretrained=True)
        print(resnext50)
        resnext50.fc = torch.nn.Linear(in_features=2048 ,out_features=num_classes , bias=True)

        torch.nn.init.xavier_uniform_(resnext50.fc.weight)
        stdv = 1/math.sqrt(2048)
        resnext50.fc.bias.data.uniform_(-stdv, stdv)
        self.net = resnext50

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.net(x)

class Resnext101T2142(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnext101 = torch.hub.load('pytorch/vision:v0.8.0', 'resnext101_32x8d', pretrained=True)
        print(resnext101)
        resnext101.fc = torch.nn.Linear(in_features=2048 ,out_features=num_classes , bias=True)

        torch.nn.init.xavier_uniform_(resnext101.fc.weight)
        stdv = 1/math.sqrt(2048)
        resnext101.fc.bias.data.uniform_(-stdv, stdv)
        self.net = resnext101

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.net(x)

class EfficientnetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        efficientnetb0 = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        self.net = efficientnetb0

    def forward(self, x):
        return self.net(x)
    
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x