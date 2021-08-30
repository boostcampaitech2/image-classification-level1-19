import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import models
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

class EffiModel(nn.Module):
    def __init__(self, num_classes=18) -> None:
        super(EffiModel, self).__init__()
        self.num_classes = num_classes
        self.model = self.get_efficient_b4()
        

    def get_efficient_b4(self):
        efficient = EfficientNet.from_pretrained('efficientnet-b4')
        efficient.classifier = nn.Sequential(
    nn.Linear( 1280* 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2),
    nn.Linear(4096, self.num_classes),)

        return efficient

    def forward(self, x):
        return self.model(x)

class ResnetModel(nn.Module):
    def __init__(self, num_classes=18) -> None:
        super(ResnetModel, self).__init__()
        self.num_classes = num_classes
        self.model = self.get_resnet()
        

    def get_resnet(self):
        resnet = models.resnet18(pretrained=True)
        resnet.classifier = nn.Sequential(
    nn.Linear( 512* 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2),
    nn.Linear(4096, self.num_classes),)

        return resnet

    def forward(self, x):
        return self.model(x)


class EnsembleModel(nn.Module):
    '''
    This is ensemble class for improving score
    Args:
        models: A list of MaskModel
        device(str): Cuda or CPU
    '''
    def __init__(self, models, num_classes=18, device='cuda'):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([m.model for m in models])
        self.device = device
        self.num_classes = num_classes
    
    # just input x into all model sequentially
    def forward(self, x):
        output = torch.zeros([x.size(0), self.num_classes]).to(self.device)
        for model in self.models:
            output += model(x)
        return output



# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes=18) -> None:
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=18)
        self.model.classifier = nn.Sequential(
    nn.Linear(1280* 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096,num_classes),)

    def forward(self, x):
        return self.model(x)
