import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


# Custom Model Template    
class Vgg19(nn.Module) :
    def __init__(self, num_classes = 18, pretrained = True):
        super(Vgg19, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        vgg19 = models.vgg19_bn(pretrained = self.pretrained)
        self.model = vgg19
        self.model.classifier[6] = nn.Linear(in_features = 4096, out_features = self.num_classes, bias = True)
        
        
    def forward(self, x) :
        return self.model(x)
    
    
class Resnet18(nn.Module) :
    def __init__(self, num_classes = 18, pretrained = True):
        super(Resnet18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        resnet18 = models.resnet18(pretrained = self.pretrained)
        self.model = resnet18
        self.model.fc = nn.Linear(in_features = resnet18.fc.in_features, out_features = self.num_classes, bias = True)
        
    def forward(self, x) :
        return self.model(x)
    
class Resnet50(nn.Module) :
    def __init__(self, num_classes = 18, pretrained = True):
        super(Resnet50, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        resnet50 = models.resnet50(pretrained = self.pretrained)
        self.model = resnet50
        self.model.fc = nn.Linear(in_features = resnet50.fc.in_features, out_features = self.num_classes, bias = True)
        
    def forward(self, x) :
        return self.model(x)
    
class Googlenet(nn.Module) :
    def __init__(self, num_classes = 18, pretrained = True):
        super(Googlenet, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        googlenet = models.googlenet(pretrained = self.pretrained)
        self.model = googlenet
        self.model.fc = nn.Linear(in_features = googlenet.fc.in_features, out_features = self.num_classes, bias = True)
        
    def forward(self, x) :
        return self.model(x)

class Densenet121(nn.Module) :
    def __init__(self, num_classes = 18, pretrained = True):
        super(Densenet121, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        densenet121 = models.densenet121(pretrained = self.pretrained)
        self.model = densenet
        self.model.fc = nn.Linear(in_features = densenet121.fc.in_features, out_features = self.num_classes, bias = True)
        
    def forward(self, x) :
        return self.model(x)    