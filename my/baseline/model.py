import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, model_name, num_classes=18, pretrained=True) -> None:
        super(MyModel, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model_name = model_name.lower()

        # model selection
        try:
            if self.model_name == 'vgg19':
                self.model = self.get_vgg19()
            elif self.model_name == 'resnet18':
                self.model = self.get_resnet18()
            elif self.model_name == 'resnet50':
                self.model = self.get_resnet50()
            elif self.model_name == 'resnet101':
                self.model = self.get_resnet101()
            elif self.model_name == 'resnet152':
                self.model = self.get_resnet152()
            elif self.model_name == 'googlenet':
                self.model = self.get_googlenet()
            elif self.model_name == 'densenet121':
                self.model = self.get_densenet121()
            elif self.model_name == 'efficientnetb2':
                self.model = self.get_efficientnetb2()
            elif self.model_name == 'efficientnetb4':
                self.model = self.get_efficientnetb4()
            else:
                raise ValueError('(Model Not Found) Try Another Model')
        except ValueError as err_msg:
            print(err_msg)

    def get_vgg19(self):
        vgg19 = torchvision.models.vgg19_bn(pretrained=self.pretrained)
        vgg19.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
        return vgg19

    def get_resnet18(self):
        resnet18 = torchvision.models.resnet18(pretrained=self.pretrained)
        # resnet18.layer1.add_module(name='dropout', module=nn.Dropout2d(p=0.5))
        # resnet18.layer2.add_module(name='dropout', module=nn.Dropout2d(p=0.5))

        resnet18.fc = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=resnet18.fc.in_features, out_features=self.num_classes, bias=True)
        )
        return resnet18

    def get_resnet50(self):
        resnet50 = torchvision.models.resnet50(pretrained=self.pretrained)
        resnet50.fc = nn.Linear(in_features=resnet50.fc.in_features, out_features=self.num_classes, bias=True)
        return resnet50

    def get_resnet101(self):
        resnet101 = torchvision.models.resnet101(pretrained=self.pretrained)
        resnet101.fc = nn.Linear(in_features=resnet101.fc.in_features, out_features=self.num_classes, bias=True)
        return resnet101

    def get_resnet152(self):
        resnet152 = torchvision.models.resnet152(pretrained=self.pretrained)
        resnet152.fc = nn.Linear(in_features=resnet152.fc.in_features, out_features=self.num_classes, bias=True)
        return resnet152

    def get_googlenet(self):
        googlenet = torchvision.models.googlenet(pretrained=self.pretrained)
        googlenet.fc = nn.Linear(in_features=googlenet.fc.in_features, out_features=self.num_classes, bias=True)
        return googlenet

    def get_efficientnetb2(self):
        efficientnetb2 = EfficientNet.from_pretrained('efficientnet-b2', num_classes=18)
        return efficientnetb2

    def get_efficientnetb4(self):
        efficientnetb4 = EfficientNet.from_pretrained('efficientnet-b4', num_classes=18)
        return efficientnetb4

    def get_densenet121(self):
        densenet121 = torchvision.models.densenet121(pretrained=self.pretrained)
        densenet121.classifier = nn.Linear(in_features=densenet121.classifier.in_features,
                                           out_features=self.num_classes, bias=True)
        return densenet121