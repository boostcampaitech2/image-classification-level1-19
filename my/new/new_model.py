import torch
import torch.nn as nn
import torchvision
from torchvision.models.densenet import densenet121
from torchvision.models.googlenet import googlenet
from torchvision.models.vgg import vgg19

class MaskModel(nn.Module):
    '''
    A model selection for comparing each models

    Args:
        model_name(str): A model name
        num_classes(int): A length of classes
    '''
    def __init__(self, model_name, num_classes=18, pretrained=True) -> None:
        super(MaskModel, self).__init__()
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
        resnet18.fc = nn.Linear(in_features=resnet18.fc.in_features, out_features=self.num_classes, bias=True)
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

    def get_densenet121(self):
        densenet121 = torchvision.models.densenet121(pretrained=self.pretrained)
        densenet121.classifier = nn.Linear(in_features=densenet121.classifier.in_features, out_features=self.num_classes, bias=True)
        return densenet121


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