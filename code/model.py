import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import timm

from efficientnet_pytorch import EfficientNet

def resnet18(classes):
    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(in_features=512, out_features=classes, bias=True)
    
    nn.init.xavier_uniform_(resnet18.conv1.weight)
    nn.init.xavier_uniform_(resnet18.fc.weight)

    stdv = 1.0/np.sqrt(classes)
    resnet18.fc.bias.data.uniform_(-stdv, stdv)

    return resnet18

def resnet34(classes):
    resnet34 = torchvision.models.resnet34(pretrained=True)
    #resnet34.conv1 = nn.Conv2d(classes, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    resnet34.fc = nn.Linear(in_features=512, out_features=classes, bias=True)
    nn.init.xavier_uniform_(resnet34.conv1.weight)
    nn.init.xavier_uniform_(resnet34.fc.weight)

    # nn.init.kaiming_uniform_(resnet34.conv1.weight, mode='fan_out', nonlinearity='relu')
    # nn.init.normal_(resnet34.fc.weight, 0, 0.01)

    stdv = 1.0/np.sqrt(classes)
    resnet34.fc.bias.data.uniform_(-stdv, stdv)
    return resnet34

def resnet101(classes):
    resnet101 = torchvision.models.resnet101(pretrained=True)
    resnet101.fc = nn.Linear(in_features=2048, out_features=classes, bias=True)
    
    nn.init.xavier_uniform_(resnet101.conv1.weight)
    nn.init.xavier_uniform_(resnet101.fc.weight)
    
    stdv = 1.0/np.sqrt(classes)
    resnet101.fc.bias.data.uniform_(-stdv, stdv)
    
    return resnet101


def resnet152(classes):
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.fc = nn.Linear(in_features=2048, out_features=classes, bias=True)
    
    nn.init.xavier_uniform_(resnet152.conv1.weight)
    nn.init.xavier_uniform_(resnet152.fc.weight)
    
    stdv = 1.0/np.sqrt(classes)
    resnet152.fc.bias.data.uniform_(-stdv, stdv)
    
    return resnet152

class ResNetEnsemble(nn.Module):
    def __init__(self, resnet_a, resnet_b, classes):
        super(ResNetEnsemble, self).__init__()
        self.resnet_a = resnet_a
        self.resnet_b = resnet_b
        
        self.resnet_a.fc = nn.Identity()
        self.resnet_b.fc = nn.Identity()
        
        self.classifier = nn.Linear(2048+512, classes)
        
        stdv = 1.0/np.sqrt(classes)
        nn.init.xavier_uniform_(self.resnet_a.conv1.weight)
        nn.init.xavier_uniform_(self.resnet_b.conv1.weight)
        self.classifier.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        x1 = self.resnet_a(x.clone())
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.resnet_b(x)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim = 1)
        x = self.classifier(F.relu(x))
        return x

def ensemble_resnet50(classes):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet34 = torchvision.models.resnet34(pretrained=True)
    return ResNetEnsemble(resnet50, resnet34, classes)
    

def ensemble_resnet152(classes):
    resnet101 = torchvision.models.resnet101(pretrained=True)
    resnet152 = torchvision.models.resnet152(pretrained=True)
    
    return ResNetEnsemble(resnet101, resnet152, classes)
    

def inception(classes):
    inception = torchvision.models.inception_v3(pretrained=True)
    return inception

def densenet(classes):
    densenet = torchvision.models.densenet161(pretrained=True)
    densenet.classifier = nn.Linear(in_features=2208, out_features=classes, bias=True)
    stdv = 1.0/np.sqrt(classes)
    densenet.classifier.bias.data.uniform_(-stdv, stdv)

    return densenet

def inception_resnet_v2(classes):
    irnet = timm.create_model('inception_resnet_v2')
    irnet.classif = nn.Linear(in_features=1536, out_features=classes, bias=True)
    stdv = 1.0/np.sqrt(classes) 
    irnet.classif.bias.data.uniform_(-stdv, stdv)

    return irnet

def efficientnet_b0(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=classes)
    return effnet

def efficientnet_b3(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes=classes)
    return effnet

def efficientnet_b4(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b4', num_classes=classes)
    return effnet

def efficientnet_b7(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=classes)
    return effnet

def get_model(model_name:str, classes):
    model_dict = {
        'resnet18':resnet18,
        'resnet34':resnet34,
        'resnet101':resnet101,
        'resnet152':resnet152,
        'densenet':densenet,
        'ensemble_resnet50':ensemble_resnet50,
        'ensemble_resnet152':ensemble_resnet152,
        'inception_resnet_v2':inception_resnet_v2,
        'efficientnet_b0':efficientnet_b0,
        'efficientnet_b3':efficientnet_b3,
        'efficientnet_b4':efficientnet_b4,
        'efficientnet_b7':efficientnet_b7,
    }

    return model_dict[model_name](classes)