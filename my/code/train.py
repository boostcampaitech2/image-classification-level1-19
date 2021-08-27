import os
import copy
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from utils import is_val_loss_decreasing
from tqdm import tqdm
from pandas.core.algorithms import mode
from model import EnsembleModel, MaskModel
from inference import test_prediction
from dataset import MaskDataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

# reproduct
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

# check loader
def check_loader(loaders):
    X, y = next(iter(loaders))
    print('X[0] shape : ', X[0].shape)
    print('y[0] value : ', y[0])
    print('X length : ', len(X))

# check image
def check_image(image):
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)

    # ============================
    # histogram of color
    # ============================
    # plt.axis('off')
    # histo = plt.subplot(1, 2, 2)
    # histo.set_ylabel('Count')
    # histo.set_xlabel('Pixel Intensity')
    # plt.hist(image.flatten(), bins=10, lw=0, alpha=0.5, color='r')

# return transforms
def get_transformer(aug_flag=True):
    transformer = dict()
    transformer['origin'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
                                ])

    if aug_flag:
        transformer['aug1'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.RandomRotation(5),
                                transforms.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8)),
                                transforms.ToTensor(),
                                ])

        transformer['aug2'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                    (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
                                ])

        transformer['aug3'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                ])

    return transformer

def main(config):
    root = './input/data'
    transform = get_transformer(True)
    
    mask_train_origin = MaskDataset(root, transform['origin'], train=True)
    mask_test_origin = MaskDataset(root, transform['origin'], train=False)

    # age_flag = True => augmentating only age >= 60
    # mask_train_aug1 = MaskDataset(root, transform['aug1'], age_flag=False)
    # mask_train_aug2 = MaskDataset(root, transform['aug2'], age_flag=False)
    # mask_train_aug3 = MaskDataset(root, transform['aug3'], age_flag=False)

    # check_image(mask_train_origin[0]['image'][0])

    # train_val = ConcatDataset([mask_train_origin, mask_train_aug1, mask_train_aug2, mask_train_aug3])
    # train_val = ConcatDataset([mask_train_origin, mask_train_aug1])

    train, val = train_test_split(mask_train_origin, test_size=0.1, shuffle=True, random_state=43)
    print('=========== Train and Val ===========')

    loaders = {
        'train': DataLoader(train, batch_size=128, num_workers=4, pin_memory=True, drop_last=True),
        'val': DataLoader(val, batch_size=128, num_workers=4, pin_memory=True, drop_last=True),
        'test': DataLoader(mask_test_origin, batch_size=128, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {
        'train': len(train),
        'val': len(val),
        'test': len(mask_test_origin)
    }

    # check X, y
    check_loader(loaders['train'])
    print('loaders[train] length : ', len(loaders['train']))

    # gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyper params
    num_epochs = 10
    num_classes = len(mask_train_origin.classes)

    # make models
    # vgg19 = MaskModel('vgg19', num_classes, pretrained=True)
    # resnet18 = MaskModel('resnet18', num_classes, pretrained=True)
    # googlenet = MaskModel('googlenet', num_classes, pretrained=True)
    # densenet121 = MaskModel('densenet121', num_classes, pretrained=True)
    resnet101 = MaskModel('resnet101', num_classes, pretrained=True)

    ############################## freezing doesn't work well ##############################
    # for param in resnet101.model.parameters():
    #     param.grad_requires = False

    # resnet101.model.fc = nn.Linear(in_features=resnet101.model.fc.in_features, out_features=num_classes, bias=True)
    # nn.init.xavier_uniform_(resnet101.model.fc.weight)
    ############################## freezing doesn't work well ##############################

    # models = [vgg19, resnet50, googlenet, densenet121]

    training(num_epochs, resnet101.model, 'resnet101_freezing', loaders, dataset_sizes, device, mask_test_origin)
    test_prediction(resnet101.model, 'resnet101_freezing', device, loaders['test'], mask_test_origin)

    # training models
    # for m in models:
    #     training(num_epochs, m.model, m.model_name, loaders, dataset_sizes, device, mask_test_origin)
    
    # ensemble(models, num_classes, device, loaders['test'], mask_test_origin)

# ================================================================================================

# ensemble
def ensemble(models, num_classes, device, test_loader, mask_test_origin):
    print('=============== Start Ensemble ===============')
    ensemble_model = EnsembleModel(models, num_classes, device)
    ensemble_model.to(device)
    test_prediction(ensemble_model, 'ensemble', device, test_loader, mask_test_origin)

# ================================================================================================

# training
def training(num_epochs, model, model_name, loaders, dataset_sizes, device, mask_test_origin):

    print('=============== Start Training ===============')
    
    # ============================
    # for viz
    # ============================
    losses = {'train':[], 'val':[]}
    accuracies = {'train':[], 'val':[]}
    lr = []
    
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # ============================
    # optimizers
    # ============================
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, eps=1e-08)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # ============================
    # schedulers
    # ============================
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=epochs, steps_per_epoch=len(loaders['train']), cycle_momentum=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)

    # time tracking
    since = time.time()

    # init
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(loaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # back prop only training
                    outp = model(inputs)
                    loss = criterion(outp, labels)
                    _, pred = torch.max(outp, 1)
            
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                        # lr.append(scheduler.get_lr())

                running_loss += loss.item() * inputs.size(0)            # per batch_size
                running_corrects += torch.sum(pred == labels.data)      # per batch_size

            if phase == 'train':
                acc = 100. * running_corrects.double() / dataset_sizes[phase]
                scheduler.step(acc)

            epoch_loss = running_loss / dataset_sizes[phase]                        # per epoch
            epoch_acc = 100. * running_corrects.double() / dataset_sizes[phase]     # per epoch

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            if phase == 'train':
                print('Epoch : {} / {}'.format(epoch + 1, num_epochs))
            print('{} - Loss : {:.4f}, Acc : {:.4f}%'.format(phase, epoch_loss, epoch_acc))
            lr.append(scheduler._last_lr)
            
            if phase == 'val':
                print('Training Time Spent : {}m {:.4f}s'.format((time.time() - since) // 60, (time.time() - since) % 60))
            
            # update best result
            if phase == 'val' and epoch_acc > best_acc:
                print('--- Update Best Model ---')
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            
            # dividing line
            print('==' * 15)
        
        # early stopping
        if not is_val_loss_decreasing(epoch, 3, losses['val']):
            print('--- Early Stopping ---')
            break

    time_elapsed = time.time() - since
    print('Whole Time Spent : {}m {:.4f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model
    model.load_state_dict(best_model)

    # save model
    root = './code/model'
    torch.save(model, os.path.join(root, f'{model_name}_{best_acc}.pth'))

    # inference
    # test_prediction(model, model_name, device, loaders['test'], mask_test_origin)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Mask Image Classification')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = args.parse_args()
    main(config)