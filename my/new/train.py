import os
import copy
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# custom module
from new_utils import is_val_loss_decreasing, check_image, check_loader, get_transformer
from new_model import EnsembleModel, MaskModel
from new_inference import test_prediction
from new_dataset import CustomDataset, MaskDataset
from torchsampler import ImbalancedDatasetSampler

from tqdm import tqdm
from pandas.core.algorithms import mode
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split, KFold

# reproduct
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

# training
def training(model, model_name, num_epochs, loaders, dataset_sizes, fold):

    print('=============== Start Training ===============')
    device = 'cuda'
    
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

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
            total = 0.0

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
                total += labels.size(0)

            if phase == 'train':
                acc = 100. * running_corrects.double() / total
                scheduler.step(acc)

            epoch_loss = running_loss / total                                       # per epoch
            epoch_acc = 100. * running_corrects.double() / total                    # per epoch

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            if phase == 'train':
                print('Epoch : {} / {}'.format(epoch + 1, num_epochs))
            print('{} - Loss : {:.4f}, Acc : {:.4f}%'.format(phase, epoch_loss, epoch_acc))
            print()
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
    root = '././code/model'
    torch.save(model, os.path.join(root, f'{model_name}_{best_acc}.pth'))

    return epoch_acc

def common_train(kfold, model, num_epochs, train_ds, test_ds):

    loaders, dataset_sizes = dict(), dict()
    sum = 0.0

    print(' --- K-Fold Start --- ')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ds)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        loaders['train'] = DataLoader(train_ds, batch_size=128, num_workers=4, pin_memory=True, sampler=train_sampler)
        loaders['val'] = DataLoader(train_ds, batch_size=128, num_workers=4, pin_memory=True, sampler=val_sampler)

        check_loader(loaders['train'])

        dataset_sizes['train'] = len(loaders['train'])
        dataset_sizes['val'] = len(loaders['val'])

        sum += training(model.model, model.model_name, num_epochs, loaders, dataset_sizes, fold)

    print()
    print(' --- K-Fold Result --- ')
    print(f'Avg Acc : {(sum / 5):.4f}%')

    loaders['test'] = DataLoader(test_ds, batch_size=128, num_workers=4, pin_memory=True)
    test_prediction(model.model, model.model_name, loaders['test'], test_ds)

def train_mask():

    root = '././input/data'
    transform = get_transformer(True)
    mask_train_ds = CustomDataset(root, transform['origin'], category='mask')
    mask_test_ds = MaskDataset(root, transform['origin'], train=False)

    # mask_train_aug1 = MaskDataset(root, transform['aug1'], age_flag=False)
    # mask_train_aug2 = MaskDataset(root, transform['aug2'], age_flag=False)
    # mask_train_aug3 = MaskDataset(root, transform['aug3'], age_flag=False)

    # check_image(mask_train_origin[0]['image'][0])
    # train_val = ConcatDataset([mask_train_origin, mask_train_aug1, mask_train_aug2, mask_train_aug3])

    # hyper params
    k_folds = 5
    num_epochs = 1
    num_classes = len(mask_train_ds.classes)

    # =========================== K-Fold ===========================
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # make models
    resnet101 = MaskModel('resnet101', num_classes, pretrained=True)

    # training
    common_train(kfold, resnet101, num_epochs, mask_train_ds, mask_test_ds)

def train_gender():
    root = '././input/data'
    transform = get_transformer(True)
    gender_train_ds = CustomDataset(root, transform['origin'], category='gender')
    gender_test_ds = MaskDataset(root, transform['origin'], train=False)

    # mask_train_aug1 = MaskDataset(root, transform['aug1'], age_flag=False)
    # mask_train_aug2 = MaskDataset(root, transform['aug2'], age_flag=False)
    # mask_train_aug3 = MaskDataset(root, transform['aug3'], age_flag=False)

    # check_image(mask_train_origin[0]['image'][0])
    # train_val = ConcatDataset([mask_train_origin, mask_train_aug1, mask_train_aug2, mask_train_aug3])

    # hyper params
    k_folds = 5
    num_epochs = 1
    num_classes = len(gender_train_ds.classes)

    # =========================== K-Fold ===========================
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # make models
    resnet101 = MaskModel('resnet101', num_classes, pretrained=True)

    # training
    common_train(kfold, resnet101, num_epochs, gender_train_ds, gender_test_ds)

def train_age():
    root = '././input/data'
    transform = get_transformer(True)
    age_train_ds = CustomDataset(root, transform['origin'], category='age')
    age_test_ds = MaskDataset(root, transform['origin'], train=False)

    # mask_train_aug1 = MaskDataset(root, transform['aug1'], age_flag=False)
    # mask_train_aug2 = MaskDataset(root, transform['aug2'], age_flag=False)
    # mask_train_aug3 = MaskDataset(root, transform['aug3'], age_flag=False)

    # check_image(mask_train_origin[0]['image'][0])
    # train_val = ConcatDataset([mask_train_origin, mask_train_aug1, mask_train_aug2, mask_train_aug3])

    # hyper params
    k_folds = 5
    num_epochs = 1
    num_classes = len(age_train_ds.classes)

    # =========================== K-Fold ===========================
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # make models
    resnet101 = MaskModel('resnet101', num_classes, pretrained=True)

    # training
    common_train(kfold, resnet101, num_epochs, age_train_ds, age_test_ds)

def main(config):
    # train_mask()
    train_age()
    # train_gender()

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