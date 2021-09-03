import argparse
import glob
import json
import os
import random
import re
from sklearn.metrics import f1_score
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SEED = 123

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_dataloader(dataset, train_idx, valid_idx, args):
    train_set = torch.utils.data.Subset(dataset, indices=train_idx)
    val_set = torch.utils.data.Subset(dataset, indices=valid_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last = True,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False
    )

    return train_loader, val_loader

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def k_fold_train(data_dir, model_dir, args):

    s = "{:=^100}".format(" start k-fold training ")
    print(s)
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- define single or multi model
    task_list = None
    if args.single:
        task_list = ['multi']
    else:
        task_list = ['mask', 'gender', 'age']
    
    for task in task_list:

        print("="*30, f"current task is: '{task}'", "="*32)
        # -- dataset
        dataset_moduel = getattr(import_module("dataset"), args.dataset)
        dataset = dataset_moduel(
            data_dir = data_dir,
            class_by = task if task != "multi" else None,
        )
        num_classes = dataset.num_classes

        # -- augmentation
        transform_module = getattr(import_module("dataset"), args.augmentation)
        transform = transform_module(
            resize = args.resize,
            mean = dataset.mean,
            std = dataset.std,
        )
        dataset.set_transform(transform)

        # -- k-fold
        accumulation_step = args.accumulation_step
        skf_module = getattr(import_module("sklearn.model_selection"), "StratifiedKFold")
        skf = skf_module(args.fold_nums)

        labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

        for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            
            s = "{:=^100}".format(f" k-fold: {i}/{args.fold_nums} ")
            print(s)
            # -- data_loader
            train_loader, val_loader = get_dataloader(dataset, train_idx, valid_idx, args)

            # -- model
            model_module = import_module("model")
            model = model_module.get_model(args.model, num_classes).to(device)
            #model = torch.nn.DataParallel(model)

            # -- loss & metric
            criterion = import_module("loss").create_criterion(args.criterion)
            opt_module = getattr(import_module("torch.optim"), args.optimizer)
            optimizer = opt_module(
                model.parameters(),
                lr=args.lr,
                weight_decay=5e-4,
            )
            scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

            # -- logging
            logger = SummaryWriter(log_dir=save_dir)
            with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

            # -- training
            best_val_acc = 0
            best_val_loss = np.inf
            counter = 0
            best_f1 = 0

            for epoch in range(args.epochs):

                # train loop
                model.train()
                loss_value = 0
                matches = 0
                epoch_f1 = 0
                n_iter = 0

                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    # -- Gradient Accumulation
                    if (idx+1) % accumulation_step == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_f1 += f1_score(labels.cpu().numpy(),preds.cpu().numpy(),average='macro')
                    n_iter += 1
                    loss_value += loss.item()
                    matches += (preds==labels).sum().item()

                    if(idx+1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        current_lr = get_lr(optimizer)
                        print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} ||training-f1 {epoch_f1/n_iter:.4f}|| lr {current_lr}"
                )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation result...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []

                    epoch_f1 = 0
                    n_iter = 0

                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                        epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                        n_iter += 1

                    epoch_f1 = epoch_f1/n_iter
                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(valid_idx)
                    best_val_loss = min(best_val_loss, val_loss)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        print(f"New best model for val accuracy ! : {val_acc:4.2%} saving the best model ...")
                        torch.save(model, f"{save_dir}/{task}_{i:02}_{epoch:03}_{val_acc:4.2%}_{val_loss:4.2}.pt")
                        counter = 0
                    else:
                        counter += 1
                    
                    print(
                        f"current val acc : {val_acc:4.2%}, loss: {val_loss:4.2} ,f1_score : {epoch_f1:.4f}|| "
                        f"best  val acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} , best_f1_score : {best_f1:.4f}"
                    )
                    if args.early_stop and counter == args.patience:
                        print("early stopping")
                        print(f"best acc: {best_val_acc:4.2%}") 

                        break

def general_train(data_dir, model_dir, args):
    s = "{:=^100}".format(" start general training ")
    print(s)
    seed_everything(args.seed)
    
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- define single or multi model
    task_list = None
    if args.single: # True: multi model, False: single model
        task_list = ['multi']
    else:
        task_list = ['mask', 'gender', 'age']

    for task in task_list:

        s = "{:=^100}".format(f"current task is: '{task}'")
        print(s)
        # -- dataset
        dataset_module = getattr(import_module("dataset"), args.dataset) # default: MaskSplitByProfileDataset
        dataset = dataset_module(
            data_dir = data_dir,
            class_by = task if task != "multi" else None, # if 'class_by' is 'None', inject 'None' to class_by
        )
        num_classes = dataset.num_classes
        # -- augmentation
        transform_module = getattr(import_module("dataset"), args.augmentation) # default: CustomAugmentaion
        transform = transform_module(
            resize = args.resize,
            mean = dataset.mean,
            std = dataset.std,
        )
        dataset.set_transform(transform)

        # -- data_loader
        train_set, val_set = dataset.split_dataset()

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = import_module("model") # default: Efficientnet B4
        model = model_module.get_model(args.model, num_classes).to(device)
        #model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = import_module("loss").create_criterion(args.criterion)
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            model.parameters(),
            lr=args.lr,
            weight_decay=5e-4,
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
        # -- training  
        best_val_acc = 0
        best_val_loss = np.inf
        counter = 0
        best_f1 = 0
        for epoch in range(args.epochs):

            # train loop
            model.train()
            loss_value = 0
            matches = 0
            epoch_f1 = 0
            n_iter = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                epoch_f1 += f1_score(labels.cpu().numpy(),preds.cpu().numpy(),average='macro')
                n_iter += 1
                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1)% args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} ||training-f1 {epoch_f1/n_iter:.4f}|| lr {current_lr}"
                )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                epoch_f1 = 0
                n_iter = 0

                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    n_iter += 1
                
                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"New best model for val accuracy ! : {val_acc:4.2%} saving the best model ...")
                    torch.save(model, f"{save_dir}/{task}_00_{epoch:03}_{val_acc:4.2%}_{val_loss:4.2}.pt")
                    counter = 0
                else:
                    counter += 1

                print(
                        f"current val acc : {val_acc:4.2%}, loss: {val_loss:4.2} ,f1_score : {epoch_f1:.4f}|| "
                        f"best  val acc : {best_val_acc:4.2%}, best loss: a{best_val_loss:4.2} , best_f1_score : {best_f1:.4f}"
                    )
                if args.early_stop and counter >= args.patience:
                    print("early stopping")
                    print(f"best acc: {best_val_acc:4.2%}") 
                    counter = 0
                    break
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 256], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='efficientnet_b4', help='model type (default: efficientnet_b4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--single', default=False, action='store_true', help='selecting multiple(True) or single(False) class classification model (default: True)')

    # Hyperparameter environment
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')

    # K-Fold Environment
    parser.add_argument('--k_fold', default=False, action='store_true', help='selecting wether apply k-fold or not (default: False)')
    parser.add_argument('--fold_nums', type=int, default=5, help='how many make fold (default: 5)')
    parser.add_argument('--accumulation_step', type=int, default=2, help='loss grad step (default:2)')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/eval/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # Custom Environment
    parser.add_argument('--early_stop', default=True, action='store_false', help='early stopping, if it no more get a better from training (default: Tre)')
    parser.add_argument('--patience', type=int, default=3, help='a variable for early stopping (default:3)')
    parser.add_argument('--num_workers', type=int, default=4, help='num worker for dataloader (default: 2)')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    # diverging point (k-fold or general)
    if args.k_fold:
        k_fold_train(data_dir, model_dir, args)
    else:
        general_train(data_dir, model_dir, args)