import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms

from copy import copy
from torchsummary import summary
from new.new_dataset import MaskDataset
from importlib import import_module
from pathlib import Path
from dataset import MaskBaseDataset
from inference import inference, inference_combine
from loss import create_criterion
from sklearn.model_selection import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import *

std = (0.5, 0.5, 0.5)
mean = (0.2, 0.2, 0.2)

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

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

# auto increment for path
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

# Starting Point
def common_train(data_dir, model_dir, args):
    global std, mean

    # -- reproducibility
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio=args.val_ratio,
        task_type=args.task_type
    )
    print(' ============ original dataset : ', len(dataset))

    num_classes = dataset.num_classes  # 18
    print(' ============ num classes : ', num_classes)

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        crop=(args.img_height, args.img_width)
    )

    # -- std and mean of RGB
    std = dataset.std
    mean = dataset.mean

    # -- original not transformed dataset
    dataset.set_transform(transform, False)

    # -- original => train : val
    train_set, val_set = dataset.split_dataset()
    val_set, test_set = train_test_split(val_set, test_size=0.5, shuffle=True, random_state=42)

    # -- additional transformed dataset
    additional_dataset = copy(dataset)
    additional_dataset.set_transform(get_train_transform(additional_dataset.mean, additional_dataset.std, args), False)

    additional_train_set, additional_val_set = additional_dataset.split_dataset(True)
    print(' ============ transformed dataset : ', len(additional_train_set))

    # -- original + transformed => train_set
    train_set = ConcatDataset([train_set, additional_train_set])
    print(' ============ final dataset : ', len(train_set))

    # -- train general or k-fold
    if args.fold_flag:
        fold_train(train_set, test_set, save_dir, dataset_module, num_classes, args)
    else:
        general_train(train_set, val_set, test_set, save_dir, dataset_module, num_classes, args)

# K-Fold CV
def fold_train(train_set, test_set, save_dir, dataset_module, num_classes, args):
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    print(' --- K-Fold Start --- ')
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        len_valid = len(val_sampler)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, pin_memory=True, sampler=val_sampler)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, pin_memory=True)

        print(' ============ (K-Fold) train loader length : ', len(train_loader))
        print(' ============ (K-Fold) val loader length : ', len(val_loader))
        print(' ============ (K-Fold) test loader length : ', len(test_loader))

        # -- training
        train_model(train_loader, val_loader, test_loader, len_valid, save_dir, args, dataset_module, num_classes, fold_num=fold_num)
        print(f' --- K-Fold_{fold_num} End --- ')

# General
def general_train(train_set, val_set, test_set, save_dir, dataset_module, num_classes, args):
    use_cuda = torch.cuda.is_available()

    # -- elderly increase is True
    if args.age_flag:
        # --- elderly dataset
        elderly_dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
        elderly_dataset = elderly_dataset_module(
            data_dir=data_dir,
            val_ratio=args.val_ratio,
            task_type=args.task_type,
            age_flag=args.age_flag
        )

        print(' ============ elderly dataset : ', len(elderly_dataset))

        elderly_dataset.set_transform(transform, True)
        elderly_train_set, elderly_val_set = elderly_dataset.split_dataset()
        elderly_val_set, elderly_test_set = train_test_split(elderly_val_set, test_size=0.5, shuffle=True, random_state=42)

        train_set = ConcatDataset([train_set, elderly_train_set])
        val_set = ConcatDataset([val_set, elderly_val_set])
        test_set = ConcatDataset([test_set, elderly_test_set])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        # drop_last=True,
        # sampler=weighted_sampler
    )

    print(' ============ train loader length : ', len(train_loader))

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        # drop_last=True,
    )

    print(' ============ val loader length : ', len(val_loader))

    # test set for prediction and score
    test_loader = DataLoader(
        test_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
    )

    print(' ============ test loader length : ', len(test_loader))
    len_valid = len(val_set)

    # -- training
    train_model(train_loader, val_loader, test_loader, len_valid, save_dir, args, dataset_module, num_classes)

# Real Training
def train_model(train_loader, val_loader, test_loader, len_valid, save_dir, args, dataset_module, num_classes=18, fold_num=1):
    global std, mean

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: MyModel
    model = model_module(
        model_name=args.model_name
    ).to(device)

    model = model.model

    # --- init params
    init_fc_params(model)

    # --- freezing strategy
    # init_freezing(model)

    # --- model summary
    summary(model, (3, args.img_height, args.img_width))
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.9)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):

        # train loop
        model.train()
        loss_value = 0
        matches = 0
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

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)

                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        # each epoch
        scheduler.step()

        # val loop
        with torch.no_grad():
            print("[Info] --- Calculating Validation Result ---")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None

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

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, mean, std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len_valid

            if args.early_stopping and val_loss >= best_val_loss:
                print(' --- Early Stopping ---')
                break

            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model")

                if not args.fold_flag:
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                else:
                    torch.save(model.state_dict(), f"{save_dir}/best_{fold_num}.pth")
                best_val_acc = val_acc

            if not args.fold_flag:
                torch.save(model.state_dict(), f"{save_dir}/last.pth")
            else:
                torch.save(model.state_dict(), f"{save_dir}/last_{fold_num}.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()

    # test prediction
    test_prediction(model, test_loader, num_classes=num_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='MyModel', help='model type (default: MyModel)')

    # -- add custom args
    parser.add_argument('--model_name', type=str, default='resnet18', help='model detail type (default: resnet18)')
    parser.add_argument('--early_stopping', type=int, default=1, help='early stopping flag (default: True)')
    parser.add_argument('--img_width', type=int, default=200, help='image width to resize (default: 200)')
    parser.add_argument('--img_height', type=int, default=250, help='image height to resize (default: 250)')
    parser.add_argument('--task_type', type=str, default='all', help='task type whether is all or not (default: all)')
    parser.add_argument('--age_flag', type=int, default=0, help='selection of age >= 60 (default: False)')
    parser.add_argument('--fold_flag', type=int, default=0, help='fold flag (default: False)')
    parser.add_argument('--k_folds', type=int, default=5, help='split ratio (default: 5)')

    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--train_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--test_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.train_data_dir
    model_dir = args.model_dir
    common_train(data_dir, model_dir, args)

    # inference
    data_dir = args.test_data_dir
    output_dir = '/opt/ml/input/data/eval/submission'
    # inference(data_dir, model_dir, output_dir, args)
    # inference_combine(output_dir, args)