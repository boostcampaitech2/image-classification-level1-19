import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm

def test_prediction(model, test_loader, num_classes=18):
    print('=============== Start Test Prediction ===============')
    device = 'cuda'
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            y_pred += output.argmax(dim=-1).tolist()
            y_true += labels.tolist()

    calculate_score(y_true, y_pred, num_classes)
    print('================= Done =================')

def calculate_score(y_true, y_pred, num_classes=18):
    # plot_cm(confusion_matrix(y_true, y_pred))
    print('==' * 30)
    print('confusion_matrix')
    print('==' * 30)
    print(confusion_matrix(y_true, y_pred))
    print('==' * 30)
    print('f1_score')
    print('==' * 30)
    print(f1_score(y_true, y_pred, average=None))
    print('==' * 30)
    print('accuracy_score')
    print('==' * 30)
    print(accuracy_score(y_true, y_pred))
    print('==' * 30)
    print('precision_score')
    print('==' * 30)
    print(precision_score(y_true, y_pred, average=None))
    print('==' * 30)
    print('recall_score')
    print('==' * 30)
    print(recall_score(y_true, y_pred, average=None))
    print('==' * 30)
    print('classification_report')
    print('==' * 30)
    print(df_classification_report(y_true, y_pred, num_classes=num_classes))

def plot_cm(cm):
    classes = [i for i in range(18)]
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues)
    plt.xticks(range(18), classes, fontsize=16)
    plt.yticks(range(18), classes, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.show()

def df_classification_report(y_true, y_pred, num_classes=18):
    classes = [i for i in range(num_classes)]
    report = classification_report(y_true, y_pred, output_dict=True, target_names=classes)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

def init_fc_params(model, init_type='xavier'):
    # weights init
    if init_type == 'xavier':
        nn.init.xavier_uniform_(model.fc[1].weight)
    elif init_type == 'kaiming':
        nn.init.kaiming_uniform_(model.fc[1].weight)

    # bias init
    stdv = 1. / np.sqrt(model.fc[1].weight.size(1))
    model.fc[1].bias.data.uniform_(-stdv, stdv)

def init_freezing(model, child_num=6):
    ct = 0
    for child in model.children():
        ct += 1
        if ct < child_num:
            for param in child.parameters():
                param.requires_grad = False

def add_hparams_to_tensorboard(save_dir):
    config_list = [
        'seed', 'epochs', 'dataset', 'augmentation', 'batch_size',
        'optimizer', 'lr', 'val_ratio', 'criterion', 'lr_decay_step'
    ]

    final_config = dict()
    config = json.load(open(os.path.join(save_dir, 'config.json')))
    for k, v in config.items():
        if k in config_list:
            final_config[k] = v

    with SummaryWriter() as w:
        w.add_hparams(final_config, {'hparam/accuracy': 0, 'hparam/loss': 0})

# labeling for all task
def get_ans(mask, gender, age):
    if mask == 0:
        if gender == 0 and age == 0:
            return 0
        if gender == 0 and age == 1:
            return 1
        if gender == 0 and age == 2:
            return 2
        if gender == 1 and age == 0:
            return 3
        if gender == 1 and age == 1:
            return 4
        if gender == 1 and age == 2:
            return 5
    elif mask == 1:
        if gender == 0 and age == 0:
            return 6
        if gender == 0 and age == 1:
            return 7
        if gender == 0 and age == 2:
            return 8
        if gender == 1 and age == 0:
            return 9
        if gender == 1 and age == 1:
            return 10
        if gender == 1 and age == 2:
            return 11
    else:
        if gender == 0 and age == 0:
            return 12
        if gender == 0 and age == 1:
            return 13
        if gender == 0 and age == 2:
            return 14
        if gender == 1 and age == 0:
            return 15
        if gender == 1 and age == 1:
            return 16
        if gender == 1 and age == 2:
            return 17

# -- get label list
def get_label_list(dataset):
    label_list = []
    for ds in tqdm(dataset):
        label_list.append(ds[1])
    return label_list

# get train dataset transform => lots of transform
def get_train_transform(mean, std, args):
    return transforms.Compose([
        transforms.RandomResizedCrop((args.img_height, args.img_width), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# -- sub transform for train
def get_train_transform_second(mean, std, args):
    return transform.Compose([
         transform.Resize((args.img_height, args.img_width)),
         transform.RandomHorizontalFlip(),
         transform.RandomRotation(10),
         transform.RandomAffine(translate=(0.05, 0.05), degrees=0),
         transform.ToTensor(),
         transform.RandomErasing(inplace=True, scale=(0.01, 0.23)),
         transform.Normalize(mean=mean, std=std)
    ])

# -- sub transform for train
def get_train_transform_third(mean, std, args):
    return transform.Compose([
        transform.Resize((args.img_height, args.img_width)),
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomRotation(15),
        transform.RandomAffine(translate=(0.08, 0.1), degrees=15),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

# get valid transform => at least transform
def get_valid_transform(mean, std, args):
    return transforms.Compose([
        transforms.RandomResizedCrop((args.img_height, args.img_width), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# def undersampling_df(df:pd.DataFrame):
#     drop_list = []
#     for drop_idx in df.index:
#         if df.loc[drop_idx]['label'] == 0:
#             if np.random.randint(6) != 0:
#                 drop_list.append(drop_idx)
#         elif df.loc[drop_idx]['label'] == 1:
#             if np.random.randint(5) != 0:
#                 drop_list.append(drop_idx)
#         elif df.loc[drop_idx]['label'] == 3:
#             if np.random.randint(9) != 0:
#                 drop_list.append(drop_idx)
#         elif df.loc[drop_idx]['label'] == 4:
#             if np.random.randint(10) != 0:
#                 drop_list.append(drop_idx)
#         elif df.loc[drop_idx]['label'] in [9, 10, 12, 15, 16]:
#             if np.random.randint(2) != 0:
#                 drop_list.append(drop_idx)
#     return drop_list

# def train_by_df():
#     data_dir = '/opt/ml/input/data/train'
#     img_dir = f'{data_dir}/images'
#     df_path = f'{data_dir}/new_train.csv'
#     df = pd.read_csv(df_path, delimiter=',', encoding='utf-8-sig')
#
#     drop_list = undersampling_df(df)
#     drop_df = df.drop(drop_list)
#
#     train_df, val_df = train_test_split(drop_df, test_size=0.2, shuffle=True, random_state=42)