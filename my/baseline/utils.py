import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

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

# def plot_confusion_matrix(cm):
#     classes = [i for i in range(18)]
#     plt.figure()
#     plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues)
#     plt.xticks(18, classes, fontsize=16)
#     plt.yticks(18, classes, fontsize=16)
#     plt.xlabel('Predicted Label', fontsize=18)
#     plt.ylabel('True Label', fontsize=18)
#     plt.show()

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