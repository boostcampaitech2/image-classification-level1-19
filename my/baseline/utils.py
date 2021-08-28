import matplotlib.pyplot as plt
import torch
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm


def test_prediction(model, test_loader):
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

    calculate_score(y_true, y_pred)
    print('================= Done =================')

def calculate_score(y_true, y_pred):
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
    print(df_classification_report(y_true, y_pred))

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

def df_classification_report(y_true, y_pred):
    classes = [i for i in range(18)]
    report = classification_report(y_true, y_pred, output_dict=True, target_names=classes)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
