import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure(figsize=(18,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    
    return all_preds

def _confusion_matrix(all_preds, all_targets, classes):
    stacked = torch.stack((all_targets, all_preds), dim=1)
    cmt = torch.zeros(classes, classes, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    
    return cmt
    
def get_stacked(targets, preds):
    return torch.stack((targets, preds), dim=1)

def get_f1_score(targets, preds):
    pass

def get_confusion_matrix(targets, preds):
    pass