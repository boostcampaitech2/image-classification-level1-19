import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

# def load_model(saved_model, num_classes, device):
#     model_cls = getattr(import_module("model"), args.model)
#     model = model_cls(
#         num_classes=num_classes
#     )

#     # tarpath = os.path.join(saved_model, 'best.tar.gz')
#     # tar = tarfile.open(tarpath, 'r:gz')
#     # tar.extractall(path=saved_model)

#     model_path = os.path.join(saved_model, 'best.pth')
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     return model
def encode_multi_class(mask_label, gender_label, age_label) -> int:
    return mask_label * 6 + gender_label * 3 + age_label

def get_info(model):
    task, k, epoch, acc, loss = model.split('_') # task, k-fold split, epoch, accuracy, loss
    acc = acc.replace("%", "")
    return int(k), float(acc)

def is_single(args):
    tasks = []
    model_list = [model for model in os.listdir(args.model_dir) if not model.startswith('.')]
    model_list = [model for model in model_list if model.endswith('pt')]
    
    for model in model_list:
        task, k, epoch, acc, loss = model.split('_')
        tasks.append(task)
    tasks = set(tasks)

    return True if 'multi' in tasks else False

def get_n_splits(args):
    n_split=0
    model_list = [model for model in os.listdir(args.model_dir) if not model.startswith('.')]
    model_list = [model for model in model_list if model.endswith('pt')]
    for model in model_list:
        task, k, epoch, acc, loss = model.split('_')
        n_split = max(n_split, int(k))
    
    return n_split + 1

def get_best_models(task, args):
    model_list = [model for model in os.listdir(args.model_dir) if model.startswith(task)]
    best_model = {}
    for model in model_list:
        k, acc = get_info(model)
        try:
            _, _acc = get_info(best_model[int(k)])
            if acc > _acc:
                best_model[k] = model
        except:
            best_model[k] = model

    return list(best_model.values())

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):

    s = "{:=^100}".format(f" start inference for {model_dir} ")
    print(s)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    # -- test dataset
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- define task
    tasks = ["mask", "gender", "age"] if not is_single(args) else ['multi']

    # -- inference
    n_splits = get_n_splits(args)
    combine = {}
    
    s = "{:=^100}".format(f" fold nums: {n_splits}, single model: {is_single(args)} ")
    print(s)
    for task in tasks:
        s = "{:=^100}".format(f" current working task: {task} ")
        print(s)
        best_models = get_best_models(task, args)
        oof_pred = None

        for best_model in best_models:
            s = "{:=^100}".format(f" calculating using : {best_model} ")
            print(s)
            best_model = torch.load(os.path.join(args.model_dir, best_model)).to(device)
            best_model.eval()

            all_predictions = []
            with torch.no_grad():
                for images in loader:
                    images = images.to(device)
                    pred = best_model(images)
                    all_predictions.extend(pred.cpu().numpy())
                
                fold_pred = np.array(all_predictions)
            if oof_pred is None:
                oof_pred = fold_pred / n_splits
            else:
                oof_pred += fold_pred / n_splits

        combine[task] = np.argmax(oof_pred, axis=1)

    # -- calculate 
    if is_single(args):
        info['ans'] = combine['multi']
    else:
        multi_class = []
        for mask, gender, age in zip(combine['mask'], combine['gender'], combine['age']):
            multi_class.append(encode_multi_class(mask, gender, age))
        info['ans'] = multi_class
    info.to_csv(os.path.join(args.model_dir, 'submission.csv'), index=False)

    s = "{:=^100}".format(" Inference Done! ")
    print(s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    
def main():
    pass