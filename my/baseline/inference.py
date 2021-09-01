import argparse
import os
import pandas as pd
import torch

from model import Ensemble
from utils import get_ans
from torch.utils.data import DataLoader
from importlib import import_module
from dataset import TestDataset, MaskBaseDataset

def load_model(saved_model, num_classes, device, args):
    model_cls = getattr(import_module("model"), args.model)
    
    # ==============================
    # 학습 시, 모델명 변경 주의
    # ==============================
    model = model_cls(
        model_name=args.model_name
    )
    model = torch.nn.DataParallel(model.model)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # -- ensemble flag
    # if args.fold_flag:
    #     models = []
    #     for i in range(5):
    #         model_path = os.path.join(saved_model, f'proc58/best_{i + 1}.pth')
    #         model.load_state_dict(torch.load(model_path, map_location=device))
    #         models.append(model)
    #     ensemble_model = Ensemble(device, models)
    #     return ensemble_model

    model_path = os.path.join(saved_model, 'proc71/best_4.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device, args).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join('/opt/ml/input/data/eval/info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print(" --- Calculating Inference Results --- ")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output_{args.model_name}_{args.task_type}.csv'), index=False)
    print(f' --- {args.task_type} Inference Done --- ')


# labeling for all task => age / mask / gender to one
def inference_combine(output_dir, args):
    gender_df = pd.read_csv(os.path.join(output_dir, f'output_{args.model_name}_gender.csv'), delimiter=',', encoding='utf-8-sig')
    mask_df = pd.read_csv(os.path.join(output_dir, f'output_{args.model_name}_mask.csv'), delimiter=',', encoding='utf-8-sig')
    age_df = pd.read_csv(os.path.join(output_dir, f'output_{args.model_name}_age.csv'), delimiter=',', encoding='utf-8-sig')

    gender_list = gender_df['ans'].tolist()
    mask_list = mask_df['ans'].tolist()
    age_list = age_df['ans'].tolist()

    ans = []
    for i in range(len(gender_list)):
        ans.append(get_ans(mask_list[i], gender_list[i], age_list[i]))

    gender_df['final_ans'] = ans
    gender_df.to_csv(os.path.join(output_dir, f'output_{args.model_name}_final.csv'), index=False)
    print(' --- Inference Combine Done --- ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validating (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    inference_combine(output_dir, args)