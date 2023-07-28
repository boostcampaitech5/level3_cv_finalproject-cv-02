import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import shutil
from argparse import ArgumentParser
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import os
from glob import glob

from model import MyModels
from dataset import InferenceDataset

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='efficientnetv2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--pretrained_model_name', type=str, default='22_efficientnetv2_0.985.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = A.Compose([
        A.Resize(225, 225),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_dataset = InferenceDataset(args.data_path, transform=transforms)

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        num_workers=4,
        shuffle=False
    )

    my_model = MyModels()
    model = getattr(my_model, args.model_name)().to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'pretrained', args.pretrained_model_name)))
    model.eval()

    result = []

    for img in tqdm(val_loader):
        img = img.to(device)

        output = torch.sigmoid(model(img))
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        output = output.detach().cpu().numpy().astype(np.uint8)
        result.append(output.flatten().tolist())

    result = sum(result, [])

    df = pd.read_csv('./data/total.csv', index_col=False)
    df['path'] = df['path'].str.replace('./save', '.')
    df['path'] = df['path'].str.replace("\\", '/')
    df['check'] = result
    
    # csv 저장 경로 생성 
    os.makedirs(os.path.join(args.save_path, 'csv'), exist_ok=True)

    # 저장
    df.to_csv(os.path.join(args.save_path, 'csv', './result.csv'), index=False)



if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)