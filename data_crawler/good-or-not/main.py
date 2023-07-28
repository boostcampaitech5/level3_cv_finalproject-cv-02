import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from argparse import ArgumentParser
from tqdm import tqdm
import random
import numpy as np
import os

from model import MyModels
from dataset import TrainDataset

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default='efficientnetv2')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./data/vton/')
    parser.add_argument('--save_path', type=str, default='./save/')
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

    train_transforms = A.Compose([
        A.Resize(225, 225),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.OneOf([
            A.GaussNoise(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Downscale(always_apply=False, p=0.2, scale_min=0.25, scale_max=0.25, interpolation=0),
        ], p=0.3),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(225, 225),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


    train_dataset = TrainDataset(data_path = args.data_path, transform=train_transforms, mode='train')
    val_dataset = TrainDataset(data_path = args.data_path, transform=val_transforms, mode='test')

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers= 4,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers= 4,
        drop_last=True
    )

    my_model = MyModels()

    model = getattr(my_model, args.model_name)()
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)

    #train
    model = model.to(device)

    
    best_acc = -1
    for epoch in tqdm(range(1, args.epoch+1)):
        model.train()

        total_loss = 0.
        for step, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)

            ## forward 
            output = torch.sigmoid(model(image))

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step+1) % 5 == 0:
                print(
                    f'Epoch : [{epoch}/{args.epoch}], '
                    f'Step : [{step+1}/{len(train_loader)}], '
                    f'Loss : {round(loss.item(), 4)}'
                )   
        
        print('-'*80)
        print(
            f'Epoch : {epoch}, train/mean_epoch_loss : {round(total_loss/args.batch_size, 4)}'
        )

        #val
        print(f'Validation : {epoch}')
        model.eval()

        accuracy = 0
        total_loss = 0

        with torch.no_grad():
            for step, (image, label) in enumerate(val_loader):
                image, label = image.to(device), label.to(device)

                output = torch.sigmoid(model(image))

                loss = criterion(output, label)
                total_loss += loss.item()

                output[output >= 0.5] = 1
                output[output < 0.5] = 0

                accuracy += (output==label).sum().item()

            
            print(
                f'Val Accuracy : {round(accuracy / (len(val_loader)*args.batch_size), 4)} ~ '
                f'Val Mean_loss : {round(total_loss/(step+1), 4)}'
            )

            if accuracy / len(val_loader) > best_acc:
                best_acc = accuracy / len(val_loader)
                torch.save(model.state_dict(), os.path.join(args.save_path, 'pretrained', f'{epoch}_{args.model_name}_{round(accuracy / (len(val_loader)*args.batch_size), 4)}.pth'))
                print(f'Save New model in {os.path.join(args.save_path, "pretrained")}')

            scheduler.step(loss)


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)