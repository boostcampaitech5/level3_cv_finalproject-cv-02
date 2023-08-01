import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import pandas as pd
import cv2
from glob import glob


class TrainDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='train'):
        self.train_data_path = os.path.join(data_path, f'{mode}')
        self.img_file_list = glob(self.train_data_path+'/*/*.png')

        self.transform = transform

    def __getitem__(self, idx):

        img_path = self.img_file_list[idx]
        image = cv2.imread(img_path)
    
        if self.transform is not None:
            image = self.transform(image=image)['image']

        if img_path.split('/')[-2] == 'yes':
            label = torch.FloatTensor([1])
        else:
            label = torch.FloatTensor([0])

        return image, label

    def __len__(self):
        return len(self.img_file_list)


class InferenceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path= data_path
        self.csv_path = os.path.join(data_path, 'total.csv')
        self.df = pd.read_csv(self.csv_path, index_col=0)['path'].to_numpy()


        
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.df[idx].replace('./save', './data/save').replace("\\", '/')
        
        image = cv2.imread(img_path)
        if image is None:
            print(img_path)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image
    
    def __len__(self):
        return len(self.df)


    
