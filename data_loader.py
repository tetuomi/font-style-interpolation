import os
from glob import glob
from random import random

import cv2
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode

from fannet import FANnet
from style_encoder import StyleEncoder
from utils import preprocessing, preprocessing_myfonts


class LoadDataset(data.Dataset):
    def __init__(self, model, path_list, da_rate=0.3, image_size=64, margin=5, dataset_name='myfonts'):
        self.model = model
        self.path_list = path_list
        self.da_rate = da_rate
        self.image_size = image_size
        self.margin = margin
        self.dataset_name = dataset_name
        self.transform = Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda t: (t * 2) - 1)
                        ])
        self.da_transform = Compose([
                            # 垂直水平シフト
                            transforms.RandomAffine(degrees=0., translate=(0.2, 0.2), fill=1., interpolation=InterpolationMode.BILINEAR),
                            ])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path, label = self.path_list[index]
        if self.dataset_name == 'google_fonts':
            img = self.read_img(os.path.join(path, f"{chr(ord('A') + label)}.png"))
        elif self.dataset_name == 'myfonts':
            img = np.load(path)['arr_0'][label]
            img = self.transform(preprocessing_myfonts(img, img_size=self.image_size, margin=self.margin)).float()

        with torch.no_grad():
            feature = self.model.style_encode(((img+1)*0.5).unsqueeze(0)).squeeze(0)

        if random() < self.da_rate:
            img = self.da_transform(img)
        label = torch.tensor(label)

        return img, feature, label

    def read_img(self, path):
        transform = Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda t: (t * 2) - 1)
                            ])
        img = cv2.imread(path, 0)
        img = preprocessing(img, img_size=self.image_size, margin=self.margin)

        return transform(img).float()

def make_data_list_google_font(num_class):
    data_list = {'train': [], 'val': [], 'test': []}
    df = pd.read_csv('csv_files/google_fonts_drop_none.csv')

    for data_type in ['train', 'val', 'test']:
        if data_type == 'val':
            data_type_df = df[df['data_type'] == 'valid']
        else:
            data_type_df = df[df['data_type'] == data_type]

        for _, row in data_type_df.iterrows():
            p = os.path.join('../font2img/image', row['font'])
            if os.path.isdir(p) == False: continue

            for label in range(num_class):
                data_list[data_type] += [(p, label)]

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('VAL SIZE: {}'.format(len(data_list['val'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_list_myfonts(num_class):
    data_list = {'train': [], 'val': [], 'test': []}

    for data_type in ['train', 'val', 'test']:
        path_list = glob(f'../font2img/myfonts/{data_type}/*')

        for i in range(len(path_list)):
            for label in range(num_class):
                data_list[data_type] += [(path_list[i], label)]

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('VAL SIZE: {}'.format(len(data_list['val'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, image_size, num_class, encoder_path, dataset_name='google_fonts', da_rate=0., margin=5):
    print(f'DATASET NAME IS {dataset_name}')
    assert dataset_name in ['google_fonts', 'myfonts'], f'dataset_name must be google_fonts or myfonts. but {dataset_name} is given.'
    if dataset_name == 'google_fonts':
        path_list = make_data_list_google_font(num_class)
    elif dataset_name == 'myfonts':
        path_list = make_data_list_myfonts(num_class)

    if encoder_path == './weight/style_encoder_fannet.pth':
        model = FANnet(num_class)
    elif encoder_path == './weight/style_encoder_fannet2.pth':
        model = StyleEncoder(num_class)
    else:
        raise ValueError('encoder_path must be weight/style_encoder_fannet.pth or weight/style_encoder_fannet2.pth')

    model.to('cuda')
    model.load_state_dict(torch.load(encoder_path, map_location='cuda'))
    model.to('cpu')
    model.eval()

    train_dataset = LoadDataset(model, path_list['train'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)
    val_dataset = LoadDataset(model, path_list['val'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)
    test_dataset = LoadDataset(model, path_list['test'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader
