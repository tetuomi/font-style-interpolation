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

from utils import preprocessing, preprocessing_myfonts
from style_encoder import StyleEncoder

transform = Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

class LoadDataset(data.Dataset):
    def __init__(self, data_list, da_rate):
        self.data_list = data_list
        self.da_rate = da_rate
        self.da_transform = Compose([
                            # 垂直水平シフト
                            transforms.RandomAffine(degrees=0., translate=(0.2, 0.2), fill=1., interpolation=InterpolationMode.BILINEAR),

                            # 垂直水平シフト＋拡大縮小
                            # transforms.Lambda(lambda t: 1 - (t + 1) * 0.5), # -1 -> 1, 1 -> 0
                            # transforms.RandomAffine(degrees=0., scale=(0.8, 1.2), translate=(0., 0.2), fill=0., interpolation=InterpolationMode.BILINEAR),
                            # transforms.Lambda(lambda t: 2 * (1 - t) - 1), # 1 -> -1, 0 -> 1
                        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.data_list[index][0]
        if random() < self.da_rate:
            img = self.da_transform(img)
        feature = self.data_list[index][1]
        return img, feature

def read_img(path, image_size, margin=5):
    img = cv2.imread(path, 0)
    img = preprocessing(img, img_size=image_size, margin=margin)

    return transform(img).float()

def make_data_list_google_font(image_size, num_style, encoder_path, encoder_zdim, margin=5):
    data_list = {'train': [], 'test': []}
    df = pd.read_csv('csv_files/google_fonts_drop_none.csv')

    model = StyleEncoder(encoder_zdim)
    model.load_state_dict(torch.load(encoder_path))
    model.eval()

    style_i = 0
    for data_type in ['train']:
        data_type_df = df[df['data_type'] == data_type]

        for i in range(len(data_type_df)):
            if num_style <= style_i: break
            p = os.path.join('../font2img/image', data_type_df.loc[i, 'font'])
            if os.path.isdir(p) == False: continue

            img = read_img(os.path.join(p, 'A.png'), image_size, margin=margin)
            with torch.no_grad():
                feature = model(((img+1)*0.5).unsqueeze(0)).squeeze(0)
            data_list[data_type] += [(img, feature)]

            style_i += 1

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_list_myfonts(image_size, num_style, encoder_path, encoder_zdim, margin=5):
    data_list = {'train': [], 'test': []}

    model = StyleEncoder(encoder_zdim)
    model.load_state_dict(torch.load(encoder_path))
    model.eval()

    style_i = 0
    for data_type in ['train']:
        path_list = glob(f'../font2img/myfonts/{data_type}/*')

        for i in range(len(path_list)):
            if num_style <= style_i: break

            img = np.load(path_list[i])['arr_0'][0] # A のみ
            img = transform(preprocessing_myfonts(img, img_size=image_size, margin=margin)).float()
            with torch.no_grad():
                feature = model(((img+1)*0.5).unsqueeze(0)).squeeze(0)
            data_list[data_type] += [(img, feature)]

            style_i += 1

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, image_size, num_style, encoder_path, encoder_zdim, dataset_name='google_fonts', da_rate=0., margin=5):
    print(f'DATASET NAME IS {dataset_name}')
    assert dataset_name in ['google_fonts', 'myfonts'], f'dataset_name must be google_fonts or myfonts. but {dataset_name} is given.'
    if dataset_name == 'google_fonts':
        data_list = make_data_list_google_font(image_size, num_style, encoder_path, encoder_zdim, margin=margin)
    elif dataset_name == 'myfonts':
        data_list = make_data_list_myfonts(image_size, num_style, encoder_path, encoder_zdim, margin=margin)

    train_dataset = LoadDataset(data_list['train'], da_rate=da_rate)
    test_dataset = LoadDataset(data_list['test'], da_rate=da_rate)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader
