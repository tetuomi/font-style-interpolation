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


transform = Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda t: (t * 2) - 1)
                    ])

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
        img, feature, label = self.path_list[index]

        if random() < self.da_rate:
            img = self.da_transform(img)

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

def make_data_list_myfonts(num_class, model, image_size, device, margin=5):
    data_list = {'train': [], 'val': [], 'test': []}

    for data_type in ['train', 'val', 'test']:
        path_list = glob(f'../font2img/myfonts/{data_type}/*')

        imgs = []
        for i in range(len(path_list)):
            for label in range(num_class):
                img = np.load(path_list[i])['arr_0'][label]
                img = transform(preprocessing_myfonts(img, img_size=image_size, margin=margin)).float()
                imgs.append(img)

        imgs = torch.cat(imgs).unsqueeze(1)
        b = 512
        feats = []
        with torch.no_grad():
            for i in range(0, imgs.size(0), b):
                feat = model.style_encode((imgs[i:i+b].to(device)+1)*0.5) # encoderは0-1入力
                feats.append(feat.cpu().detach().clone())
        
        feats = torch.cat(feats)

        for i in range(imgs.size(0)):        
            data_list[data_type].append((imgs[i], feats[i], torch.tensor(i%num_class)))

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('VAL SIZE: {}'.format(len(data_list['val'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, image_size, num_class, encoder_path, device, dataset_name='google_fonts', da_rate=0., margin=5):
    encoder_name = encoder_path.split('_')[-1].split('.')[0]
    print(f'DATASET NAME IS {dataset_name}')
    print(f'STYLE ENCODER IS {encoder_name}')
    
    if encoder_name == 'fannet':
        model = FANnet(num_class)
    elif encoder_name == 'fannet2':
        model = StyleEncoder(num_class)
    else:
        raise ValueError('encoder_path must be weight/style_encoder_fannet.pth or weight/style_encoder_fannet2.pth')

    model.to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.eval()
    
    assert dataset_name in ['google_fonts', 'myfonts'], f'dataset_name must be google_fonts or myfonts. but {dataset_name} is given.'
    if dataset_name == 'google_fonts':
        # path_list = make_data_list_google_font(num_class)
        pass
    elif dataset_name == 'myfonts':
        path_list = make_data_list_myfonts(num_class, model, image_size, device, margin=margin)


    train_dataset = LoadDataset(model, path_list['train'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)
    val_dataset = LoadDataset(model, path_list['val'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)
    test_dataset = LoadDataset(model, path_list['test'], da_rate=da_rate, image_size=image_size, margin=margin, dataset_name=dataset_name)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader
