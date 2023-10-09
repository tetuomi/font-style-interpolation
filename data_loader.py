import os

import cv2
import pandas as pd

import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from utils import preprocessing

class LoadDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

def read_img(path, image_size):
    transform = Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    img = cv2.imread(path, 0)
    img = preprocessing(img, img_size=image_size, margin=5)

    return transform(img).float()

def make_data_list(image_size):
    data_list = {'train': [], 'test': []}
    df = pd.read_csv('csv_files/google_fonts_drop_none.csv')

    style_i = 0
    for data_type in ['train']:
        data_type_df = df[df['data_type'] == data_type].sample(100, random_state=0).reset_index(drop=True)
        for i in range(len(data_type_df)):
            p = os.path.join('../font2img/image', data_type_df.loc[i, 'font'])
            if os.path.isdir(p) == False: continue

            for j in range(26):
                img = read_img(os.path.join(p, chr(ord('A') + j) + '.png'), image_size)
                data_list[data_type] += [(img, j, style_i)]

            style_i += 1

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, image_size):
    data_list = make_data_list(image_size)

    train_dataset = LoadDataset(data_list['train'])
    test_dataset = LoadDataset(data_list['test'])

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader
