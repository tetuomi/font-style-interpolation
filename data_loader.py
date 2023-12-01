import os

import cv2
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from utils import preprocessing
from style_encoder import StyleEncoder

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

def make_data_list(image_size, num_style, encoder_path, encoder_zdim):
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

            img = read_img(os.path.join(p, 'A.png'), image_size)
            with torch.no_grad():
                feature = model(((img+1)*0.5).unsqueeze(0)) ## 作業中止
            data_list[data_type] += [(img, feature)]

            style_i += 1

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, image_size, num_style, encoder_path, encoder_zdim):
    data_list = make_data_list(image_size, num_style, encoder_path, encoder_zdim)

    train_dataset = LoadDataset(data_list['train'])
    test_dataset = LoadDataset(data_list['test'])

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader
