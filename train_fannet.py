import os
import json
import random
import requests
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from fannet import FANnet
from style_encoder import StyleEncoder


class LoadDataset(data.Dataset):
    def __init__(self, data_list, num_char=26, image_size=64, margin=5):
        self.data_list = data_list
        self.num_char = num_char
        self.image_size = image_size
        self.margin = margin

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = self.data_list[index]['path']
        char = self.data_list[index]['char']
        img = np.load(path)['arr_0'][char]
        img = torch.tensor(preprocessing_myfonts(img, img_size=self.image_size, margin=self.margin))
        img = img.unsqueeze(0).float()

        gt_label = random.randint(0, self.num_char-1)
        gt_img = np.load(path)['arr_0'][gt_label]
        gt_img = torch.tensor(preprocessing_myfonts(gt_img, img_size=self.image_size, margin=self.margin))
        gt_img = gt_img.unsqueeze(0).float()

        gt_onehot_label = nn.functional.one_hot(torch.tensor(gt_label), num_classes=self.num_char).float()

        return img, gt_img, gt_onehot_label

def make_data_list(num_char):
    data_list = {'train': [], 'val': [], 'test': []}

    for data_type in ['train', 'val', 'test']:
        path_list = glob(f'../font2img/myfonts/{data_type}/*')

        for i in range(len(path_list)):
            data_list[data_type] += [{'path': path_list[i], 'char': c} for c in range(num_char)]

    print('TRAIN SIZE: {}'.format(len(data_list['train'])))
    print('VALID SIZE: {}'.format(len(data_list['val'])))
    print('TEST SIZE: {}'.format(len(data_list['test'])))

    return data_list

def make_data_loader(batch_size, num_char, image_size, margin=5):
    data_list = make_data_list(num_char)

    train_dataset = LoadDataset(data_list['train'], num_char=num_char, image_size=image_size, margin=margin)
    valid_dataset = LoadDataset(data_list['val'], num_char=num_char, image_size=image_size, margin=margin)
    test_dataset = LoadDataset(data_list['test'], num_char=num_char, image_size=image_size, margin=margin)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True),
        'val': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    }

    return dataloader

def train(model, dataloader_dict, optimizer, num_epochs, log_dir, device, model_path):
    print('使用デバイス：', device)

    model = model.to(device)

    min_val_loss = 1e9
    writer = SummaryWriter(log_dir=log_dir)
    non_updated_cnt = 0

    PATIENT_TIME = 50
    W_REC = 1e0

    for epoch in range(num_epochs):
        print('')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_rec_loss = 0.0

            for img, gt, gt_label in dataloader_dict[phase]:
                img = img.to(device)
                gt = gt.to(device)
                gt_label = gt_label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y = model(img, gt_label)

                    # Compute loss
                    rec_loss = nn.L1Loss()(y, gt)

                    ## summarize loss
                    loss = W_REC*rec_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_rec_loss += rec_loss.item() * img.size(0)
                    epoch_loss += loss.item() * img.size(0)

            epoch_rec_loss = epoch_rec_loss / len(dataloader_dict[phase].dataset)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)

            writer.add_scalar('loss/' + phase + '/rec', epoch_rec_loss, epoch)
            writer.add_scalar('loss/' + phase + '/total', epoch_loss, epoch)

            print('Epoch {}/{}  {} Loss: {:.4f} (min val loss: {:.4f})'.format(epoch+1, num_epochs, phase, epoch_loss, min_val_loss))

            if phase == 'val' and epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                torch.save(model.state_dict(), model_path)
                print('model is saved')

            if phase == 'val':
                if epoch_loss > min_val_loss:
                    non_updated_cnt += 1
                    if non_updated_cnt >= PATIENT_TIME:
                        writer.close()
                        return
                else:
                    non_updated_cnt = 0
    writer.close()


if __name__=='__main__':
    freeze_seed(1234)

    LR = 1e-5
    EPOCHS = 2000
    NUM_CHAR = 26
    IMAGE_SIZE = 64
    BATCH_SIZE = 512
    MODEL_NAME = 'fannet' # 'fannet', 'fannet2'
    MODEL_PATH = f'./weight/style_encoder_{MODEL_NAME}_retrain.pth'
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # save先のディレクトリがなかったら作る
    if os.path.isdir('logs') == False:
        os.makedirs('logs')
    if os.path.isdir('weight') == False:
        os.makedirs('weight')


    num_log_dir = str(len(glob('logs/*')) + 1)
    log_dir = os.path.join('logs', 'log' + num_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    dataloader_dict = make_data_loader(BATCH_SIZE, NUM_CHAR, image_size=IMAGE_SIZE)

    if MODEL_NAME == 'fannet':
        model = FANnet(NUM_CHAR)
    elif MODEL_NAME == 'fannet2':
        model = StyleEncoder(NUM_CHAR)
    else:
        raise ValueError('MODEL_NAME must be fannet or fannet2')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, dataloader_dict, optimizer, EPOCHS, log_dir, DEVICE, MODEL_PATH)

    # send slack message
    requests.post(os.getenv('SLACK_URL'), data=json.dumps({'text': f":white_check_mark: log{num_log_dir} All finished !!!"}))
