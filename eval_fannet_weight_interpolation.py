import pandas as pd

from torch import nn
import torch.utils.data as data

from utils import *
from fannet import FANnet


class LoadDataset(data.Dataset):
    def __init__(self, data_list, num_class, image_size=64, margin=5):
        self.data_list = data_list
        self.num_class = num_class
        self.image_size = image_size
        self.margin = margin

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        light_path = self.data_list[index]['light_path']
        bold_path = self.data_list[index]['bold_path']
        gt_path = self.data_list[index]['gt_path']
        gt_fontname = self.data_list[index]['gt_fontname']
        label = self.data_list[index]['label']

        light_img = self.read_img(light_path)
        bold_img = self.read_img(bold_path)
        gt_img = self.read_img(gt_path)
        label_onehot = nn.functional.one_hot(torch.tensor(label), num_classes=self.num_class).float()

        return light_img, bold_img, label_onehot, gt_img, gt_fontname

    def read_img(self, path):
        img = cv2.imread(path, 0)
        img = preprocessing(img, img_size=self.image_size, margin=self.margin)
        img = torch.tensor(img).float().unsqueeze(0)

        return img

def make_data_list(num_class, category):
    df = pd.read_csv('csv_files/weight_interpolation.csv')

    data_list = {c: [] for c in category}
    for cate in data_list.keys():
        cate_df = df[df['category'] == cate].reset_index(drop=True)
        for _, row in cate_df.iterrows():
            for label in range(num_class):
                light_path = '../font2img/image/' + row['Light'] + f"/{chr(label + ord('A'))}.png"
                bold_path = '../font2img/image/' + row['Bold'] + f"/{chr(label + ord('A'))}.png"
                gt_path = '../font2img/image/' + row['Medium'] + f"/{chr(label + ord('A'))}.png"
                gt_fontname = row['Medium']

                data_list[cate].append({'light_path': light_path, 'bold_path': bold_path, 'gt_path': gt_path, 'gt_fontname': gt_fontname, 'label': label})

    for k, v in data_list.items():
        print(f'{k} SIZE: {len(v)}')

    return data_list

def make_data_loader(batch_size, image_size, num_class, category, margin=5):
    data_list = make_data_list(num_class, category)

    dataset = {c: LoadDataset(data_list[c], num_class, image_size=image_size, margin=margin) for c in category}
    dataloader = {c: data.DataLoader(dataset[c], batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True) for c in category}

    return dataloader


if __name__ == '__main__':
    ALPHA = 0.5
    SEED = 7777
    NUM_CLASS = 26
    IMAGE_SIZE = 64
    BATCH_SIZE = 16
    SAVE_IMG_DIR = 'result/weight_interpolation/FANnet'
    SAVE_TXT_PATH = 'result/weight_interpolation/mse.txt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = './weight/style_encoder_fannet_retrain.pth'
    CATEGORY = ['SERIF', 'SANS_SERIF', 'DISPLAY', 'HANDWRITING']

    freeze_seed(SEED)
    print(f'Using device: {DEVICE}')

    dataloader = make_data_loader(BATCH_SIZE, IMAGE_SIZE, NUM_CLASS, CATEGORY)

    model = FANnet(NUM_CLASS)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    loss = {c: 0. for c in CATEGORY}
    for cate in CATEGORY:
        # make dir
        os.makedirs(os.path.join(SAVE_IMG_DIR, cate), exist_ok=True)
        
        for light_img, bold_img, label, gt_img, gt_fontname in dataloader[cate]:
            with torch.no_grad():
                light_img = light_img.to(DEVICE)
                bold_img = bold_img.to(DEVICE)
                label = label.to(DEVICE)
                gt_img = gt_img.to(DEVICE)

                # interpolation
                gen = model.interpolate(light_img, bold_img, label, ALPHA)
                
                # calc loss
                loss[cate] += nn.MSELoss()(gen, gt_img).item() * light_img.size(0)
                
                # save generated image
                save_filename = [n + '_' + chr(l + ord('A')) for n, l in zip(gt_fontname, torch.argmax(label.cpu().detach(), dim=1))]
                save_generated_image(gen.cpu().detach().clone(), os.path.join(SAVE_IMG_DIR, cate), save_filename)

        loss[cate] /= len(dataloader[cate].dataset)
        print(f'{cate} loss: {loss[cate]:.3f}')

    with open(SAVE_TXT_PATH, 'a') as f:
        f.write('\n========================\n')
        f.write('FANnet weight interpolation\n')
        f.write(MODEL_PATH + '\n')
        for cate, loss in loss.items():
            f.write(f'{cate} MSE: {loss:.3f}\n')
