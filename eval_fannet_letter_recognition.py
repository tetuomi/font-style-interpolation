import pandas as pd

from torch import nn
import torch.utils.data as data
import torchvision.models as models

from utils import *
from fannet import FANnet


class LoadDataset(data.Dataset):
    def __init__(self, data_list, num_class, image_size=64, margin=40):
        self.data_list = data_list
        self.num_class = num_class
        self.image_size = image_size
        self.margin = margin

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        style1_path = self.data_list[index]['style1']
        style2_path = self.data_list[index]['style2']
        style1_name = self.data_list[index]['style1_name']
        style2_name = self.data_list[index]['style2_name']
        label = self.data_list[index]['label']

        style1_img = self.read_img(style1_path)
        style2_img = self.read_img(style2_path)

        return style1_img, style2_img, label, style1_name, style2_name

    def read_img(self, path):
        img = cv2.imread(path, 0)
        img = preprocessing(img, img_size=self.image_size, margin=self.margin)
        img = torch.tensor(img).float().unsqueeze(0)

        return img

def make_data_list(num_class, category):
    df = pd.read_csv('csv_files/letter_recognition.csv')

    data_list = {c: [] for c in category}
    for cate in data_list.keys():
        cate_df = df[df['category'] == cate].reset_index(drop=True)
        for _, row in cate_df.iterrows():
            for label in range(num_class):
                style1_path = '../font2img/image/' + row['style1'] + f"/{chr(label + ord('A'))}.png"
                style2_path = '../font2img/image/' + row['style2'] + f"/{chr(label + ord('A'))}.png"

                data_list[cate].append({'style1': style1_path, 'style2': style2_path,\
                                        'style1_name': row['style1'], 'style2_name': row['style2'], 'label': label})

    for k, v in data_list.items():
        print(f'{k} SIZE: {len(v)}')

    return data_list

def make_data_loader(batch_size, image_size, num_class, category, margin=40):
    data_list = make_data_list(num_class, category)

    dataset = {c: LoadDataset(data_list[c], num_class, image_size=image_size, margin=margin) for c in category}
    dataloader = {c: data.DataLoader(dataset[c], batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True) for c in category}

    return dataloader

class LoadDatasetMyFonts(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # style1, style2, style1_name, style2_name, label
        return self.data_list[index]

def make_data_list_myfonts(num_class, image_size=64, margin=5):
    df = pd.read_csv('csv_files/letter_recognition_myfonts.csv')

    data_list = []
    for _, row in df.iterrows():
        style1_imgs = np.load(f"../font2img/myfonts/test/{row['style1']}")['arr_0']
        style2_imgs = np.load(f"../font2img/myfonts/test/{row['style2']}")['arr_0']
        for label in range(num_class):
            style1_img = style1_imgs[label]
            style2_img = style2_imgs[label]
            style1_img = torch.tensor(preprocessing_myfonts(style1_img, image_size, margin)).unsqueeze(0)
            style2_img = torch.tensor(preprocessing_myfonts(style2_img, image_size, margin)).unsqueeze(0)
            
            data_list.append((style1_img, style2_img, label, row['style1'].split('.')[0], row['style2'].split('.')[0]))
            
    print(f'SIZE: {len(data_list)}')

    return data_list

def make_data_loader_myfonts(batch_size, image_size, num_class, margin=5):
    data_list = make_data_list_myfonts(num_class, image_size=image_size, margin=margin)

    dataset = LoadDatasetMyFonts(data_list)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return {'MYFONTS_RANDOM': dataloader}


if __name__ == '__main__':
    ALPHA = 0.5
    SEED = 7777
    NUM_CLASS = 26
    IMAGE_SIZE = 64
    BATCH_SIZE = 256
    CLASSIFIER_PATH = './weight/char_classifier2.pth'
    SAVE_IMG_DIR = 'result/letter_recognition/FANnet'
    SAVE_TXT_PATH = 'result/letter_recognition/acc.txt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = './weight/style_encoder_fannet_retrain.pth'
    # CATEGORY = ['SERIF', 'SANS_SERIF', 'DISPLAY', 'HANDWRITING', 'RANDOM']
    CATEGORY = ['MYFONTS_RANDOM'] # for myfonts

    freeze_seed(SEED)
    print(f'Using device: {DEVICE}')
    print('FANnet')

    # dataloader = make_data_loader(BATCH_SIZE, IMAGE_SIZE, NUM_CLASS, CATEGORY)
    dataloader = make_data_loader_myfonts(BATCH_SIZE, IMAGE_SIZE, NUM_CLASS, margin=5) # for myfonts

    model = FANnet(NUM_CLASS)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    classifier = models.resnet18()
    classifier.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, NUM_CLASS)
    classifier.to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()

    correct        = {c: 0. for c in CATEGORY}
    style1_correct = {c: 0. for c in CATEGORY}
    style2_correct = {c: 0. for c in CATEGORY}
    for cate in CATEGORY:
        # make dir
        os.makedirs(os.path.join(SAVE_IMG_DIR, cate), exist_ok=True)

        for style1_img, style2_img, label, style1_name, style2_name in dataloader[cate]:
            with torch.no_grad():
                style1_img = style1_img.to(DEVICE)
                style2_img = style2_img.to(DEVICE)
                label_onehot = nn.functional.one_hot(label, num_classes=NUM_CLASS).float().to(DEVICE)

                # interpolation
                gen = model.interpolate(style1_img, style2_img, label_onehot, ALPHA)

                # recognize generated image
                out = classifier(gen)
                pred = torch.argmax(out, dim=1).cpu().detach().clone()
                correct[cate] += torch.sum(pred == label.data)

                # recognize style1 image and style2 image
                style1_out = classifier(style1_img)
                style1_pred = torch.argmax(style1_out, dim=1).cpu().detach().clone()
                style1_correct[cate] += torch.sum(style1_pred == label.data)

                style2_out = classifier(style2_img)
                style2_pred = torch.argmax(style2_out, dim=1).cpu().detach().clone()
                style2_correct[cate] += torch.sum(style2_pred == label.data)

                # # save generated image
                save_filename = [s1 + '_' + s2 + '_gt_' + chr(gt + ord('A')) + '_pred_' + chr(p + ord('A')) \
                                    for s1, s2, gt, p in zip(style1_name, style2_name, label, pred)]
                save_generated_image(gen.cpu().detach().clone(), os.path.join(SAVE_IMG_DIR, cate), save_filename)

        correct[cate] /= len(dataloader[cate].dataset)
        style1_correct[cate] /= len(dataloader[cate].dataset)
        style2_correct[cate] /= len(dataloader[cate].dataset)
        print(f'{cate} generated image acc: {correct[cate]:.3f}')
        print(f'{cate} style1 acc: {style1_correct[cate]:.3f}')
        print(f'{cate} style2 acc: {style2_correct[cate]:.3f}')

    with open(SAVE_TXT_PATH, 'a') as f:
        f.write('\n========================\n')
        f.write('FANnet letter_recognition\n')
        f.write(f'CLASSIFIER: {CLASSIFIER_PATH}\n')
        f.write(f'MODEL: {MODEL_PATH}\n')
        # write result of generated image
        f.write('generated image ACC\n')
        for cate, acc in correct.items():
            f.write(f'{cate} ACC: {acc:.3f}\n')
        # write result of style1 image
        f.write('style1 image ACC\n')
        for cate, acc in style1_correct.items():
            f.write(f'{cate} ACC: {acc:.3f}\n')
        # write result of style2 image
        f.write('style2 image ACC\n')
        for cate, acc in style2_correct.items():
            f.write(f'{cate} ACC: {acc:.3f}\n')
