from tqdm import tqdm

import pandas as pd

from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from utils import *
from unet import Unet
from fannet import FANnet


### global variables

# define timesteps
timesteps = 1000

# define betas
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # alpha_var
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # [1. , *alphas_cumpprod[:-1]]
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

### end global variables

class LoadDataset(data.Dataset):
    def __init__(self, data_list, num_class, image_size=64, margin=5):
        self.data_list = data_list
        self.num_class = num_class
        self.image_size = image_size
        self.margin = margin

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # light_img, light_feat, bold_img, bold_feat, gt_img, gt_fontname, label
        return self.data_list[index]

def read_img(path, transform, image_size=64, margin=5):
    img = cv2.imread(path, 0)
    img = preprocessing(img, img_size=image_size, margin=margin)
    img = transform(img).float().unsqueeze(0)

    return img

def make_data_list(encoder, num_class, category, device, image_size=64, margin=5):
    df = pd.read_csv('csv_files/weight_interpolation.csv')
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1)
                ])
    data_dic = {c: [] for c in category}
    for cate in data_dic.keys():
        cate_df = df[df['category'] == cate].reset_index(drop=True)

        # gather data within the same category
        _data = {'light_img': [], 'bold_img': [], 'gt_img': [], 'gt_fontname': [], 'label': [],}
        for _, row in cate_df.iterrows():
            for label in range(num_class):
                light_path = '../font2img/image/' + row['Light'] + f"/{chr(label + ord('A'))}.png"
                bold_path = '../font2img/image/' + row['Bold'] + f"/{chr(label + ord('A'))}.png"
                gt_path = '../font2img/image/' + row['Medium'] + f"/{chr(label + ord('A'))}.png"

                light_img = read_img(light_path, transform, image_size=image_size, margin=margin)
                bold_img = read_img(bold_path, transform, image_size=image_size, margin=margin)
                gt_img = read_img(gt_path, transform, image_size=image_size, margin=margin)

                gt_fontname = row['Medium']

                _data['light_img'].append(light_img)
                _data['bold_img'].append(bold_img)
                _data['gt_img'].append(gt_img)
                _data['gt_fontname'].append(gt_fontname)
                _data['label'].append(label)

        # encode
        _data['light_img'] = torch.cat(_data['light_img'])
        _data['bold_img'] = torch.cat(_data['bold_img'])
        _data['gt_img'] = torch.cat(_data['gt_img'])
        b = 512
        light_feats = []
        bold_feats = []
        with torch.no_grad():
            for i in range(0, _data['light_img'].size(0), b):
                # range of encoder input is 0-1
                light_feat = encoder.style_encode((_data['light_img'][i:i+b].to(device)+1)*0.5)
                bold_feat = encoder.style_encode((_data['bold_img'][i:i+b].to(device)+1)*0.5)
                light_feats.append(light_feat.cpu().detach().clone())
                bold_feats.append(bold_feat.cpu().detach().clone())

        light_feats = torch.cat(light_feats)
        bold_feats = torch.cat(bold_feats)

        for i in range(_data['light_img'].size(0)):
            if i % num_class == 0:
                # use average feature
                light_ave_feat = light_feats[i:i+num_class].mean(dim=0)
                bold_ave_feat = bold_feats[i:i+num_class].mean(dim=0)
            data_dic[cate].append((_data['light_img'][i], light_ave_feat,\
                                    _data['bold_img'][i], bold_ave_feat,\
                                    _data['gt_img'][i], _data['gt_fontname'][i], _data['label'][i]))

    for k, v in data_dic.items():
        print(f'{k} SIZE: {len(v)}')

    return data_dic

def make_data_loader(encoder, batch_size, image_size, num_class, category, device, margin=5):
    data_dic = make_data_list(encoder, num_class, category, device, image_size=image_size, margin=margin)

    dataset = {c: LoadDataset(data_dic[c], num_class, image_size=image_size, margin=margin) for c in category}
    dataloader = {c: data.DataLoader(dataset[c], batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True) for c in category}

    return dataloader

@torch.no_grad()
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def condition_blend_sampling_ddim(model, classes, style1, style2, class_scale=1., style_scale=1., alpha=0.5, image_size=64):
    b = classes.shape[0]
    device = classes.device
    x = torch.randn(b, 1, image_size, image_size).to(device)

    total_timesteps, sampling_timesteps, eta = timesteps, timesteps//10, 1.

    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    x_start = None

    # condition blend
    style = alpha * style1 + (1-alpha) * style2

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        t = torch.full((b,), time, device=device, dtype=torch.long)
        pred_noise = model.forward_with_cond_scale(x, t, classes, style,\
                                                class_scale=class_scale, style_scale=style_scale, rescaled_phi=0.)
        x_start  = extract(sqrt_recip_alphas_cumprod, t, x.shape) * x -\
                        extract(sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise

        if time_next < 0:
            x = x_start
            continue

        alpha = alphas_cumprod[time]
        alpha_next = alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(x)
        x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    return x

@torch.no_grad()
def noise_blend_sampling_ddim(model, classes, style1, style2, class_scale=1., style_scale=1., alpha=0.5, image_size=64):
    b = classes.shape[0]
    device = classes.device
    x = torch.randn(b, 1, image_size, image_size).to(device)

    total_timesteps, sampling_timesteps, eta = timesteps, timesteps//10, 1.

    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        t = torch.full((b,), time, device=device, dtype=torch.long)
        style1_noise = model.forward_with_cond_scale(x, t, classes, style1,\
                                                    class_scale=class_scale, style_scale=style_scale, rescaled_phi=0.)
        style2_noise = model.forward_with_cond_scale(x, t, classes, style2,\
                                                    class_scale=class_scale, style_scale=style_scale, rescaled_phi=0.)
        # noise blend
        pred_noise = alpha * style1_noise + (1-alpha) * style2_noise
        x_start  = extract(sqrt_recip_alphas_cumprod, t, x.shape) * x -\
                        extract(sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise

        if time_next < 0:
            x = x_start
            continue

        alpha = alphas_cumprod[time]
        alpha_next = alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(x)
        x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    return x

@torch.no_grad()
def image_blend(model, classes, style1, style2, image1, image2, class_scale=1., style1_scale=0., style2_scale=0., sampling_t=500):
    # image blend
    image3 = 2*(1 - torch.clamp(2 - (image1 + 1)*0.5 - (image2 + 1)*0.5, min=0., max=1.)) - 1

    b = classes.shape[0]
    device = classes.device
    t = torch.full((b,), sampling_t-1, device=device, dtype=torch.long)
    x = q_sample(image3, t)

    for i in tqdm(reversed(range(0, sampling_t)), desc='sampling loop time step', total=sampling_t):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        nocond_noise = model(x, t, classes, style1, class_drop_prob=1., style_drop_prob=1.)
        pred_noise = nocond_noise.clone()

        if class_scale > 0.:
            pred_noise += class_scale  * (model(x, t, classes, style1, class_drop_prob=0., style_drop_prob=1.) - nocond_noise)
        if style1_scale > 0.:
            pred_noise += style1_scale * (model(x, t, classes, style1, class_drop_prob=1., style_drop_prob=0.) - nocond_noise)
        if style2_scale > 0.:
            pred_noise += style2_scale * (model(x, t, classes, style2, class_drop_prob=1., style_drop_prob=0.) - nocond_noise)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        if i == 0:
            x = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            _noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * _noise

    return x


if __name__ == '__main__':
    # important experiment parameters
    ALPHA = 0.5        # blending rate
    APPROACH = 'Image' # must be 'Noise' or 'Condition' or 'Image'
    CLASS_SCALE = 1.   # Common to three approaches
    STYLE_SCALE = 1.   # Noise or Condition
    SAMPLING_T = 400   # Image
    STYLE1_SCALE = 0.  # Image
    STYLE2_SCALE = 0.  # Image

    # DM
    CHANNELS = 1
    UNET_DIM = 128
    NUM_CLASS = 26
    UNET_DIM_MULTS = (1, 2, 4, 8,)
    ENCODER_PATH = './weight/style_encoder_fannet.pth'
    MODEL_PATH = './weight/log33_fannet_step_320000.pth'

    # others
    SEED = 7777
    IMAGE_SIZE = 64
    BATCH_SIZE = 128
    SAVE_TXT_PATH = 'result/weight_interpolation/mse.txt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_IMG_DIR = f'result/weight_interpolation/{APPROACH}'
    CATEGORY = ['SERIF', 'SANS_SERIF', 'DISPLAY', 'HANDWRITING']

    freeze_seed(SEED)
    print(f'Using device: {DEVICE}')
    print(f'APPROACH: {APPROACH}')

    encoder = FANnet(NUM_CLASS)
    encoder.to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()

    model = Unet(
        dim=UNET_DIM,
        channels=CHANNELS,
        dim_mults=UNET_DIM_MULTS,
        num_class=NUM_CLASS,
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataloader = make_data_loader(encoder, BATCH_SIZE, IMAGE_SIZE, NUM_CLASS, CATEGORY, DEVICE)

    loss = {c: 0. for c in CATEGORY}
    for cate in CATEGORY:
        # make dir
        os.makedirs(os.path.join(SAVE_IMG_DIR, cate), exist_ok=True)

        for light_img, light_feat, bold_img, bold_feat, gt_img, gt_fontname, label in dataloader[cate]:
            light_img = light_img.to(DEVICE)
            light_feat = light_feat.to(DEVICE)
            bold_img = bold_img.to(DEVICE)
            bold_feat = bold_feat.to(DEVICE)
            gt_img = gt_img.to(DEVICE)
            label = label.to(DEVICE)

            # interpolation
            if APPROACH == 'Noise':
                gen = noise_blend_sampling_ddim(model, label, light_feat, bold_feat, \
                                                class_scale=CLASS_SCALE, style_scale=STYLE_SCALE, \
                                                alpha=ALPHA, image_size=IMAGE_SIZE)
            elif APPROACH == 'Condition':
                gen = condition_blend_sampling_ddim(model, label, light_feat, bold_feat, \
                                                    class_scale=CLASS_SCALE, style_scale=STYLE_SCALE, \
                                                    alpha=ALPHA, image_size=IMAGE_SIZE)
            elif APPROACH == 'Image':
                gen = image_blend(model, label, light_feat, bold_feat, light_img, bold_img, \
                                class_scale=CLASS_SCALE, style1_scale=STYLE1_SCALE, style2_scale=STYLE2_SCALE, \
                                sampling_t=SAMPLING_T)
            else:
                raise ValueError('APPROACH must be Noise or Condition or Image')

            # calc loss
            with torch.no_grad():
                loss[cate] += nn.MSELoss()(torch.clamp(gen, min=-1., max=1.), gt_img).item() * label.size(0)

            # save generated image
            save_filename = [n + '_' + chr(l + ord('A')) for n, l in zip(gt_fontname, label.cpu().detach().clone())]
            gen_from_0_to_1 = torch.clamp((gen.cpu().detach().clone()+1)*0.5, min=0., max=1.)
            save_generated_image(gen_from_0_to_1, os.path.join(SAVE_IMG_DIR, cate), save_filename)

        loss[cate] /= len(dataloader[cate].dataset)
        print(f'{cate} loss: {loss[cate]:.3f}')

    with open(SAVE_TXT_PATH, 'a') as f:
        f.write('\n========================\n')
        f.write(f'{APPROACH} weight interpolation\n')
        f.write(MODEL_PATH + '\n')
        if APPROACH == 'Image':
            f.write(f'sampling t: {SAMPLING_T}, class scale: {CLASS_SCALE}, '\
                    f'style1 scale: {STYLE_SCALE}, style2 scale: {STYLE2_SCALE}\n')
        else:
            f.write(f'class scale: {CLASS_SCALE}, style scale: {STYLE_SCALE}\n')
        for cate, loss in loss.items():
            f.write(f'{cate} MSE: {loss:.3f}\n')
