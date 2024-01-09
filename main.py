import os
import json
import math
import argparse
import requests
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils import *
from unet import Unet
from data_loader import make_data_loader

### global variables
timesteps = 1000

# define betas
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # alpha_var
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # [1. , *alphas_cumpprod[:-1]]
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
# 順方向の拡散過程
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # どのくらい元画像を残すか，徐々に小さくなる
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # どのくらいノイズを加えるか，徐々に大きくなる

# calculations for posterior q(x_{t-1} | x_t, x_0)
# 逆方向の拡散過程
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # sigma_t^2 どのくらいノイズを加えるか，徐々に小さくなる．非常に小さい
### end global variables

# forward diffusion (using the nice property)
@autocast(enabled = False)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, classes, style, t, t_index, class_scale=3., style_scale=3., rescaled_phi=0.):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    pred_noise = model.forward_with_cond_scale(x, t, classes, style, class_scale=class_scale, style_scale=style_scale, rescaled_phi=rescaled_phi)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, classes, style, shape, class_scale=3., style_scale=3., rescaled_phi=0.):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, classes, style, t, i, class_scale=class_scale, style_scale=style_scale, rescaled_phi=rescaled_phi)
    return img

@torch.no_grad()
def sample(model, classes, style, image_size, batch_size=16, channels=3, class_scale=1., style_scale=3., rescaled_phi=0.):
    return p_sample_loop(model, classes, style, shape=(batch_size, channels, image_size, image_size), class_scale=class_scale, style_scale=style_scale, rescaled_phi=rescaled_phi)

def p_losses(denoise_model, x_start, t, char_label, style, noise=None, loss_type='l1'):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # ordinal diffusion loss
    predicted_noise = denoise_model(x_noisy, t, char_label, style)

    if loss_type == 'l1':
        criterion = torch.nn.L1Loss()
    elif loss_type == 'l2':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'huber':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise NotImplementedError()

    diff_loss = criterion(noise, predicted_noise)

    return diff_loss

def train(model, dataloader, optimizer, params):
    step = 0
    writer = SummaryWriter(log_dir=f"logs/log{params['experiment_id']}")

    while step < params['total_steps']:
        for batch, style, label in dataloader:
            optimizer.zero_grad()

            batch = batch.to(params['device'])
            style = style.to(params['device'])

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch.size(0),), device=params['device']).long()

            diff_loss = p_losses(model, batch, t, label, style, loss_type='huber')
            loss = params['W_DIFF']*diff_loss

            if step % 100 == 0:
                print('step', step, 'Loss:', loss.item(), 'diff:', diff_loss.item())
                writer.add_scalar('loss/total', loss.item(), step)
                writer.add_scalar('loss/diffusion', diff_loss.item(), step)

            loss.backward()
            optimizer.step()

            if step != 0 and step % (params['total_steps']//10) == 0:
                torch.save(model.module.state_dict(), params['model_filename'] + f"_step_{step}.pth")
                print('saved model')

                with torch.no_grad():
                    model.eval()
                    _classes = torch.tensor([dataloader.dataset[i][2] for i in range(100)], device=params['device'])
                    _style = torch.cat([dataloader.dataset[i][1].unsqueeze(0) for i in range(100)]).to(params['device'])
                    images = sample(model.module, _classes, _style, params['image_size'], batch_size=_style.size(0), channels=params['channels'],
                                    class_scale=1., style_scale=1., rescaled_phi=0.)
                    images = (images + 1) * 0.5
                    save_image(images.cpu(), f"result/log{params['experiment_id']}_step_{step}.png", nrow=10)
                    model.train()
            step += 1

    torch.save(model.module.state_dict(), params['model_filename'] + "_step_final.pth")
    print('saved model')
    writer.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--device_ids', nargs='*', help='device_ids e.x. (-device 0 1 2)', type=int, default=[0])
    parser.add_argument('-encoder_name', '--encoder_name', help='encoder_name e.x. (-encoder_name fannet)', type=str, default='fannet')
    args = parser.parse_args()

    params = {
        # trainning
        'lr' : 1e-4,
        'batch_size' : 256, # 256
        'total_steps' : 3e5, 

        # model
        'channels' : 1,
        'unet_dim' : 128,
        'image_size' : 64,
        'num_class' : 26,
        'cond_drop_prob': 0.1,
        'device_ids' : args.device_ids,
        'unet_dim_mults' : (1, 2, 4, 8,),

        # weights
        'W_DIFF': 1.,

        # others
        'seed' : 7777,
        'da_rate': 0.3,
        'encoder_name' : args.encoder_name,
        'dataset_name' : 'myfonts', # 'myfonts' or 'google_fonts'
        'experiment_id' : str(len(glob('logs/*')) + 1),
        'device' : f'cuda:{args.device_ids[0]}' if torch.cuda.is_available() else 'cpu',
    }

    # save先のディレクトリがなかったら作る
    if os.path.isdir('logs') == False:
        os.makedirs('logs')
    if os.path.isdir('result') == False:
        os.makedirs('result')
    if os.path.isdir('weight') == False:
        os.makedirs('weight')

    os.makedirs(f"logs/log{params['experiment_id']}")
    encoder_name = params['style_encoder_path'].split('_')[-1].split('.')[0]
    params['model_filename'] = f"./weight/log{params['experiment_id']}_{encoder_name}"

    freeze_seed(params['seed'])

    # load model
    model = Unet(
        dim=params['unet_dim'],
        channels=params['channels'],
        dim_mults=params['unet_dim_mults'],
        num_class=params['num_class'],
        cond_drop_prob=params['cond_drop_prob'],
    )
    model = torch.nn.DataParallel(model, device_ids=params['device_ids'])
    model.to(params['device'])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    dataloader = make_data_loader(params['batch_size'], params['image_size'], params['num_class'],
                                    params['style_encoder_path'], params['device'], params['dataset_name'], params['da_rate'])

    # train the model
    train(model, dataloader['train'], optimizer, params)

    # send slack message
    requests.post(os.getenv('SLACK_URL'), data=json.dumps({'text': f":white_check_mark: log{params['experiment_id']} All finished !!!"}))
