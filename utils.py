import math
import random
from inspect import isfunction

import cv2
import numpy as np

import torch


def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def preprocessing(img,img_size=64,margin=0,threshold_w=0,threshold_h=0):
    img_white = cv2.bitwise_not(img)#np.where(img_org>0,0,255) #文字領域白
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0
    # 行の処理
    for i in range(img_white.shape[0]):
        if np.sum(img_white[i,:]) > 0:
            y_min = i
            break
    for i in reversed(range(img_white.shape[0])):
        if np.sum(img_white[i,:]) > 0:
            y_max = i+1 #rangeが0~n-1なので，arrayのインデックス調整で+1する　例：img[0:N] これも0~N-1になっているので+1しておかないとバグる
            break
    # 列の処理
    for i in range(img_white.shape[1]):
        if np.sum(img_white[:,i]) > 0:
            x_min = i
            break
    for i in reversed(range(img_white.shape[1])):
        if np.sum(img_white[:,i]) > 0:
            x_max = i+1
            break
    img = img_white[y_min:y_max,x_min:x_max]
    h = img.shape[0]
    w = img.shape[1]
    if (h<threshold_h) or (w<threshold_w):
        return 0
    if margin>0:
        img = np.pad(img,[(margin,margin),(margin,margin)],'constant')
    size = max(w,h)
    ratio = img_size/size #何倍すれば良いか
    img_resize = cv2.resize(img, (int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_CUBIC)
    # img_resize = cv2.bitwise_not(img_resize) #文字領域黒
    #0埋めの幅を決める
    if w > h:
        pad = int((img_size - h*ratio)/2)
        #np.pad()の第二引数[(上，下),(左，右)]にpaddingする行・列数
        img_resize = np.pad(img_resize,[(pad,pad),(0,0)],'constant')
    elif h > w:
        pad = int((img_size - w*ratio)/2)
        img_resize = np.pad(img_resize,[(0,0),(pad,pad)],'constant')
    #最終的にきれいに100x100にresize
    img_resize = cv2.resize(img_resize,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    img_resize = cv2.bitwise_not(img_resize)#np.where(img_resize!=0,0,255)

    return img_resize / 255.0

def preprocessing_myfonts(img,img_size=64,margin=0):
    img = np.pad(img,[(margin,margin),(margin,margin)], 'constant', constant_values=255.)
    img = cv2.resize(img, (img_size, img_size))
    return img / 255.

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# 各時刻に対応する値を返す
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

