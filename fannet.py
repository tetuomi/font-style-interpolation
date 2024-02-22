import torch
import torch.nn as nn
import torch.nn.functional as F

class FANnet(nn.Module):
    def __init__(self, num_char):
        super().__init__()

        self.enc_img = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(16), # add
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(1), # add
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
        )
        self.enc_label = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_char, 512),
            nn.ReLU(inplace=True),
        )
        self.enc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x, char_label):
        feat_x = self.enc_img(x)
        feat_label = self.enc_label(char_label)
        
        feat = torch.cat((feat_x, feat_label), axis=1)
        feat = self.enc(feat)
        feat = feat.reshape(-1, 16, 8, 8)
        y = self.dec(feat)
        
        return y

    def style_encode(self, x):
        return F.normalize(self.enc_img(x))
    
    def interpolate(self, x1, x2, char_label, alpha):
        feat_x1 = self.enc_img(x1)
        feat_x2 = self.enc_img(x2)
        feat_x = alpha * feat_x1 + (1 - alpha) * feat_x2
        feat_label = self.enc_label(char_label)
        
        feat = torch.cat((feat_x, feat_label), axis=1)
        feat = self.enc(feat)
        feat = feat.reshape(-1, 16, 8, 8)
        y = self.dec(feat)
        
        return y
