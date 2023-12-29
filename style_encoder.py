import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    def __init__(self, num_char):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_f = nn.Sequential(
            nn.Linear(256*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
        )
        self.fc_label = nn.Sequential(
            nn.Linear(num_char, 512),
        )
        self.upsample = nn.Sequential(
            nn.Linear(512*2, 256*7*7),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        feat = self.encoder(x)
        feat = feat.view(feat.shape[0], -1)
        feat_x = self.fc_f(feat)
        feat_label = self.fc_label(label)

        feat = torch.cat((feat_x, feat_label), 1)
        feat = self.upsample(feat).reshape((feat.shape[0], 256, 7, 7))
        y = self.decoder(feat)
        
        return y

    def style_encode(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.shape[0], -1)
        feat_x = self.fc_f(feat)

        return feat_x