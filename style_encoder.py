import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.zdim = zdim
        hidden_units = 512

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
        self.fc_c = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    # nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim)),
                    # nn.ReLU(inplace=True)
                    )
        self.fc_f = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    # nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim)),
                    # nn.ReLU(inplace=True)
                    )
        self.fc_label = nn.Sequential(
            nn.Linear(26, int(zdim)),
            nn.ReLU(inplace=True),
        )
        self.classifier_c = nn.Sequential(
                            nn.Linear(int(zdim), 26),
#                             nn.Softmax(dim=1)
        )
        self.upsample = nn.Sequential(
            nn.Linear(zdim*2, 256*7*7),
            nn.ReLU(inplace=True))
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

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.shape[0],-1)

        z_f = self.fc_f(z)
        z_f = F.normalize(z_f)

        return z_f
