# hyper_model.py - COIN 모델 for Multiple Targets

import torch
import torch.nn as nn
import pytorch_lightning as pl
import piq  # Perceptual Image Quality library
# from torchvision import models


class SineActivation(nn.Module):
    def __init__(self, omega=30):
        super(SineActivation, self).__init__()
        self.omega = omega
    
    def forward(self, x):
        return torch.sin(self.omega*x)
    

class HyperCOIN(pl.LightningModule):
    def __init__(self, hparams):
        super(HyperCOIN, self).__init__()
        self.save_hyperparameters(hparams)

        # 공유 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(3*len(hparams.data.train_images), hparams.arch.channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hparams.arch.channels, hparams.arch.channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # 추가 인코더 레이어
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hparams.arch.channels, hparams.arch.channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hparams.arch.channels, 3*len(hparams.data.train_images), kernel_size=3, padding=1),
            nn.Sigmoid(),
            # 추가 디코더 레이어
        )

        self.criterion = nn.MSELoss()
        
        # 모델 아키텍처 정의 (예시)
        # layers = []
        # input_channels = 3
        # for _ in range(hparams.arch.layers):
        #     layers.append(nn.Conv2d(input_channels, hparams.arch.channels, kernel_size=3, padding=1))
        #     layers.append(nn.ReLU())
        #     input_channels = hparams.arch.channels
        # layers.append(nn.Conv2d(input_channels, 3, kernel_size=3, padding=1))
        # self.model = nn.Sequential(*layers)
   
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train.lr,
            weight_decay=self.hparams.train.weight_decay,
            eps=self.hparams.train.opt_eps,
            betas=(self.hparams.train.beta1, self.hparams.train.beta2)
        )
        return optimizer