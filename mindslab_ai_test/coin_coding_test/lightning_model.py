from math import sqrt

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn

import dataloader


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1.):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # init w and b#
        # Initialize layers following SIREN paper
        w_std = (sqrt(6. / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)
        self.w0 = w0

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.w0 * x)
        return x


class COIN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        c = self.hparams.arch.channels
        self.mid = nn.ModuleList(
            [Siren(c, c) for i in range(self.hparams.arch.layers - 2)]
        )
        self.model = nn.Sequential(
            Siren(2, c, self.hparams.arch.init_scale),
            *self.mid,
            Siren(c, 3),
        )
        self.mse = nn.MSELoss()

        # Save Image for tblogger
        im = Image.open(self.hparams.data.image)
        im = (np.array(im, dtype=np.int32) - 128) / 128.  # normalize rgb [0,255]-> [-1,1]
        im = im[:, :, :3]  # remove alpha channel
        self.image = torch.tensor(im)

    def forward(self, x):
        # [B, 2] --> [B, 3]
        x = self.model(x)
        return x

    def common_step(self, x, rgb):
        output = self(x)
        loss = self.mse(output, rgb)
        return loss, output

    def training_step(self, batch, batch_nb):
        x, rgb = batch  # coordinate [B,2], rgb [B,3]
        loss, _ = self.common_step(x, rgb)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, rgb = batch
        loss, output = self.common_step(x, rgb)
        self.log('val_loss', loss)
        self.logger.log_image(output, self.image, self.current_epoch)
        return {'loss': loss, 'output': output}

    def test_step(self, batch, batch_nb):
        x, rgb = batch  # [B, H*W, 2], [B, H*W, 3]
        loss, output = self.common_step(x, rgb)
        self.log('test_loss', loss)
        with torch.no_grad():
            output = output.squeeze(0).view([self.image.shape[0], self.image.shape[1], 3])
            output = (128 * output + 128).detach().cpu().to(torch.int32).numpy()
            im = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode='RGB')
            im.save('./figure/recon.png', format='png')
        return {'test_loss': loss, 'output': output}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.train.lr,
                               eps=self.hparams.train.opt_eps,
                               betas=(self.hparams.train.beta1,
                                      self.hparams.train.beta2),
                               weight_decay=self.hparams.train.weight_decay)
        return opt

    def train_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 1)

    def test_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 2)
