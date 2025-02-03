# import argparse
# import datetime
# import os
# from glob import glob

# from omegaconf import OmegaConf as OC
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint

# from hyper_model import HyperCOIN
# from tblogger import TensorBoardLoggerExpanded
# from hyper_dataloader import create_hypercoin_dataloader

import torch
from pytorch_lightning import Trainer
from lightning_model import COIN
from omegaconf import OmegaConf as OC
from dataloader import MultiImageDataset
from torch.utils.data import DataLoader


# def train(args):
#     # Hyperparameters 로드
#     hparams = OC.load('hyper_hparameter.yaml')
#     now = datetime.datetime.now().strftime('%m_%d_%H')
#     hparams.name = f"{hparams.log.name}_{now}"
    
#     # 로그 및 체크포인트 디렉토리 생성
#     os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
#     os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    
#     # 모델과 데이터로더 준비
#     model = HyperCOIN(hparams)
#     train_loader = create_hypercoin_dataloader(hparams, cv=0)  # cv=0 for train, multiple images

#     # Tensorboard 로그 설정
#     tblogger = TensorBoardLoggerExpanded(hparams)
    
#     # 체크포인트 저장 설정
#     filename = f'{hparams.log.name}_{now}_{{epoch}}'
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=hparams.log.checkpoint_dir,
#         filename=filename,
#         verbose=True,
#         save_last=True,
#         save_top_k=3,
#         monitor='val_loss',
#         mode='min'
#     )
    
#     # Trainer 설정
#     trainer = Trainer(
#         checkpoint_callback=checkpoint_callback,
#         gpus=hparams.train.gpus,
#         check_val_every_n_epoch=100,
#         max_epochs=hparams.train.max_epochs,
#         logger=tblogger,
#         progress_bar_refresh_rate=4,
#         resume_from_checkpoint=None if args.resume_from is None else sorted(
#             glob(os.path.join(hparams.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt')))[-1],
#     )
    
#     # 모델 학습
#     trainer.fit(model, train_loader)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-r', '--resume_from', type=int, required=False, help="Resume Checkpoint epoch number")
#     args = parser.parse_args()
#     train(args)
def train():
    hparams = OC.load('hyper_hparameter.yaml')
    model = COIN(hparams)
    
    dataset = MultiImageDataset(hparams)
    dataloader = DataLoader(dataset, batch_size=hparams.train.batch_size, shuffle=True)
    
    trainer = Trainer(gpus=hparams.train.gpus, max_epochs=100)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    train()