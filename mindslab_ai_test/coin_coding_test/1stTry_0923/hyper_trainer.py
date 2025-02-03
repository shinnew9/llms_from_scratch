# hyper_trainer.py

import argparse
import datetime
from glob import glob
import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from coin_coding_test.hyper_model import HyperCOIN
from hyper_dataloader import get_hyper_dataloader
from tblogger import TensorBoardLoggerExpanded

def train(args):
    hparams = OmegaConf.load('hyper_hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.log.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    
    # 모델 초기화
    model = HyperCOIN(hparams)
    
    # 데이터 로더
    train_dataloader, val_dataloader = get_hyper_dataloader(hparams)
    
    # TensorBoard Logger
    tblogger = TensorBoardLoggerExpanded(hparams)
    
    # 체크포인트 콜백
    if val_dataloader is not None:
        monitor_metric = 'val_loss'
    else:
        monitor_metric = 'train_loss'

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.log.checkpoint_dir,
        filename=f'{hparams.log.name}_{{epoch}}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='train_loss',  # val_loss를 변경함
        mode='min',
        prefix=''
    )
    
    # 트레이너 설정
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.train.gpus,
        check_val_every_n_epoch=1 if val_dataloader is not None else None,
        max_epochs=100,
        logger=tblogger,
        progress_bar_refresh_rate=20,
        resume_from_checkpoint=None if args.resume_from is None else sorted(
            glob(os.path.join(hparams.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt'))
        )[-1],
    )
    
    # 학습 시작
    if val_dataloader is not None:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type=int, required=False, help="Resume Checkpoint epoch number")
    args = parser.parse_args()
    train(args)
