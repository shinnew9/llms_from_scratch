# hyper_test.py

import argparse
import torch
from omegaconf import OmegaConf
from coin_coding_test.hyper_model import HyperCOIN
from hyper_dataloader import get_hyper_dataloader
from pytorch_lightning import Trainer

def test(hparams):
    # 데이터 로더 로드
    dataloader = get_hyper_dataloader(hparams)
    
    # 모델 로드
    model = HyperCOIN.load_from_checkpoint(hparams.log.checkpoint_dir + '/best.ckpt', hparams=hparams)
    
    # 트레이너 설정
    trainer = Trainer(
        gpus=hparams.train.gpus,  # if torch.cuda.is_available() else 0
        logger=False
    )
    
    # 테스트 실행
    trainer.test(model, dataloaders=dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, default='hyper_hparameter.yaml', help='Path to hyperparameter file')
    args = parser.parse_args()
    
    # 하이퍼파라미터 로드
    hparams = OmegaConf.load(args.hparams)
    
    test(hparams)
