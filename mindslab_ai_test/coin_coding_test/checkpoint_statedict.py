import torch
from glob import glob
import argparse
from omegaconf import OmegaConf as OC
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,
            required = False, help = "Resume Checkpoint epoch number")
    args = parser.parse_args()
    hparam = OC.load('hyper_hparameter.yaml')  # hparameter.yaml
    print(sorted(glob(os.path.join(hparam.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt')))[-1])
    ckpt = torch.load(sorted(glob(os.path.join(hparam.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt')))[-1], map_location = torch.device('cpu'))
    param = ckpt['state_dict']
    torch.save(param, os.path.join(hparam.log.checkpoint_dir, f'state_dict={args.resume_from}.ckpt'), _use_new_zipfile_serialization=False)
