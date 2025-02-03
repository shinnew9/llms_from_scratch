# hyper_dataloader.py

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch

class HyperImageDataset(Dataset):
    def __init__(self, image_paths, num_images_per_sample=3, transform=None):
        self.image_paths = image_paths
        self.num_images_per_sample = num_images_per_sample
        self.transform = transform

        if len(self.image_paths) % self.num_images_per_sample != 0:
            # 패딩: 마지막 샘플이 부족할 경우 반복해서 채움
            padding = self.num_images_per_sample - (len(self.image_paths) % self.num_images_per_sample)
            self.image_paths += self.image_paths[:padding]

    def __len__(self):
        return len(self.image_paths) //self.num_images_per_sample
    
    def __getitem__(self, idx):
        start = idx*self.num_images_per_sample
        end = start + self.num_images_per_sample 
        images = []
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        # 이미지를 하나의 텐서로 합침
        images = torch.cat(images, dim=0)  # [C * N, H, W]
        return images

def get_hyper_dataloader(hparams):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = HyperImageDataset(
        image_paths=hparams.data.train_images,
        transform=transform
    )
    if 'val_images' in hparams.data:
        val_dataset = HyperImageDataset(
            image_paths = hparams.data.val_images,
            transform = transform
        )

        train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.train.batch_size,
        shuffle=True,
        num_workers=0
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=hparams.train.batch_size,
            shuffle=True,
            num_workers=0
        )
        return train_dataloader, val_dataloader
    
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = hparams.train.batch_size,
            shuffle = True,
            num_workers=0
        )
        return train_dataloader, None

