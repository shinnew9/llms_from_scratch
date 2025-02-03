from os import path
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

def create_hypercoin_dataloader(hparams, cv):
    def collate_fn(batch):
        x_list = list()
        rgb_list = list()
        for x, rgb in batch:
            x_list.append(x)
            rgb_list.append(rgb)
        x_list = torch.stack(x_list, dim=0)
        rgb_list = torch.stack(rgb_list, dim=0)

        return x_list, rgb_list

    DS = ImageDataset(hparams, cv)
    return DataLoader(dataset=DS,
                      batch_size=1,
                      shuffle=True if cv == 0 else False,
                      pin_memory=True,
                      drop_last=True if cv == 0 else False)

class ImageDataset(Dataset):
    def __init__(self, image_data, labels):
        self.data = image_data
        self.labels = labels


        # # 이미지 파일 경로 설정
        # image_folder = hparams.data.image_dir  # 예: 'C:/Users/YoojinShin/vscodeprojects/mindslab_ai_test/coin_coding_test/figure'
        # image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])

        # # 이미지 데이터 불러오기
        # for img_file in image_files:
        #     assert path.isfile(img_file), f"File not found: {img_file}"
        #     im = Image.open(img_file)
        #     im = (np.array(im, dtype=np.float32) - 128) / 128.  # [0, 255] -> [-1, 1]
        #     im = im[:, :, :3]  # 알파 채널 제거
        #     self.images.append(torch.tensor(im, dtype=torch.float32))

        # # 첫 번째 이미지로 해상도 가져오기
        # self.H, self.W, _ = self.images[0].shape

        # # 좌표 설정
        # h = torch.arange(self.H) / (self.H - 1.) * 2 - 1.  # [-1, 1]
        # w = torch.arange(self.W) / (self.W - 1.) * 2 - 1.  # [-1, 1]
        # self.x = torch.stack(torch.meshgrid(h, w), dim=-1).view(-1, 2)

    def __len__(self):
        return len(self.data)  # Return the total number of samples

    def __getitem__(self, idx):
        # Fetch the image and label at index 'idx'
        image = self.data[idx]
        label = self.labels[idx]

        return image, label