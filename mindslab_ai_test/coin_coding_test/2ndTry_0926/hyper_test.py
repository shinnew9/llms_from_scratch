# import torch
# from PIL import Image
# import numpy as np
# from hyper_model import HyperCOIN

# def load_image(image_path):
#     """이미지 파일을 열고, 높이와 너비를 반환"""
#     image = Image.open(image_path)
#     image = np.array(image) / 255.0  # 이미지 정규화 [0, 1]
#     image_height, image_width = image.shape[:2]  # 이미지의 높이와 너비 추출
#     return image, image_height, image_width

# def create_image_coords(image_height, image_width):
#     """이미지 크기에 맞는 좌표 생성"""
#     h = torch.linspace(-1, 1, steps=image_height)  # [-1, 1] 사이의 y 좌표
#     w = torch.linspace(-1, 1, steps=image_width)  # [-1, 1] 사이의 x 좌표
#     coords = torch.stack(torch.meshgrid(h, w), dim=-1).view(-1, 2)  # 좌표 생성
#     return coords

# def reconstruct_image(model, image_coords, img_idx, device='cpu'):
#     model.to(device)
#     image_coords = image_coords.to(device)
#     img_idx = torch.tensor([img_idx], device=device).repeat(len(image_coords), 1)
    
#     with torch.no_grad():
#         output = model(image_coords, img_idx)
    
#     output = output.cpu().numpy().reshape((image_height, image_width, 3))  # RGB 형식으로 재구성
#     return (output * 255).astype(np.uint8)  # [0, 255]로 변환

# def save_image(reconstructed_image, filename):
#     """재구성된 이미지를 저장"""
#     Image.fromarray(reconstructed_image).save(filename)

# if __name__ == '__main__':
#     # 예시 이미지 경로
#     image_path = 'path_to_image.png'  # 이미지 파일 경로를 입력하세요
    
#     # 이미지 로드 및 크기 추출
#     image, image_height, image_width = load_image(image_path)
    
#     # 이미지 크기에 맞는 좌표 생성
#     image_coords = create_image_coords(image_height, image_width)
    
#     # 모델 로드 및 재구성
#     model = HyperCOIN()  # 학습된 모델을 불러오세요
#     reconstructed_image = reconstruct_image(model, image_coords, img_idx=0)
    
#     # 재구성된 이미지 저장
#     save_image(reconstructed_image, 'reconstructed_img.png')


# import torch
# from lightning_model import COIN
# from omegaconf import OmegaConf as OC

# def test():
#     hparams = OC.load('hyper_hparameter.yaml')
#     model = COIN(hparams)
#     model.eval()
    
#     for idx in range(len(hparams.data.images)):
#         x, rgb, img_idx = dataloader[idx]  # Assuming dataloader is properly implemented
#         output = model(x, img_idx)
        
#         # Save reconstructed image
#         output = output.view([rgb.shape[0], rgb.shape[1], 3])
#         output = (128 * output + 128).detach().cpu().to(torch.int32).numpy()
#         im = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode='RGB')
#         im.save(f'./figure/recon_{idx}.png', format='png')

# if __name__ == '__main__':
#     test()


# import torch
# from pytorch_lightning import Trainer
# from lightning_model import COIN
# from omegaconf import OmegaConf as OC
# from dataloader import MultiImageDataset
# from torch.utils.data import DataLoader

# def train():
#     hparams = OC.load('hyper_hparameter.yaml')
#     model = COIN(hparams)
    
#     dataset = MultiImageDataset(hparams)
#     dataloader = DataLoader(dataset, batch_size=hparams.train.batch_size, shuffle=True)
    
#     trainer = Trainer(gpus=hparams.train.gpus, max_epochs=100)
#     trainer.fit(model, dataloader)

# if __name__ == "__main__":
#     train()

import torch
from PIL import Image
from lightning_model import COIN
from omegaconf import OmegaConf as OC
from dataloader import MultiImageDataset
import numpy as np

def test():
    hparams = OC.load('hyper_hparameter.yaml')
    model = COIN.load_from_checkpoint('checkpoint_path.ckpt')  # Add correct path to checkpoint
    model.eval()

    dataset = MultiImageDataset(hparams)
    
    for idx in range(len(dataset)):
        x, rgb, img_idx = dataset[idx]  # Load test data
        x = x.unsqueeze(0)  # Add batch dimension if necessary
        
        with torch.no_grad():
            output = model(x, img_idx.unsqueeze(0))  # Forward pass
        
        # Convert output back to an image
        output = output.view([dataset.H, dataset.W, 3]).cpu().numpy()
        output = (output * 128 + 128).astype(np.uint8)
        
        # Save reconstructed image
        im = Image.fromarray(np.clip(output, 0, 255), mode='RGB')
        im.save(f'./figure/recon_{idx}.png', format='png')

if __name__ == "__main__":
    test()
