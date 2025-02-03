import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))  # Sort to ensure a consistent order
        self.image_files = [f for f in self.image_files if f.endswith('.png')]  # Filter for PNG images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert the image to RGB format

        if self.transform:
            image = self.transform(image)

        return image, idx  # Return the image and its index (idx can act as the image index input for the model)

def get_test_loader(batch_size=1, image_size=128):
    image_dir = './figure'  # Path to your image folder

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to the desired size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
    ])

    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
