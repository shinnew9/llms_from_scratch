import torch
import torch.nn as nn
import pytorch_lightning as pl

from hyper_dataloader import ImageDataset
from torch.utils.data import Dataset, DataLoader


class HyperCOIN(pl.LightningModule):
    def __init__(self, hparams):
        super(HyperCOIN, self).__init__()
        
        self.hidden_features = hparams.model.hidden_features
        self.out_features = hparams.model.out_features
        self.hidden_layers = hparams.model.hidden_layers
        
        # Define the MLP structure
        self.layers = [nn.Linear(2 + 1, self.hidden_features), nn.ReLU()]  # 2 input features + 1 index
        
        for _ in range(self.hidden_layers):
            self.layers += [nn.Linear(self.hidden_features, self.hidden_features), nn.ReLU()]
        
        self.layers.append(nn.Linear(self.hidden_features, self.out_features))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, img_idx):
        img_idx = img_idx.view(-1, 1)  # Reshape index
        x = torch.cat([x, img_idx], dim=-1)  # Concatenate index with input
        return self.model(x)


    def training_step(self, batch, batch_idx):
        # print(batch)  # Print the structure of the batch to see what's inside
        inputs, y =batch
        x, img_idx = inputs[:, :-1], inputs[:, -1]  # Split inputs into x and img_idx based on feature size
        
        y_hat = self.forward(x, img_idx)  # Forward pass
        loss = nn.CrossEntropyLoss()(y_hat, y)  # Calculate loss
        self.log('train_loss', loss)  # Log training loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)  # Example optimizer

    def train_dataloader(self):
        # Assuming you have a dataset defined
        train_dataset = ImageDataset()  # Replace with your dataset
        return DataLoader(train_dataset, batch_size=32)  # Adjust batch size as needed

