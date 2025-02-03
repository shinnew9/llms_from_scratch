import torch
import torch.nn as nn

class COINModel(nn.Module):
    def __init__(self):
        super(COINModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)  # Final layer to output the image
        return x

# Example usage
if __name__ == "__main__":
    model = COINModel()
    inputs = torch.randn(1, 3, 128, 128)  # Example input tensor
    outputs = model(inputs)  # Forward pass
    print("Input Shape:", inputs.shape)
    print("Output Shape:", outputs.shape)
