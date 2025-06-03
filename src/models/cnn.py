import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class CNN(nn.Module):
    """
    Simple CNN model based on https://arxiv.org/pdf/1602.05629

    :in_channels: Number of input channels
    :num_classes: Number of output classes
    """
    def __init__(self, num_classes: int = 10, dataset: str = None):
        super().__init__()
        in_channels = 3 if dataset and 'mnist' not in dataset else 1
        self.conv1 = nn.Conv2d(in_channels, 32, 5, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512) if dataset and 'mnist' not in dataset else nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == '__main__':
    model = CNN(num_classes=10, dataset='femnist')
    summary(model, input_size=(1, 1, 28, 28))
    