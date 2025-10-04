import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision import models


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
    

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False, dataset: str = None):
        super().__init__()
        self.net = models.resnet18(pretrained=pretrained)
        in_channels = 3 if dataset and 'mnist' not in dataset else 1
        # adapt first conv for CIFAR (32x32) as described in https://arxiv.org/pdf/1512.03385
        # original conv: 7x7 stride 2 -> often changed to 3x3 stride 1 for CIFAR
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()  # remove 3x3 stride 2 pooling
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = ResNet18(num_classes=10, dataset='cifar10')
    summary(model, input_size=(1, 3, 32, 32))
    
    