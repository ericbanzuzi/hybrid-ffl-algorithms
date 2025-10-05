import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision import models

NUM_CLASSES = {
    "cifar10": 10,
    "femnist": 62,
    "mnist": 10,
}

IN_CHANNELS = {
    "cifar10": 3,
    "femnist": 1,
    "mnist": 1,
}


class CNN(nn.Module):
    """
    Simple CNN model for FEMNIST based on https://arxiv.org/abs/2012.04221

    :in_channels: Number of input channels
    :num_classes: Number of output classes
    """

    def __init__(self, dataset: str = None):
        super().__init__()
        in_channels = IN_CHANNELS.get(dataset, 1)
        num_classes = NUM_CLASSES.get(dataset, 62)
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding="same")

        # Fully connected layers
        self.fc1 = (
            nn.Linear(8 * 8 * 32, 128)
            if dataset == "cifar10"
            else nn.Linear(7 * 7 * 32, 128)
        )
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout_conv = nn.Dropout(p=0.25)
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    """
    Simple CNN model based on https://arxiv.org/pdf/1602.05629

    :in_channels: Number of input channels
    :num_classes: Number of output classes
    """

    def __init__(self, dataset: str = None):
        super().__init__()
        in_channels = IN_CHANNELS.get(dataset, 3)
        num_classes = NUM_CLASSES.get(dataset, 10)

        self.conv1 = nn.Conv2d(in_channels, 32, 5, padding="same")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = (
            nn.Linear(64 * 8 * 8, 512)
            if dataset and "mnist" not in dataset
            else nn.Linear(64 * 7 * 7, 512)
        )
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNMnist(nn.Module):
    """
    Simple CNN model for FEMNIST based on https://arxiv.org/pdf/2501.03392

    :in_channels: Number of input channels
    :num_classes: Number of output classes
    """

    def __init__(self, dataset: str = None):
        super().__init__()
        in_channels = IN_CHANNELS.get(dataset, 1)
        num_classes = NUM_CLASSES.get(dataset, 62)

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    print("\nCNNCifar:")
    summary(CNNCifar(dataset="cifar10"), input_size=(1, 3, 32, 32))

    print("\nCNNMnist:")
    summary(CNNMnist(dataset="femnist"), input_size=(1, 1, 28, 28))

    print("\nCNN:")
    summary(CNN(dataset="femnist"), input_size=(1, 1, 28, 28))
