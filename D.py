import torch
from torch import nn
import torch.nn.functional as F


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 512, 4, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 4, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024 * 6 * 6, 1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = x.view(-1, 1024 * 6 * 6)
        return torch.sigmoid(self.fc(x))

