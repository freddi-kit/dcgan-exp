import torch
from torch import nn
import torch.nn.functional as F


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv0 = nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return torch.tanh(x)
