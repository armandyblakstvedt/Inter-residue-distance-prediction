from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels):
        super(Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(16)

        self.res1 = nn.Conv2d(in_channels=input_channels, out_channels=512, kernel_size=1, stride=1)
        self.res2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        self.res3 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, stride=1)

        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        # First block: conv1 -> conv2 + residual
        shortcut1 = self.res1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + shortcut1
        out = self.dropout(out)

        # Second block: conv3 -> conv4 + residual
        shortcut2 = self.res2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out + shortcut2
        out = self.dropout(out)

        # Third block: conv5 -> conv6 -> conv7 + residual
        shortcut3 = self.res3(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = out + shortcut3

        out = F.relu(self.conv8(out))

        out = (out + out.transpose(2, 3)) / 2

        return out
