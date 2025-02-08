from torch import nn
import torch.nn.functional as F

# 1D CNN model adjusted for 10800 inputs -> 80200 outputs


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Final conv layer to collapse channel dimension; resulting tensor will be [B,1,L]
        self.conv_output = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x expected shape: [1, batch_size, 10800]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)  # after four poolings, length becomes 10800/16=675

        # Upsample from length 675 to 80200
        x = F.interpolate(x, size=80200, mode='linear', align_corners=False)
        # Collapse channel dimension with a convolution
        x = self.conv_output(x)  # now shape: [batch, 1, 80200]
        # Flatten to [batch, 80200]
        x = F.leaky_relu(x)
        x = x.view(x.size(0), -1)
        return x
