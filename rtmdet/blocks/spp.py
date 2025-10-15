import torch
import torch.nn as nn

from .convs import ConvModule


class SPPBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        c_half = channels // 2

        self.conv1 = ConvModule(channels, c_half, kernel_size=1, stride=1, padding=0)

        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in [5, 9, 13]
            ]
        )

        conv2_channels = c_half * (len(self.poolings) + 1)

        self.conv2 = ConvModule(
            conv2_channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.poolings[0](x)
        x2 = self.poolings[1](x1)
        x3 = self.poolings[2](x2)

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.conv2(out)
        return out
