import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(
            in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0
        )
        self.act = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
