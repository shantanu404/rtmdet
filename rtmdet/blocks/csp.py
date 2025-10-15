import torch
import torch.nn as nn

from .channel_attention import ChannelAttention
from .convs import ConvModule, DWConvModule


class CSPNextBlock(nn.Module):
    def __init__(self, channels, skip_connection):
        super().__init__()

        self.skip_connection = skip_connection
        self.conv1 = ConvModule(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DWConvModule(
            channels, channels, kernel_size=5, stride=1, padding=2
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip_connection:
            out = out + residual

        return out


class CSPLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_blocks, shortcut, use_attention=False
    ):
        super().__init__()

        assert out_channels % 2 == 0, "out_channels should be even."
        out_channels_half = out_channels // 2

        self.main_conv = ConvModule(
            in_channels, out_channels_half, kernel_size=1, stride=1, padding=0
        )

        self.short_conv = ConvModule(
            in_channels, out_channels_half, kernel_size=1, stride=1, padding=0
        )

        self.final_conv = ConvModule(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.blocks = nn.Sequential(
            *[
                CSPNextBlock(out_channels_half, skip_connection=shortcut)
                for _ in range(num_blocks)
            ]
        )

        self.attention = (
            ChannelAttention(out_channels) if use_attention else nn.Identity()
        )

    def forward(self, x):
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x = torch.cat((x_main, x_short), dim=1)
        x = self.final_conv(self.attention(x))
        return x
