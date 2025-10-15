import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.SyncBatchNorm(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DWConvModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=None
    ):
        super().__init__()

        self.depthwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels,
        )

        self.pointwise_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
