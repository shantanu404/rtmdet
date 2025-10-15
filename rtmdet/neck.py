import torch
import torch.nn as nn

from .blocks.convs import ConvModule
from .blocks.csp import CSPLayer


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.channels = [320, 640, 1280]
        self.depth = 4
        self.out_channels = out_channels

        self.num_levels = len(self.channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    self.channels[i],
                    self.channels[i - 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.top_down_blocks.append(
                CSPLayer(
                    in_channels=self.channels[i - 1] * 2,
                    out_channels=self.channels[i - 1],
                    num_blocks=self.depth,
                    shortcut=False,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.downsamples.append(
                ConvModule(
                    self.channels[i],
                    self.channels[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels=self.channels[i] * 2,
                    out_channels=self.channels[i + 1],
                    num_blocks=self.depth,
                    shortcut=False,
                )
            )

        self.out_convs = nn.ModuleList()

        for i in range(self.num_levels):
            self.out_convs.append(
                ConvModule(
                    self.channels[i],
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

    def forward(self, inputs):
        inner_outputs = [inputs[-1]]
        for i in range(self.num_levels - 1, 0, -1):
            high = inner_outputs[0]
            low = inputs[i - 1]

            idx = self.num_levels - 1 - i

            high_reduced = self.reduce_layers[idx](high)
            inner_outputs[0] = high_reduced

            high_upsampled = self.upsample(high_reduced)
            fused = torch.cat((high_upsampled, low), dim=1)
            inner_out = self.top_down_blocks[idx](fused)
            inner_outputs.insert(0, inner_out)

        outs = [inner_outputs[0]]

        for idx in range(self.num_levels - 1):
            low = outs[-1]
            high = inner_outputs[idx + 1]

            low_downsampled = self.downsamples[idx](low)
            fused = torch.cat((low_downsampled, high), dim=1)
            out = self.bottom_up_blocks[idx](fused)
            outs.append(out)

        for i in range(self.num_levels):
            outs[i] = self.out_convs[i](outs[i])

        return outs
