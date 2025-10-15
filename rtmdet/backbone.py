import torch.nn as nn

from .blocks.convs import ConvModule
from .blocks.csp import CSPLayer
from .blocks.spp import SPPBottleneck


class CSPNeXt(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = [40, 40, 80, 160, 320, 640, 1280]
        self.depths = [4, 8, 8, 4]

        self.stem = nn.Sequential(
            ConvModule(3, self.channels[0], stride=2),
            ConvModule(self.channels[0], self.channels[1]),
            ConvModule(self.channels[1], self.channels[2]),
        )

        self.stage1 = nn.Sequential(
            ConvModule(self.channels[2], self.channels[3], stride=2),
            CSPLayer(
                self.channels[3],
                self.channels[3],
                num_blocks=self.depths[0],
                shortcut=True,
                use_attention=True,
            ),
        )

        self.stage2 = nn.Sequential(
            ConvModule(self.channels[3], self.channels[4], stride=2),
            CSPLayer(
                self.channels[4],
                self.channels[4],
                num_blocks=self.depths[1],
                shortcut=True,
                use_attention=True,
            ),
        )

        self.stage3 = nn.Sequential(
            ConvModule(self.channels[4], self.channels[5], stride=2),
            CSPLayer(
                self.channels[5],
                self.channels[5],
                num_blocks=self.depths[2],
                shortcut=True,
                use_attention=True,
            ),
        )

        self.stage4 = nn.Sequential(
            ConvModule(self.channels[5], self.channels[6], stride=2),
            SPPBottleneck(self.channels[6]),
            CSPLayer(
                self.channels[6],
                self.channels[6],
                num_blocks=self.depths[3],
                shortcut=False,
                use_attention=True,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x8 = self.stage2(x)
        # print(x8)
        # exit(0)
        x16 = self.stage3(x8)
        x32 = self.stage4(x16)
        return x8, x16, x32
