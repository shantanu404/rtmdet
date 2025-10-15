import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import ConvModule


class MaskFeatModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_levels, num_prototypes, stacked_convs
    ):
        super().__init__()

        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            convs.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(out_channels, num_prototypes, kernel_size=1)

    def forward(self, x):
        fusion_feats = [x[0]]
        size = x[0].shape[-2:]

        for i in range(1, self.num_levels):
            f = F.interpolate(x[i], size=size, mode="bilinear")
            fusion_feats.append(f)

        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)

        mask_feats = self.stacked_convs(fusion_feats)
        mask_feats = self.projection(mask_feats)

        return mask_feats
