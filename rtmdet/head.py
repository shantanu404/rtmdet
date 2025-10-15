import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.convs import ConvModule
from .blocks.maskfeat import MaskFeatModule


class RTMDetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        strides,
        num_stacked_convs=2,
        num_dyconvs=3,
        num_prototypes=8,
        num_dyconv_channels=8,
        pred_kernel_size=1,
        channels=320,
    ):
        super().__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.num_stacked_convs = num_stacked_convs
        self.num_dyconvs = num_dyconvs
        self.num_prototypes = num_prototypes
        self.num_dyconv_channels = num_dyconv_channels
        self.pred_kernel_size = pred_kernel_size

        self.strides = strides
        self.num_levels = len(strides)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()

        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(self.num_dyconv_channels * (self.num_prototypes + 2))
                bias_nums.append(self.num_dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.num_dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.num_dyconv_channels * self.num_dyconv_channels)
                bias_nums.append(self.num_dyconv_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for _ in range(self.num_levels):
            cls_tower = nn.ModuleList()
            reg_tower = nn.ModuleList()
            kernel_tower = nn.ModuleList()

            cls_tower.append(
                ConvModule(
                    in_channels, self.channels, kernel_size=3, stride=1, padding=1
                )
            )
            reg_tower.append(
                ConvModule(
                    in_channels, self.channels, kernel_size=3, stride=1, padding=1
                )
            )
            kernel_tower.append(
                ConvModule(
                    in_channels, self.channels, kernel_size=3, stride=1, padding=1
                )
            )

            for _ in range(self.num_stacked_convs - 1):
                cls_tower.append(
                    ConvModule(
                        self.channels, self.channels, kernel_size=3, stride=1, padding=1
                    )
                )
                reg_tower.append(
                    ConvModule(
                        self.channels, self.channels, kernel_size=3, stride=1, padding=1
                    )
                )
                kernel_tower.append(
                    ConvModule(
                        self.channels, self.channels, kernel_size=3, stride=1, padding=1
                    )
                )

            self.cls_convs.append(cls_tower)
            self.reg_convs.append(reg_tower)
            self.kernel_convs.append(kernel_tower)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.channels,
                    num_classes,
                    kernel_size=self.pred_kernel_size,
                    stride=1,
                    padding=pred_pad_size,
                )
            )

            self.rtm_reg.append(
                nn.Conv2d(
                    self.channels,
                    4,
                    kernel_size=self.pred_kernel_size,
                    stride=1,
                    padding=pred_pad_size,
                )
            )

            self.rtm_kernel.append(
                nn.Conv2d(
                    self.channels,
                    self.num_gen_params,
                    kernel_size=self.pred_kernel_size,
                    stride=1,
                    padding=pred_pad_size,
                )
            )

        for i in range(self.num_levels):
            for j in range(self.num_stacked_convs):
                self.cls_convs[i][j].conv = self.cls_convs[0][j].conv
                self.reg_convs[i][j].conv = self.reg_convs[0][j].conv

        self.mask_head = MaskFeatModule(
            in_channels=in_channels,
            out_channels=self.channels,
            num_levels=self.num_levels,
            num_prototypes=self.num_prototypes,
            stacked_convs=4,
        )

    def forward(self, x):
        mask_feats = self.mask_head(x)

        level_preds = []

        for i in range(self.num_levels):
            cls_feat = x[i]
            reg_feat = x[i]
            kernel_feat = x[i]
            stride = self.strides[i]

            for cls_layer in self.cls_convs[i]:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs[i]:
                reg_feat = reg_layer(reg_feat)

            for kernel_layer in self.kernel_convs[i]:
                kernel_feat = kernel_layer(kernel_feat)

            cls_score = self.rtm_cls[i](cls_feat)
            bbox_pred = F.relu(self.rtm_reg[i](reg_feat)) * stride
            kernel_pred = self.rtm_kernel[i](kernel_feat)

            cls_score = cls_score.flatten(2).permute(0, 2, 1)
            bbox_pred = bbox_pred.flatten(2).permute(0, 2, 1)
            kernel_pred = kernel_pred.flatten(2).permute(0, 2, 1)
            
            level = torch.cat([cls_score, bbox_pred, kernel_pred], dim=2)
            level_preds.append(level)

        level_preds = torch.cat(level_preds, dim=1)

        return level_preds, mask_feats

    def parse_dynamic_params(self, flatten_kernels):
        n_insts = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)

        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels, self.weight_nums + self.bias_nums, dim=1
            )
        )

        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]

        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    n_insts * self.num_dyconv_channels, -1, 1, 1
                )
                bias_splits[i] = bias_splits[i].reshape(
                    n_insts * self.num_dyconv_channels
                )
            else:
                weight_splits[i] = weight_splits[i].reshape(n_insts, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_insts)

        return weight_splits, bias_splits