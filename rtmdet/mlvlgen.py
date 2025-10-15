import torch
from torch.nn.modules.utils import _pair


class MlvlPointGenerator:
    def __init__(self, strides, offset=0.0):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        return len(self.strides)

    @property
    def num_base_priors(self):
        return [1 for _ in self.strides]

    def grid_priors(self, featmap_size, device="cuda", with_stride=False):
        assert self.num_levels == len(featmap_size)
        mlvl_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_size[i], level_idx=i, device=device, with_stride=with_stride
            )
            mlvl_priors.append(priors)
        return mlvl_priors

    def single_level_grid_priors(
        self, featmap_size, level_idx, device="cuda", with_stride=False
    ):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(device)
            stride_h = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(device)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)

        all_points = shifts.to(device)
        return all_points

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        else:
            return yy.reshape(-1), xx.reshape(-1)
