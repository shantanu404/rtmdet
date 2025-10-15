from abc import ABC, abstractmethod

import torch


class BaseBBoxCoder(ABC):
    encode_size = 4

    def __init__(self, use_box_type=False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        pass

    @abstractmethod
    def decode(self, bboxes, pred_bboxes):
        pass


class DistancePointBBoxCoder(BaseBBoxCoder):
    def __init__(self, clip_boarder=True, **kwargs):
        super().__init__(**kwargs)
        self.clip_boarder = clip_boarder

    def encode(self, bboxes, gt_bboxes):
        raise NotImplementedError

    def decode(self, points, pred_bboxes, max_shape=(640, 640)):
        x1 = points[:, 0] - pred_bboxes[:, 0]
        y1 = points[:, 1] - pred_bboxes[:, 1]
        x2 = points[:, 0] + pred_bboxes[:, 2]
        y2 = points[:, 1] + pred_bboxes[:, 3]

        bboxes = torch.stack([x1, y1, x2, y2], -1)

        if self.clip_boarder:
            bboxes[..., 0::2] = bboxes[..., 0::2].clamp(min=0, max=max_shape[1] - 1)
            bboxes[..., 1::2] = bboxes[..., 1::2].clamp(min=0, max=max_shape[0] - 1)

        return bboxes
