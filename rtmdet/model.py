import torch
import torch.nn as nn

from .backbone import CSPNeXt
from .head import RTMDetHead
from .neck import CSPNeXtPAFPN


class RTMDet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN(out_channels=320)
        self.bbox_head = RTMDetHead(
            in_channels=320,
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        cls_scores, bbox_preds, kernel_preds, mask_feats = self.bbox_head(features)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feats

    def predict(self, x, img_metas):
        self.eval()
        with torch.no_grad():
            cls_scores, bbox_preds, kernel_preds, mask_feats = self.forward(x)
            results = []
            results = self.bbox_head.predict_by_feat(
                cls_scores,
                bbox_preds,
                kernel_preds,
                mask_feats,
                img_metas,
            )
        return results
