from typing_extensions import NamedTuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from .backbone import CSPNeXt
from .bbox import DistancePointBBoxCoder
from .head import RTMDetHead
from .mlvlgen import MlvlPointGenerator
from .neck import CSPNeXtPAFPN

InstanceData = NamedTuple(
    "InstanceData",
    [
        ("bboxes", torch.Tensor),
        ("scores", torch.Tensor),
        ("labels", torch.Tensor),
        ("masks", torch.Tensor),
    ],
)

class RTMDetPre(nn.Module):
    def __init__(self, means, stds, device='cuda'):
        super().__init__()
        means = torch.tensor(means).reshape(1, 3, 1, 1)
        stds = torch.tensor(stds).reshape(1, 3, 1, 1)

        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
    
    def forward(self, img_batch):
        # assert img_batch.shape[1:] == (3, 640, 640), "Input tensor must have shape (N, 3, 640, 640)"
        img_batch = (img_batch - self.means) / self.stds
        return img_batch

class RTMDetPost(nn.Module):
    def __init__(self, num_classes, parse_kernel_fn, priors, coord, mask_stride, nms_pre_threshold, nms_threshold):
        super().__init__()
        self.num_classes = num_classes
        self.parse_kernel_fn = parse_kernel_fn

        self.register_buffer("coord", coord)
        self.register_buffer("priors", priors)

        self.mask_stride = mask_stride
        self.nms_pre_threshold = nms_pre_threshold
        self.nms_threshold = nms_threshold

        self.bbox_coder = DistancePointBBoxCoder()

    def forward(self, x):
        level_preds, mask_feats = x

        batch_size, _, _ = level_preds.shape

        mask_feats = mask_feats.squeeze(0)

        results = []

        for img_id in range(batch_size):
            cls_scores = level_preds[img_id, :, : self.num_classes].squeeze(0)
            box_preds = level_preds[img_id, :, self.num_classes : self.num_classes + 4].squeeze(0)
            kernel_preds = level_preds[img_id, :, self.num_classes + 4 :].squeeze(0)

            cls_scores = cls_scores.sigmoid()
            scores, label = cls_scores.max(dim=1)
            keep_idxs = scores > self.nms_pre_threshold

            scores = scores[keep_idxs]
            label = label[keep_idxs]
            box_preds = box_preds[keep_idxs, :]
            kernel_preds = kernel_preds[keep_idxs]
            batch_priors = self.priors[keep_idxs]
            boxes = self.bbox_coder.decode(batch_priors[..., :2], box_preds)

            keep_idxs = batched_nms(boxes, scores, label, self.nms_threshold)

            scores = scores[keep_idxs]
            label = label[keep_idxs]
            boxes = boxes[keep_idxs]
            kernel_preds = kernel_preds[keep_idxs]
            batch_priors = batch_priors[keep_idxs]

            mask = self._mask_predict_by_feat_single(mask_feats, kernel_preds, batch_priors)
            mask = F.interpolate(
                mask.unsqueeze(0),
                scale_factor=self.mask_stride,
                mode="bilinear",
            )
            mask = (mask.sigmoid().squeeze(0) > 0.5).to(torch.bool)

            results.append(InstanceData(boxes, scores, label, mask))

        return results

    def _bbox_mask_post_process(self, instance_data, mask_feats):
        stride = self.prior_generator.strides[0][0]

        keep_idxs = batched_nms(
            instance_data.bboxes, instance_data.scores, instance_data.labels, 0.5
        )
        filtered_bboxes = instance_data.bboxes[keep_idxs]
        filtered_priors = instance_data.priors[keep_idxs]
        filtered_scores = instance_data.scores[keep_idxs]
        filtered_labels = instance_data.labels[keep_idxs]
        filtered_kernels = instance_data.kernels[keep_idxs]

        # process mask
        mask_logits = self._mask_predict_by_feat_single(
            mask_feats, filtered_kernels, filtered_priors
        )

        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            scale_factor=stride,
            mode="bilinear",
        )

        new_masks = (mask_logits.sigmoid().squeeze(0) > 0.5).to(
            torch.bool
        )

        instance_data = InstanceData(
            bboxes=filtered_bboxes,
            priors=filtered_priors,
            scores=filtered_scores,
            labels=filtered_labels,
            kernels=filtered_kernels,
            masks=new_masks,
        )

        return instance_data


    def _mask_predict_by_feat_single(self, mask_feats, kernels, priors):
        num_inst = priors.size(0)
        h, w = mask_feats.size()[-2:]

        coord = self.coord.unsqueeze(0)

        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)

        relative_coord = (points - coord).permute(0, 2, 1) / (
            strides[..., 0].reshape(-1, 1, 1) * 8
        )
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feats = torch.cat(
            [relative_coord, mask_feats.repeat(num_inst, 1, 1, 1)], dim=1
        )

        weights, biases = self.parse_kernel_fn(kernels)

        n_layers = len(weights)
        x = mask_feats.reshape(1, -1, h, w)

        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = F.relu(x)

        x = x.reshape(num_inst, h, w)
        return x


class RTMDet(nn.Module):
    def __init__(self, num_classes, strides):
        super().__init__()

        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN(out_channels=320)
        self.bbox_head = RTMDetHead(
            in_channels=320,
            num_classes=num_classes,
            strides=strides,
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        return self.bbox_head(features)

class RTMDetPipeline(nn.Module):
    def __init__(self, means = [103.530, 116.280, 123.675], stds = [57.375, 57.120, 58.395], num_classes = 80, strides=[8, 16, 32]):
        super().__init__()
        self.input_size = (640, 640)
        self.prior_generator = MlvlPointGenerator(strides)

        priors = self.prior_generator.grid_priors(
            featmap_size=[
                (self.input_size[0] // stride, self.input_size[1] // stride)
                for stride in strides
            ],
            with_stride=True,
        )

        coord = priors[0][:, :2].clone()
        self.priors = torch.cat(priors)

        self.preprocess = RTMDetPre(means, stds)
        self.model = RTMDet(num_classes, strides)
        self.postprocess = RTMDetPost(
            num_classes=num_classes,
            parse_kernel_fn=self.model.bbox_head.parse_dynamic_params,
            priors=self.priors,
            coord=coord,
            mask_stride=strides[0],
            nms_pre_threshold=0.3,
            nms_threshold=0.3,
        )

    def predict(self, imgs, retina_face=True, device="cuda"):
        imgs, og_sizes, processed_sizes = self._prepare_images(imgs)
        img_batch = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(device)

        self.eval()
        with torch.no_grad():
            results = self.forward(img_batch)

        if retina_face:
            for i, result in enumerate(results):
                h, w = og_sizes[i]
                nh, nw = processed_sizes[i]
                scale = min(640 / h, 640 / w)

                boxes = result.bboxes / scale

                masks = result.masks[:, :nh, :nw]
                masks = F.interpolate(
                    masks.unsqueeze(0).float(),
                    size=(h, w),
                    mode="bilinear",
                ).squeeze(0)

                results[i] = InstanceData(
                    bboxes=boxes,
                    scores=result.scores,
                    labels=result.labels,
                    masks=masks,
                )
        return results

    def _prepare_images(self, imgs):
        resized_imgs = []
        original_sizes = []
        processed_sizes = []
        for img in imgs:
            h, w = img.shape[:2]
            original_sizes.append((h, w))

            ratio = min(self.input_size[0] / h, self.input_size[1] / w)
            nh, nw = int(h * ratio), int(w * ratio)
            processed_sizes.append((nh, nw))

            img = cv2.resize(img, (nw, nh))

            pad_right = self.input_size[1] - nw
            pad_bottom = self.input_size[0] - nh
            img = cv2.copyMakeBorder(
                img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
            )

            img = cv2.resize(img, self.input_size)
            resized_imgs.append(img)

        return resized_imgs, original_sizes, processed_sizes

    def forward(self, img_batch):
        img_batch = self.preprocess(img_batch)
        features = self.model(img_batch)
        results = self.postprocess(features)
        return results
