import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from .bbox import DistancePointBBoxCoder
from .blocks.convs import ConvModule
from .blocks.maskfeat import MaskFeatModule
from .instance import InstanceData
from .mlvlgen import MlvlPointGenerator
from .utils import filter_scores_and_topk, select_single_mlvl


class RTMDetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
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

        self.prior_generator = MlvlPointGenerator([8, 16, 32])
        self.num_levels = self.prior_generator.num_levels

        self.bbox_coder = DistancePointBBoxCoder()

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

        cls_scores = []
        bbox_preds = []
        kernel_preds = []

        for i in range(self.num_levels):
            cls_feat = x[i]
            reg_feat = x[i]
            kernel_feat = x[i]
            stride = self.prior_generator.strides[i]

            for cls_layer in self.cls_convs[i]:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs[i]:
                reg_feat = reg_layer(reg_feat)

            for kernel_layer in self.kernel_convs[i]:
                kernel_feat = kernel_layer(kernel_feat)

            cls_score = self.rtm_cls[i](cls_feat)
            bbox_pred = F.relu(self.rtm_reg[i](reg_feat)) * stride[0]
            kernel_pred = self.rtm_kernel[i](kernel_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            kernel_preds.append(kernel_pred)

        return cls_scores, bbox_preds, kernel_preds, mask_feats

    def predict_by_feat(
        self,
        cls_scores,
        bbox_preds,
        kernel_preds,
        mask_feats,
        batch_img_metas,
    ):
        assert len(cls_scores) == len(bbox_preds)

        num_levels = len(cls_scores)

        featmap_size = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_size=featmap_size,
            device=cls_scores[0].device,
            with_stride=True,
        )

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            kernel_pred_list = select_single_mlvl(kernel_preds, img_id)

            results = self._predict_by_single_feat(
                cls_score_list,
                bbox_pred_list,
                kernel_pred_list,
                mask_feats[img_id],
                mlvl_priors,
                img_meta,
            )

            result_list.append(results)

        return result_list

    def _predict_by_single_feat(
        self,
        cls_scores_list,
        bbox_preds_list,
        kernel_preds_list,
        mask_feats,
        mlvl_priors,
        img_meta,
    ):
        nms_pre_threshold = 0.5

        mlvl_bbox_preds = []
        mlvl_kernels = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, bbox_pred, kernel_pred, priors) in enumerate(
            zip(cls_scores_list, bbox_preds_list, kernel_preds_list, mlvl_priors)
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, self.num_gen_params)
            scores = cls_score.sigmoid()

            results = filter_scores_and_topk(
                scores,
                nms_pre_threshold,
                64,
                results=dict(
                    bbox_pred=bbox_pred, kernel_pred=kernel_pred, priors=priors
                ),
            )

            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            kernel_pred = filtered_results["kernel_pred"]
            priors = filtered_results["priors"]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_kernels.append(kernel_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_preds = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors[..., :2], bbox_preds)

        result = InstanceData()

        result.bboxes = bboxes
        result.priors = priors
        result.scores = torch.cat(mlvl_scores)
        result.labels = torch.cat(mlvl_labels)
        result.kernels = torch.cat(mlvl_kernels)

        return self._bbox_mask_post_process(result, mask_feats, img_meta)

    def _bbox_mask_post_process(self, instance_data, mask_feats, img_meta):
        stride = self.prior_generator.strides[0][0]

        if instance_data.bboxes.numel() > 0:
            keep_idxs = batched_nms(
                instance_data.bboxes, instance_data.scores, instance_data.labels, 0.6
            )
            instance_data = instance_data[keep_idxs]

            # process mask
            mask_logits = self._mask_predict_by_feat_single(
                mask_feats, instance_data.kernels, instance_data.priors
            )

            instance_data.masks = mask_logits.sigmoid()
        else:
            h, w = img_meta["img_shape"][:2]
            instance_data.masks = torch.zeros(
                size=(instance_data.bboxes.size(0), h, w),
                dtype=torch.float,
                device=instance_data.bboxes.device,
            )

        return instance_data

    def _parse_dynamic_params(self, flatten_kernels):
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

    def _mask_predict_by_feat_single(self, mask_feats, kernels, priors):
        num_inst = priors.size(0)
        h, w = mask_feats.size()[-2:]

        if num_inst == 0:
            return torch.empty(
                (0, h, w), dtype=mask_feats.dtype, device=mask_feats.device
            )

        coord = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx=0, device=mask_feats.device
        ).reshape(1, -1, 2)

        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)

        relative_coord = (points - coord).permute(0, 2, 1) / (
            strides[..., 0].reshape(-1, 1, 1) * 8
        )
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feats = torch.cat(
            [relative_coord, mask_feats.repeat(num_inst, 1, 1, 1)], dim=1
        )

        weights, biases = self._parse_dynamic_params(kernels)

        n_layers = len(weights)
        x = mask_feats.reshape(1, -1, h, w)

        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = F.relu(x)

        x = x.reshape(num_inst, h, w)
        return x
