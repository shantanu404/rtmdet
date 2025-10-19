import cv2
import numpy as np
import torch
import torch.nn as nn

from .model import RTMDet
from .ckpt_loader import load_mmdet_checkpoint

class RTMDetPipeline:
    def __init__(self, device='cuda', weight_path=None):
        self.model = RTMDet(num_classes=80)
        self.device = device

        if weight_path:
            state = load_mmdet_checkpoint(weight_path)
            self.model.load_state_dict(state, strict=True)

        self.means = [103.530, 116.280, 123.675]
        self.stds = [57.375, 57.120, 58.395]

        self.model.to(self.device)
        self.model.eval()

    def predict(self, imgs, retina=False):
        preprocessed_imgs, img_metas = self.preprocess_images(imgs)

        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    results = self.model.predict(preprocessed_imgs, img_metas)
        else:
            with torch.no_grad():
                results = self.model.predict(preprocessed_imgs, img_metas)

        if retina:
            for meta, result in zip(img_metas, results):
                scale_factor = meta['scale_factor']
                result.bboxes /= scale_factor

                result.masks = result.masks.cpu()


                original_h, original_w = meta['original_shape'][:2]
                og_shape = max(original_h, original_w)

                result.masks = nn.functional.interpolate(
                    result.masks.unsqueeze(0),
                    size=(og_shape, og_shape),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                result.masks = result.masks[:, :original_h, :original_w]
                result.masks = result.masks > 0.5
        
        return results

    def preprocess_images(self, imgs):
        preprocessed_imgs = []
        img_metas = []

        for img in imgs:
            h, w = img.shape[:2]

            # Resize keeping aspect ratio
            scale = min(640 / h, 640 / w)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh))

            # Pad to 640x640
            top = 0
            bottom = 640 - nh
            left = 0
            right = 640 - nw
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
            )
            img = cv2.resize(img, (640, 640))
            img_meta = {"img_shape": (640, 640, 3), "scale_factor": scale, "original_shape": (h, w, 3)}
            preprocessed_imgs.append(img)
            img_metas.append(img_meta)

        preprocessed_imgs = np.stack(preprocessed_imgs, axis=0)
        preprocessed_imgs = torch.from_numpy(preprocessed_imgs).float()
        preprocessed_imgs = preprocessed_imgs.permute(0, 3, 1, 2)  # NHWC to NCHW
        preprocessed_imgs = (preprocessed_imgs - torch.tensor(self.means).view(1, 3, 1, 1)) / torch.tensor(self.stds).view(
            1, 3, 1, 1
        )
        preprocessed_imgs = preprocessed_imgs.to(self.device)

        return preprocessed_imgs, img_metas