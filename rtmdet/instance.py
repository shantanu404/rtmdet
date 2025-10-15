from dataclasses import dataclass, field

import torch


@dataclass
class InstanceData:
    bboxes: torch.Tensor = field(default_factory=torch.Tensor)
    priors: torch.Tensor = field(default_factory=torch.Tensor)
    scores: torch.Tensor = field(default_factory=torch.Tensor)
    labels: torch.Tensor = field(default_factory=torch.Tensor)
    kernels: torch.Tensor = field(default_factory=torch.Tensor)
    masks: torch.Tensor = field(default_factory=torch.Tensor)

    def __getitem__(self, index):
        return InstanceData(
            bboxes=self.bboxes[index]
            if self.bboxes.numel() > 0
            else torch.empty((0, 4)),
            priors=self.priors[index]
            if self.priors.numel() > 0
            else torch.empty((0, 4)),
            scores=self.scores[index] if self.scores.numel() > 0 else torch.empty((0,)),
            labels=self.labels[index] if self.labels.numel() > 0 else torch.empty((0,)),
            kernels=self.kernels[index]
            if self.kernels.numel() > 0
            else torch.empty((0,)),
            masks=self.masks[index] if self.masks.numel() > 0 else torch.empty((0,)),
        )
