from typing import Dict, TypeAlias

import numpy as np
import torch
from torch.serialization import add_safe_globals

StateDict: TypeAlias = Dict[str, torch.Tensor]

HistoryBufferDummy = type("HistoryBuffer", (), {})
HistoryBufferDummy.__module__ = "mmengine.logging.history_buffer"


def load_mmdet_checkpoint(path):
    add_safe_globals(
        [
            HistoryBufferDummy,
            np.dtype,
            np.core.multiarray.scalar,
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.float64,
            np.dtypes.Float64DType,
            np.dtypes.Int64DType,
        ]
    )

    ckpt = torch.load(path, weights_only=True)

    state_dict = ckpt.get("state_dict", ckpt)

    return state_dict


def check_params_update(model, sd):
    before_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(sd, strict=False)
    after_sd = model.state_dict()

    for name, before in before_sd.items():
        after = after_sd[name]
        if torch.equal(before, after):
            print(f"[WARN] param {name} NOT updated")
