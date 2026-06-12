"""
Wrapper for the railway-specific depth network, compatible with UniDepthV2.infer() interface.

infer(rgbs, intrinsics=None) -> {"depth": tensor [1, 1, H, W]}
  rgbs: [C, H, W] or [1, C, H, W], uint8, RGB, on CUDA
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

_RAILWAY_NET_CODE = os.path.join(os.path.dirname(__file__), "..", "railway_depth_net", "code")
sys.path.insert(0, os.path.abspath(_RAILWAY_NET_CODE))
from train_depth_anything_lora_metric_v4 import load_full_model, forward_branches  # noqa: E402

_INPUT_SIZE = 518


class RailwayDepthModel:
    def __init__(self, ckpt_dir, model_name="./Depth-Anything-V2-Small-hf", device="cuda"):
        # model_name can be a local path or HuggingFace repo id
        _model_name = os.path.join(os.path.abspath(_RAILWAY_NET_CODE), model_name) \
            if model_name.startswith("./") else model_name
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(_model_name)
        self.model, self.metric_head = load_full_model(_model_name, ckpt_dir, device=device)
        self.model.eval()
        if self.metric_head is not None:
            self.metric_head.eval()

    @torch.no_grad()
    def infer(self, rgbs: torch.Tensor, intrinsics=None):
        """
        rgbs: [C, H, W] uint8 RGB tensor on CUDA
        returns: {"depth": [1, 1, H, W] float32 metric depth in meters}
        """
        if rgbs.ndim == 3:
            rgbs = rgbs.unsqueeze(0)  # [1, C, H, W]
        _, _, H, W = rgbs.shape

        # Normalize to [0,1] float and run through HF processor normalization
        rgb_float = rgbs.float() / 255.0  # [1, 3, H, W]
        # Resize to model input size
        rgb_resized = F.interpolate(rgb_float, size=(_INPUT_SIZE, _INPUT_SIZE),
                                    mode="bilinear", align_corners=False)
        # Apply ImageProcessor mean/std normalization
        mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        pixel_values = (rgb_resized - mean) / std

        with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu"):
            _, pred_metric, _ = forward_branches(
                self.model, self.metric_head, pixel_values,
                (_INPUT_SIZE, _INPUT_SIZE), compute_metric=True
            )

        # Upsample back to original resolution
        depth = F.interpolate(pred_metric.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)  # [1, 1, H, W]
        return {"depth": depth}
