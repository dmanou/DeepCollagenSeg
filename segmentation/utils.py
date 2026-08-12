"""Inference helpers for the segmentation pipeline."""

import numpy as np
import torch
from PIL import Image


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model, weights_path, device):
    """Load a checkpoint (dict with a `model_state_dict` key, or a bare
    state_dict) and return the model on `device` in eval mode."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_mask(logits, threshold=0.5):
    """Sigmoid + threshold -> binary numpy mask, one per item in the batch."""
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()
    return mask.detach().cpu().numpy()


def save_mask_png(mask, out_path):
    """Save a single-channel binary mask (values in {0,1}) as an 8-bit PNG."""
    mask_uint8 = (np.squeeze(mask) * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(out_path)


def compute_mask_area(mask, orig_h, orig_w, target_size, mpp=None):
    """Physical area (mm²) of a predicted mask.

    The mask is predicted at `target_size` resolution, but the tile was
    resized *down* from its original (orig_h, orig_w) to get there. So one
    mask pixel does not cover `mpp` µm² of tissue - it covers
    `mpp * scale` µm² on each axis, where `scale` is the (measured, not
    assumed) resize factor. `mpp` is (mpp_x, mpp_y) in microns/pixel *at
    the tile's original resolution*; pass None if unknown, in which case
    only the pixel count is returned.
    """
    n_pixels = int(mask.sum())

    if mpp is None:
        return n_pixels, None

    scale_x = orig_w / target_size
    scale_y = orig_h / target_size
    mpp_eff_x = mpp[0] * scale_x
    mpp_eff_y = mpp[1] * scale_y

    area_mm2 = n_pixels * mpp_eff_x * mpp_eff_y / 1e6
    return n_pixels, area_mm2
