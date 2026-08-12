"""Inference helpers for the segmentation pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.measure import label, regionprops_table

REGION_PROPERTIES = (
    "label",
    "area",
    "centroid",
    "bbox",
    "eccentricity",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
)


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


def effective_mpp(orig_h, orig_w, target_size, mpp):
    """(mpp_x, mpp_y) in microns/pixel *at the mask's resolution*.

    The mask is predicted at `target_size`, but the tile was resized down
    from its original (orig_h, orig_w) to get there. So one mask pixel
    does not cover `mpp` µm² - it covers `mpp * scale` µm² on each axis,
    where `scale` is the (measured, not assumed) resize factor. `mpp` is
    (mpp_x, mpp_y) at the tile's *original* resolution; pass None if
    unknown, in which case this returns None too.
    """
    if mpp is None:
        return None

    scale_x = orig_w / target_size
    scale_y = orig_h / target_size
    return mpp[0] * scale_x, mpp[1] * scale_y


def compute_mask_area(mask, mpp_eff=None):
    """Pixel count and physical area (mm²) of a predicted binary mask.

    `mpp_eff` is (mpp_x, mpp_y) *at the mask's resolution* - see
    `effective_mpp`. Pass None if unknown, in which case only the pixel
    count is returned.
    """
    n_pixels = int(mask.sum())

    if mpp_eff is None:
        return n_pixels, None

    area_mm2 = n_pixels * mpp_eff[0] * mpp_eff[1] / 1e6
    return n_pixels, area_mm2


def extract_collagen_features(mask, mpp_eff, tile_path):
    """Per-connected-component ("island") shape features of a binary mask.

    Labels connected components in `mask` and runs `regionprops_table` on
    them, returning one row per island with its label, area, centroid,
    bounding box, eccentricity, perimeter, axis lengths, orientation and
    solidity - all in pixels *at the mask's resolution* (`target_size`).
    `mpp_eff` (see `effective_mpp`) is attached as columns so pixel-based
    values can be converted to physical units downstream; pass None if
    unknown. Returns an empty DataFrame if the mask has no foreground.
    """
    labels = label(mask.astype(np.uint8))

    df = pd.DataFrame(regionprops_table(labels, properties=REGION_PROPERTIES))
    if df.empty:
        return df

    if mpp_eff is not None:
        df["mpp_x"] = mpp_eff[0]
        df["mpp_y"] = mpp_eff[1]
    df["tile_path"] = tile_path

    return df


def append_chunk_csv(frames, path):
    """Concatenate a list of DataFrames and append them to `path`, writing
    a header only if the file doesn't exist yet. Clears `frames` in place
    afterward, so memory use stays bounded across a long run."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)
    frames.clear()
