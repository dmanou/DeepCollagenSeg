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
