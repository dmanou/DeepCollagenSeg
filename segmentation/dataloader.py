"""Dataset and tile-path resolution used by the segmentation pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TileDataset(Dataset):
    """Loads plain RGB tiles (jpg/png/tif) for inference.
    Accepts either a list of file paths or a pandas DataFrame with a
    `tile_path` column (as produced by `preprocessing.py`). Returns each
    image alongside its source path so predictions can be traced back to
    a file.
    """

    def __init__(self, tile_paths, target_size=512):
        if isinstance(tile_paths, pd.DataFrame):
            self.tile_paths = list(tile_paths["tile_path"])
        else:
            self.tile_paths = list(tile_paths)

        self.to_image = v2.ToImage()
        self.target_size = (target_size, target_size)

    def __len__(self):
        return len(self.tile_paths)

    def _resize(self, image):
        # No-op if the tile is already at the target size: cv2.resize with a
        # scale factor of 1 is an identity transform anyway, but skipping it
        # avoids the redundant call when tiles come pre-sized.
        if image.shape[:2] == self.target_size:
            return image
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx):
        path = self.tile_paths[idx]

        image = np.array(Image.open(path).convert("RGB"))
        image = self._resize(image)
        image = self.to_image(image).float() / 255.0

        return image, str(path)


def list_tile_paths(tile_arg: str) -> list[str]:
    """Resolve a `--tile` CLI argument to a list of image paths.

    `tile_arg` can be a single image file, a directory of images (searched
    recursively), or a CSV file containing a `tile_path` column.
    """
    path = Path(tile_arg)

    if path.is_dir():
        return sorted(
            str(p) for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return list(df["tile_path"])

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [str(path)]

    raise ValueError(
        "--tile must be an image file, a folder of images, or a CSV with a "
        f"'tile_path' column, got: {tile_arg}"
    )
