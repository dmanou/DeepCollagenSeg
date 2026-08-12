"""Tile-path resolution for the StarDist pipeline.

Kept self-contained (no torch/cv2 dependency) so this pipeline can be used
on its own, without installing the segmentation pipeline's requirements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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
