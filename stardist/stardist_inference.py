#!/usr/bin/env python
"""Nucleus / cell counting on H&E tiles using a pretrained StarDist model.

Usage:
    python stardist_inference.py --tile path/to/tile_or_folder_or_csv --save_labels

`--tile` accepts:
  - a single image file (.png/.jpg/.jpeg/.tif/.tiff)
  - a folder of image files (searched recursively)
  - a CSV file with a 'tile_path' column

Uses the pretrained "2D_versatile_he" StarDist model (via the public
`stardist` PyPI package - https://github.com/stardist/stardist), which is
tuned for H&E-stained histology. No training of your own is required; the
model weights are downloaded automatically on first use.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.io import imread
from stardist.models import StarDist2D
from tqdm import tqdm

from utils import list_tile_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Nucleus counting on H&E tiles with StarDist.")
    parser.add_argument(
        "--tile", required=True,
        help="Path to a tile image, a folder of tile images, or a CSV with a 'tile_path' column.",
    )
    parser.add_argument(
        "--save_labels", action="store_true", default=False,
        help="Save each tile's instance label map (.npy) next to the report CSV. "
             "Off by default: only the CSV report (tile_path, n_cells) is written.",
    )
    parser.add_argument(
        "--model_name", default="2D_versatile_he",
        help="Pretrained StarDist2D model to use (default: H&E-tuned model).",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for intensity normalization.")
    parser.add_argument("--pmax", type=float, default=99.8, help="Upper percentile for intensity normalization.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    if args.save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = list_tile_paths(args.tile)
    if not tile_paths:
        raise SystemExit(f"No tiles found for --tile {args.tile}")
    print(f"Found {len(tile_paths)} tile(s).")

    print(f"Loading pretrained StarDist model '{args.model_name}' ...")
    model = StarDist2D.from_pretrained(args.model_name)

    records = []
    for tile_path in tqdm(tile_paths, desc="Counting cells"):
        img = imread(tile_path)
        img = normalize(img, args.pmin, args.pmax)

        labels, _ = model.predict_instances(img)
        n_cells = int(labels.max())

        record = {"tile_path": str(tile_path), "n_cells": n_cells}

        if args.save_labels:
            label_path = labels_dir / f"{Path(tile_path).stem}_labels.npy"
            np.save(label_path, labels)
            record["label_path"] = str(label_path)

        records.append(record)

    report = pd.DataFrame(records)
    report_path = output_dir / "stardist_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Saved report ({len(report)} tiles) to {report_path}")


if __name__ == "__main__":
    main()