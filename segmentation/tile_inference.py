#!/usr/bin/env python
"""Run collagen segmentation on individual tiles.

Usage:
    python tile_inference.py --tile path/to/tile_or_folder_or_csv --save_mask

`--tile` accepts:
  - a single image file (.png/.jpg/.jpeg/.tif/.tiff)
  - a folder of image files (searched recursively)
  - a CSV file with a 'tile_path' column
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataloader import TileDataset, list_tile_paths
from model.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Tile-level collagen segmentation.")
    parser.add_argument(
        "--tile", required=True,
        help="Path to a tile image, a folder of tile images, or a CSV with a 'tile_path' column.",
    )
    parser.add_argument(
        "--save_mask", action="store_true",
        help="Save each predicted binary mask as a PNG next to the report CSV.",
    )
    parser.add_argument("--weights", default="model/weights/pretrained_weights.pth")
    parser.add_argument("--output_dir", default="results/tile_inference")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mpp", type=float, default=None,
        help="Pixel size of the input tiles in microns/pixel, at their original "
             "(pre-resize) resolution. Assumes square pixels. Required to report "
             "collagen_area_mm2 and physical island sizes; without it only "
             "pixel-based values are reported.",
    )
    parser.add_argument(
        "--island_chunk_size", type=int, default=5000,
        help="Flush accumulated per-island features to disk every N rows, to "
             "bound memory usage on large runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = utils.get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "tile_inference_report.csv"
    islands_path = output_dir / "tile_inference_islands.csv"
    for path in (report_path, islands_path):
        if path.exists():
            path.unlink()

    tile_paths = list_tile_paths(args.tile)
    if not tile_paths:
        raise SystemExit(f"No tiles found for --tile {args.tile}")
    print(f"Found {len(tile_paths)} tile(s).")

    mpp = (args.mpp, args.mpp) if args.mpp is not None else None
    if mpp is None:
        print("No --mpp given: only pixel-based values will be reported (no mm2/µm).")

    dataset = TileDataset(tile_paths, target_size=args.target_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = utils.load_model(
        UNet(in_channels=3, out_channels=64, num_classes=1), args.weights, device
    )

    report_records = []
    island_chunks = []

    with torch.no_grad():
        for images, paths, orig_h, orig_w in tqdm(loader, desc="Running inference"):
            images = images.to(device, non_blocking=True)
            logits, _, _ = model(images)
            masks = utils.predict_mask(logits, threshold=args.threshold)

            for path, mask, h, w in zip(paths, masks, orig_h.tolist(), orig_w.tolist()):
                mpp_eff = utils.effective_mpp(h, w, args.target_size, mpp)
                n_pixels, area_mm2 = utils.compute_mask_area(mask, mpp_eff)

                record = {"tile_path": path, "collagen_area_px2": n_pixels}
                if area_mm2 is not None:
                    record["collagen_area_mm2"] = area_mm2

                if args.save_mask:
                    mask_path = mask_dir / f"{Path(path).stem}_mask.png"
                    utils.save_mask_png(mask, mask_path)
                    record["mask_path"] = str(mask_path)

                report_records.append(record)

                islands = utils.extract_collagen_features(mask, mpp_eff, path)
                if not islands.empty:
                    island_chunks.append(islands)

            if sum(len(c) for c in island_chunks) >= args.island_chunk_size:
                utils.append_chunk_csv(island_chunks, islands_path)

    utils.append_chunk_csv(island_chunks, islands_path)

    report = pd.DataFrame(report_records)
    report.to_csv(report_path, index=False)
    print(f"Saved tile-level report ({len(report)} tiles) to {report_path}")
    if islands_path.exists():
        print(f"Saved per-island features to {islands_path}")


if __name__ == "__main__":
    main()
