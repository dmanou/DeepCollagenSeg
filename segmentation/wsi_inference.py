#!/usr/bin/env python
"""Run collagen segmentation on a whole-slide image (WSI).

Usage:
    python wsi_inference.py --wsi path/to/slide.svs --preprocess

`--preprocess` detects tissue and tiles the slide before running inference.
Omit it to reuse tiles produced by a previous run (found under
`<output_dir>/<slide_id>/tiles/`).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataloader import TileDataset
from model.unet import UNet
from preprocessing import (
    DEFAULT_THUMBNAIL_POWER,
    DEFAULT_TILE_SIZE,
    DEFAULT_TISSUE_THRESHOLD,
    get_slide_mpp,
    tile_slide,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-slide-image collagen segmentation.")
    parser.add_argument("--wsi", required=True, help="Path to a WSI file (.svs, .ndpi, .tif, DICOM...).")
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Detect tissue and tile the slide before inference. "
             "Omit if the slide has already been tiled by a previous run.",
    )
    parser.add_argument("--weights", default="model/weights/pretrained_weights.pth")
    parser.add_argument("--output_dir", default="results/wsi_inference")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--thumbnail_power", type=float, default=DEFAULT_THUMBNAIL_POWER,
        help="Objective power used for the tissue-detection thumbnail. Works "
             "the same way for single-level and pyramidal formats.",
    )
    parser.add_argument("--tissue_threshold", type=float, default=DEFAULT_TISSUE_THRESHOLD)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_mask", action="store_true",
        help="Save each tile's predicted binary mask as a PNG.",
    )
    parser.add_argument(
        "--island_chunk_size", type=int, default=5000,
        help="Flush accumulated per-island features to disk every N rows, to "
             "bound memory usage on large slides.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = utils.get_device()
    print(f"Using device: {device}")

    slide_id = Path(args.wsi).stem
    output_dir = Path(args.output_dir) / slide_id
    tiles_dir = output_dir / "tiles"
    mask_dir = output_dir / "masks"

    if args.preprocess:
        print(f"Tiling {args.wsi} ...")
        manifest = tile_slide(
            args.wsi,
            tiles_dir,
            tile_size=args.tile_size,
            thumbnail_power=args.thumbnail_power,
            tissue_threshold=args.tissue_threshold,
        )
    else:
        manifest_path = tiles_dir / "tiles.csv"
        if not manifest_path.exists():
            raise SystemExit(
                f"No tile manifest found at {manifest_path}. Run with --preprocess first."
            )
        manifest = pd.read_csv(manifest_path)

    print(f"{len(manifest)} tissue tile(s) to run inference on.")
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    mpp = get_slide_mpp(args.wsi)
    if mpp is None:
        print("Warning: mpp unavailable for this slide - only pixel-based values will be reported.")

    report_path = output_dir / "wsi_inference_report.csv"
    islands_path = output_dir / "wsi_inference_islands.csv"
    for path in (report_path, islands_path):
        if path.exists():
            path.unlink()

    dataset = TileDataset(manifest, target_size=args.target_size)
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

    print(f"Slide-level total collagen area: {report['collagen_area_px2'].sum()} px2 (at model resolution)")
    if "collagen_area_mm2" in report.columns:
        print(f"Slide-level total collagen area: {report['collagen_area_mm2'].sum():.4f} mm2")
    print(f"Saved tile-level report ({len(report)} tiles) to {report_path}")
    if islands_path.exists():
        print(f"Saved per-island features to {islands_path}")


if __name__ == "__main__":
    main()
