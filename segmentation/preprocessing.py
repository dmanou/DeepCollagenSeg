"""Tissue detection and tiling for whole-slide images (WSI).

Works with any format tiatoolbox supports (.svs, .ndpi, .tif, DICOM, ...).

A note on the thumbnail/downsample handling, since it's easy to get subtly
wrong: the tissue mask is computed on a small thumbnail, but tile
coordinates must be reported in full-resolution (level 0) pixels. The two
are related by a downsample factor - and that factor is NOT a fixed
constant across formats.

  - A pyramidal file (.svs, .ndpi) has several native resolution levels;
    asking for "pyramid level 0" always gives you full resolution.
  - Some DICOM exports only expose a single level, so a previous version of
    this pipeline relied on a *separate* pre-computed low-resolution
    preview file, with its own known downsample relative to the full-res
    file.

To work correctly for both cases without special-casing the file format,
we request the thumbnail by physical **objective power** (`units="power"`)
rather than by pyramid level, and then *measure* the actual downsample
factor from the reader's reported slide dimensions instead of assuming it.
This is robust regardless of how many pyramid levels the file has.
"""

from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import tqdm

DEFAULT_TILE_SIZE = 1024
DEFAULT_THUMBNAIL_POWER = 1.25  # objective power used for tissue detection
DEFAULT_TISSUE_THRESHOLD = 0.05
DEFAULT_JPEG_QUALITY = 95


# -----------------------------------------------------------------------------
# Thumbnail + tissue mask
# -----------------------------------------------------------------------------

def get_thumbnail(reader: WSIReader, power: float = DEFAULT_THUMBNAIL_POWER):
    """Low-power thumbnail plus its *actual* (x, y) downsample vs. level 0.

    See module docstring for why this is measured rather than assumed.
    """
    thumbnail = reader.slide_thumbnail(resolution=power, units="power")
    slide_w, slide_h = reader.info.slide_dimensions
    thumb_h, thumb_w = thumbnail.shape[:2]

    downsample_x = slide_w / thumb_w
    downsample_y = slide_h / thumb_h

    return thumbnail, downsample_x, downsample_y


def detect_tissue(thumbnail: np.ndarray) -> np.ndarray:
    """Otsu threshold on a red/blue-vs-green heatmap; isolates H&E stained tissue.
    https://doi.org/10.1038/s41598-023-50183-4
    Schreiber algorithm
    """
    red = thumbnail[:, :, 0].astype(np.float32)
    green = thumbnail[:, :, 1].astype(np.float32)
    blue = thumbnail[:, :, 2].astype(np.float32)
    red_green = np.maximum(red - green, 0)
    blue_green = np.maximum(blue - green, 0)
    heatmap = red_green * blue_green
    threshold = threshold_otsu(heatmap)
    return heatmap > threshold


# -----------------------------------------------------------------------------
# Tile selection
# -----------------------------------------------------------------------------

def compute_tile_coordinates(
    mask: np.ndarray,
    downsample_x: float,
    downsample_y: float,
    tile_size: int = DEFAULT_TILE_SIZE,
    tissue_threshold: float = DEFAULT_TISSUE_THRESHOLD,
) -> list[tuple[int, int]]:
    """Level-0 (x, y) coordinates of tiles whose tissue coverage exceeds the threshold."""
    low_tile_w = max(1, round(tile_size / downsample_x))
    low_tile_h = max(1, round(tile_size / downsample_y))

    coords = []
    h, w = mask.shape
    for y in range(0, h, low_tile_h):
        for x in range(0, w, low_tile_w):
            patch = mask[y:y + low_tile_h, x:x + low_tile_w]
            if patch.mean() > tissue_threshold:
                coords.append((int(round(x * downsample_x)), int(round(y * downsample_y))))
    return coords


# -----------------------------------------------------------------------------
# Tissue area
# -----------------------------------------------------------------------------

def get_base_mpp(reader: WSIReader) -> tuple[float, float] | None:
    """(mpp_x, mpp_y) in microns/pixel at full resolution (level 0), if available."""
    mpp = getattr(reader.info, "mpp", None)
    if mpp is None:
        return None
    mpp = np.asarray(mpp).ravel()
    if mpp.size == 0 or np.any(np.isnan(mpp)):
        return None
    if mpp.size == 1:
        return float(mpp[0]), float(mpp[0])
    return float(mpp[0]), float(mpp[1])


def compute_tissue_area(
    reader: WSIReader, mask: np.ndarray, downsample_x: float, downsample_y: float
) -> dict:
    tissue_area_px = int(mask.sum())
    base_mpp = get_base_mpp(reader)

    result = {"tissue_area_px2_at_thumbnail": tissue_area_px}

    if base_mpp is not None:
        # mpp of the thumbnail = mpp at level 0, scaled up by the measured downsample
        mpp_x = base_mpp[0] * downsample_x
        mpp_y = base_mpp[1] * downsample_y
        result["mpp_x"] = mpp_x
        result["mpp_y"] = mpp_y
        result["tissue_area_mm2"] = tissue_area_px * mpp_x * mpp_y / 1e6
    else:
        result["mpp_x"] = None
        result["mpp_y"] = None
        result["tissue_area_mm2"] = None
        logging.warning("mpp unavailable: tissue area in mm2 not computed.")

    return result


# -----------------------------------------------------------------------------
# Tile extraction
# -----------------------------------------------------------------------------

def save_tile(
    reader: WSIReader,
    coord: tuple[int, int],
    output_dir: Path,
    slide_id: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict:
    x, y = coord
    tile = reader.read_rect(
        location=(x, y),
        size=(tile_size, tile_size),
        resolution=0,
        units="level",
    )
    if tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    tile_path = output_dir / f"tile_x_{x}_y_{y}.jpg"
    Image.fromarray(tile).save(tile_path, format="JPEG", quality=jpeg_quality)
    return {
        "slide_id": slide_id,
        "x": x,
        "y": y,
        "tile_size": tile_size,
        "tile_path": str(tile_path),
    }


def tile_slide(
    wsi_path: str,
    output_dir: Path | str,
    tile_size: int = DEFAULT_TILE_SIZE,
    thumbnail_power: float = DEFAULT_THUMBNAIL_POWER,
    tissue_threshold: float = DEFAULT_TISSUE_THRESHOLD,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    n_io_workers: int = 8,
) -> pd.DataFrame:
    """Tile a single WSI into `output_dir` (tissue tiles only).

    Returns a manifest DataFrame with one row per tile. If the slide was
    already tiled by a previous run (`.done` marker present), the existing
    manifest is reused instead of re-tiling.
    """
    output_dir = Path(output_dir)
    done_marker = output_dir / ".done"
    manifest_path = output_dir / "tiles.csv"

    if done_marker.exists() and manifest_path.exists():
        return pd.read_csv(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    reader = WSIReader.open(wsi_path)

    thumbnail, downsample_x, downsample_y = get_thumbnail(reader, thumbnail_power)
    mask = detect_tissue(thumbnail)
    coords = compute_tile_coordinates(mask, downsample_x, downsample_y, tile_size, tissue_threshold)

    slide_id = Path(wsi_path).stem

    with ThreadPoolExecutor(max_workers=n_io_workers) as executor:
        futures = [
            executor.submit(save_tile, reader, c, output_dir, slide_id, tile_size, jpeg_quality)
            for c in coords
        ]
        tile_records = [f.result() for f in as_completed(futures)]

    df = pd.DataFrame(tile_records)
    df.to_csv(manifest_path, index=False)

    areas = compute_tissue_area(reader, mask, downsample_x, downsample_y)
    areas.update({
        "n_tiles_extracted": len(coords),
        "slide_id": slide_id,
        "wsi_path": str(wsi_path),
    })
    with open(output_dir / "slide_info.json", "w") as f:
        json.dump(areas, f, indent=2)
    done_marker.touch()

    return df


def _run_slide_safe(wsi_path: str, output_root: Path, **tile_kwargs) -> bool:
    output_dir = output_root / Path(wsi_path).stem
    try:
        tile_slide(wsi_path, output_dir, **tile_kwargs)
        return True
    except Exception:
        logging.error("Failed on slide %s:\n%s", wsi_path, traceback.format_exc())
        return False


def batch_preprocess(csv_path: str, output_root: Path | str, n_slide_workers: int = 8, **tile_kwargs):
    """Standalone batch mode: tile every slide listed in a CSV (`wsi_path` column)."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(csv_path)

    n_ok = 0
    with ProcessPoolExecutor(max_workers=n_slide_workers) as executor:
        futures = [
            executor.submit(_run_slide_safe, row.wsi_path, output_root, **tile_kwargs)
            for row in tab.itertuples(index=False)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            n_ok += bool(future.result())

    print(f"{n_ok}/{len(tab)} slides tiled successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-tile a list of WSIs.")
    parser.add_argument("--csv", required=True, help="CSV with a 'wsi_path' column.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--thumbnail_power", type=float, default=DEFAULT_THUMBNAIL_POWER)
    parser.add_argument("--tissue_threshold", type=float, default=DEFAULT_TISSUE_THRESHOLD)
    args = parser.parse_args()

    batch_preprocess(
        args.csv,
        args.output_dir,
        tile_size=args.tile_size,
        thumbnail_power=args.thumbnail_power,
        tissue_threshold=args.tissue_threshold,
    )
