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

CSV_PATH = "/NAS/coolio/dorian/PROJETS/MENINGIOME_SEG/ANAPATH/model_v2/database/tab_wsi_preprocess_full.csv"
OUTPUT_ROOT = Path(
    "/NAS/coolio/dorian/PROJETS/MENINGIOME_SEG/ANAPATH/model_v2/database/unlabelled"
)

TILE_SIZE = 1024
DOWNSAMPLE = 16
LOW_TILE_SIZE = TILE_SIZE // DOWNSAMPLE
TISSUE_THRESHOLD = 0.05

JPEG_QUALITY = 95  # 90-95 = quasi indiscernable visuellement, gain de place énorme

N_SLIDE_WORKERS = 32     # nombre de lames traitées en parallèle (process pool -> vrai multi-cœur)
N_TILE_IO_WORKERS = 24   # threads pour lire/écrire les tuiles à l'intérieur d'une lame

LOG_PATH = OUTPUT_ROOT / "errors.log"
AREAS_CSV = OUTPUT_ROOT / "slide_areas.csv"
ALL_TILES_CSV = OUTPUT_ROOT / "all_tiles.csv"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# -----------------------------------------------------------------------------
# Tissue mask
# -----------------------------------------------------------------------------

def preprocessing(img: np.ndarray) -> np.ndarray:
    red = img[:, :, 0].astype(np.float32)
    green = img[:, :, 1].astype(np.float32)
    blue = img[:, :, 2].astype(np.float32)
    red_green = np.maximum(red - green, 0)
    blue_green = np.maximum(blue - green, 0)
    heatmap = red_green * blue_green
    threshold = threshold_otsu(heatmap)
    return heatmap > threshold

# -----------------------------------------------------------------------------
# Tile selection
# -----------------------------------------------------------------------------

def compute_tile_coordinates(mask: np.ndarray) -> list[tuple[int, int]]:
    """Retourne les coordonnées (niveau 0) des tuiles contenant du tissu."""
    coords = []
    h, w = mask.shape
    for y in range(0, h, LOW_TILE_SIZE):
        for x in range(0, w, LOW_TILE_SIZE):
            patch = mask[y:y + LOW_TILE_SIZE, x:x + LOW_TILE_SIZE]
            if patch.mean() > TISSUE_THRESHOLD:
                coords.append((x * DOWNSAMPLE, y * DOWNSAMPLE))
    return coords

# -----------------------------------------------------------------------------
# Tissue Area
# -----------------------------------------------------------------------------

def get_mpp(reader: WSIReader) -> tuple[float, float] | None:
    """Renvoie (mpp_x, mpp_y) en microns/pixel si disponible, sinon None."""
    mpp = getattr(reader.info, "mpp", None)
    if mpp is None:
        return None
    mpp = np.asarray(mpp).ravel()
    if mpp.size == 0 or np.any(np.isnan(mpp)):
        return None
    if mpp.size == 1:
        return float(mpp[0]), float(mpp[0])
    return float(mpp[0]), float(mpp[1])


def compute_tissue_area(reader_low: WSIReader, mask: np.ndarray) -> dict:
    """Aire de tissu (mm²), calculée directement sur le mask basse résolution.

    Le mask est produit à partir du thumbnail de `reader_low`, donc son mpp
    (µm/pixel) correspond exactement à la résolution du mask : pas besoin de
    repasser par DOWNSAMPLE ou par le reader high resolution.
    """
    tissue_area_px = int(mask.sum())
    mpp = get_mpp(reader_low)

    result = {
        "tissue_area_px2": tissue_area_px,
        "mpp_x": mpp[0] if mpp else None,
        "mpp_y": mpp[1] if mpp else None,
    }

    if mpp is not None:
        result["tissue_area_mm2"] = tissue_area_px * mpp[0] * mpp[1] / 1e6
    else:
        result["tissue_area_mm2"] = None
        logging.warning("mpp indisponible : aire en mm2 non calculée.")

    return result

# -----------------------------------------------------------------------------
# Tile extraction
# -----------------------------------------------------------------------------

def save_tile(
    reader: WSIReader, coord: tuple[int, int], output_dir: Path, slide_id: str
) -> dict:
    x, y = coord
    tile = reader.read_rect(
        location=(x, y),
        size=(TILE_SIZE, TILE_SIZE),
        resolution=0,
        units="level",
    )
    if tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    tile_path = output_dir / f"tile_x_{x}_y_{y}.jpg"
    Image.fromarray(tile).save(tile_path, format="JPEG", quality=JPEG_QUALITY)
    return {
        "slide_id": slide_id,
        "x": x,
        "y": y,
        "tile_size": TILE_SIZE,
        "tile_path": str(tile_path),
    }

def process_slide(low_wsi: str, high_wsi: str) -> dict:
    """Traite une lame : masque, extraction des tuiles, calcul d'aire.

    Retourne un dict de métadonnées (utilisé pour construire le CSV d'aires).
    Toute exception est capturée par l'appelant (run_slide_safe).
    """
    output_dir = OUTPUT_ROOT / Path(high_wsi).parts[-3] / Path(high_wsi).parts[-2]
    done_marker = output_dir / ".done"

    if done_marker.exists():
        # Lame déjà traitée -> on relit juste les infos sauvegardées
        with open(output_dir / "slide_info.json") as f:
            return json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    reader_low = WSIReader.open(low_wsi)
    reader_high = WSIReader.open(high_wsi)

    thumbnail = reader_low.slide_thumbnail(resolution=0, units="level")
    mask = preprocessing(thumbnail)
    coords = compute_tile_coordinates(mask)

    slide_id = Path(high_wsi).parts[-2]

    with ThreadPoolExecutor(max_workers=N_TILE_IO_WORKERS) as executor:
        futures = [
            executor.submit(save_tile, reader_high, c, output_dir, slide_id)
            for c in coords
        ]
        tile_records = [f.result() for f in as_completed(futures)]

    pd.DataFrame(tile_records).to_csv(output_dir / "tiles.csv", index=False)

    areas = compute_tissue_area(reader_low, mask)
    areas["n_tiles_extracted"] = len(coords)
    areas["slide_id"] = slide_id
    areas["wsi_path_hq"] = str(high_wsi)

    with open(output_dir / "slide_info.json", "w") as f:
        json.dump(areas, f, indent=2)
    done_marker.touch()

    return areas

def run_slide_safe(wsi_path_lq: str, wsi_path_hq: str) -> dict | None:
    try:
        return process_slide(wsi_path_lq, wsi_path_hq)
    except Exception:
        logging.error(
            "Echec sur la lame %s :\n%s", wsi_path_hq, traceback.format_exc()
        )
        return None
