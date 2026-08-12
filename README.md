# DeepCollagenSeg
## Overview
Pytorch implementation of the paper "XX"
![Collagen segmentation](segmentation.png)

This repository has two independent pipelines that can be installed and
used separately:
- **`segmentation/`** — collagen segmentation with a pretrained UNet (this
  paper's model).
- **`stardist/`** — nucleus/cell counting on H&E tiles, using the public,
  pretrained [StarDist](https://github.com/stardist/stardist) model (not
  part of this paper; included here as a convenient companion tool).

## Clone the repository
```bash
git clone https://github.com/dmanou/DeepCollagenSeg.git
cd DeepCollagenSeg
```

---

## Collagen segmentation (`segmentation/`)

### Create a Conda environment (Python 3.12.13)
```bash
conda create -n segcoll python=3.9.16 -y
conda activate segcoll
cd segmentation
```
### Install PyTorch first
Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
select your OS / package manager / CUDA version, and run the install
command it gives you. **Do this before the next step.** For example:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
### Install the rest (includes tiatoolbox)
```bash
pip install -r requirements.txt
```
`tiatoolbox` declares `torch>=2.5.0` among its dependencies (we only use
`tiatoolbox.wsicore.wsireader` for reading WSIs and tiling - no models, so
the exact torch version doesn't otherwise matter to us). Because pip's
default behaviour is to only upgrade a dependency if what's installed
doesn't satisfy it, your torch from step 3 is left alone here as long as
it's `>=2.5.0`.

If you install an **older** torch in step 3 (below tiatoolbox's floor),
this step will upgrade it - restore your version afterward:
```bash
pip install torch==<your version> torchvision==<your version> --index-url <your index> --force-reinstall --no-deps
```
This is safe even though `pip check` may then flag tiatoolbox's
declared requirement as unmet: we never exercise tiatoolbox's
torch-dependent code, only its WSI reader.
### Download the pretrained weights
```bash
mkdir -p model/weights
wget https://github.com/dmanou/DeepCollagenSeg/releases/download/v1.0.0/pretrained_weights.pth -O model/weights/pretrained_weights.pth
```
### Run inference

**Tile inference** — run the model on a single tile, a folder of tiles, or a
CSV listing tile paths:
```bash
python tile_inference.py --tile path/to/tile_or_folder_or_csv --save_mask
```
Writes `results/tile_inference/tile_inference_report.csv` (one row per
tile, with `collagen_area_px2` and, if `--mpp` is given, `collagen_area_mm2`)
and, with `--save_mask`, a PNG mask per tile under `results/tile_inference/masks/`.

**Whole-slide image** — tile a WSI (tissue detection + patch extraction)
and run inference on it in one command:
```bash
python wsi_inference.py --wsi path/to/slide.svs --preprocess
```
If the slide was already tiled by a previous run, drop `--preprocess` to
reuse the existing tiles. Works with any format supported by
[tiatoolbox](https://tia-toolbox.readthedocs.io) (.svs, .ndpi, .tif,
DICOM...) — the tissue-detection thumbnail is requested by objective power
rather than pyramid level, so it behaves consistently whether the file has
many resolution levels or effectively just one. Results are written to
`results/wsi_inference/<slide_id>/`.

Run `python tile_inference.py -h` or `python wsi_inference.py -h` for the
full list of options (batch size, decision threshold, tile size, etc.).

---

## Cell counting (`stardist/`)

Nucleus counting on H&E tiles using the pretrained `2D_versatile_he`
StarDist model — no training required, weights are downloaded
automatically on first use.

### Install
```bash
cd stardist
pip install -r requirements.txt
```
### Run
```bash
python stardist_inference.py --tile path/to/tile_or_folder_or_csv --save_labels
```
Writes `results/stardist_report.csv` (one row per tile, with its predicted
cell count). Instance label maps are **not** saved by default; pass
`--save_labels` to also write a `.npy` per tile under `results/labels/`.

This pipeline is fully independent from `segmentation/` — it doesn't need
PyTorch or the collagen model's weights, only the `stardist` package
(itself a thin dependency on the public
[stardist](https://pypi.org/project/stardist/) PyPI package; no code from
that project is copied into this repo).

## Project structure
```
DeepCollagenSeg/
├── segmentation/
│   ├── model/
│   │   ├── unet.py           # UNet with two auxiliary heads
│   │   ├── blocks.py          # Conv blocks used by the UNet
│   │   └── weights/           # Pretrained weights go here (see above)
│   ├── dataloader.py           # Dataset + tile-path resolution
│   ├── preprocessing.py        # Tissue detection & WSI tiling
│   ├── utils.py                 # Inference helpers (device, load_model...)
│   ├── tile_inference.py        # python tile_inference.py --tile ... --save_mask
│   ├── wsi_inference.py         # python wsi_inference.py --wsi ... --preprocess
│   └── requirements.txt
└── stardist/
    ├── stardist_inference.py   # python stardist_inference.py --tile ... --save_labels
    ├── utils.py                 # Tile-path resolution (standalone)
    └── requirements.txt
```
