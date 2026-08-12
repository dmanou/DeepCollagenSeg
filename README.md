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

### Create a Conda environment (Python 3.9.16)
```bash
conda create -n segcoll python=3.9.16 -y
conda activate segcoll
cd segmentation
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### PyTorch installation
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
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
tile, with its predicted collagen ratio) and, with `--save_mask`, a PNG
mask per tile under `results/tile_inference/masks/`.

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

Follow intasllation instruction from the [official stardist github](https://github.com/stardist/stardist).

```bash
cd stardist
```
### Run
```bash
python stardist_inference.py --tile path/to/tile_or_folder_or_csv --save_labels
```
Writes `results/stardist/stardist_report.csv` (one row per tile, with its
predicted cell count) and, with `--save_labels`, an instance label map
(`.npy`) per tile under `results/stardist/labels/`.

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
