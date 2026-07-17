# Segmentation_Collagene

## Overview

Pytorch implementation of the paper "XX"

## 1) Clone the repository

```bash
git clone https://github.com/dmanou/Segmentation_Collagene.git
cd Segmentation_Collagene
```
### 2) Create Conda environment (Python 3.9.16)

```bash
conda create -n segcoll python=3.9.16 -y
conda activate segcoll
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) PyTorch installation

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Load and store model pretrained weights

To run the inference script, you must first download the pre-trained weights.

```bash
mkdir -p model/weights
wget https://github.com/dmanou/Segmentation_Collagene/releases/download/v1.0.0/pretrained_weights.pth -O model/weights/pretrained_weights.pth
```
## Run inference

### Tile inference

### Whole Slide Image
