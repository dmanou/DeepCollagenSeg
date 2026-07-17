##
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

##
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

##
from unet import UNet
from tiatoolbox import ...


##########################################################
# Arguments
##########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--wsi_list", type=list, help="wsi to segment")
parser.add_argument("--preprocess_wsi", type="store_true", help="Preprocess WSI")
parser.add_argument("--resize_tiles", action="store_true", help="tile resolution is initially 0.25mpp with 1024x1024 resized to 0.5 mpp 512x512")
parser.add_argument("--sd", type=str, help="Output directory where model outputs will be saved")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration (default: cpu)")
parser.add_argument("--batchsize", action=int, help="bacth size (default 8)")

args = parser.parse_args()
device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs(saving_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

##########################################################
# Loaders
##########################################################

class TileDataset(Dataset):
    def __init__(self, tile_list, target_size=512):
        self.tile_list = tile_list
        self.to_image = v2.ToImage()
        self.target_size = (target_size, target_size)

    def __len__(self):
        return len(self.tile_list)

    def resize(self, patch):
        patch_512 = cv2.resize(
            patch,
            self.target_size,
            interpolation=cv2.INTER_AREA
        )

        return patch_512

    def __getitem__(self, idx):

        path = self.tile_list[idx]
        tile = Image.open(path)
        tile = self.resize(np.array(tile))
        tile = self.to_image(tile).float() / 255.0

        return tile

dataset = TileDataset(tile_list = args.tile_list)
loader = DataLoader(dataset, 
                    batch_size = args.batchsize, 
                    shuffle = False, 
                    pin_memory=True, 
                    num_workers=8, 
                    persistent_workers=True)

##########################################################
# Model
##########################################################

pretrained_weights = "weights/pretrained_weights.pth"

teacher_model = UNet(
    in_channels=3,
    out_channels=64,
    num_classes=1,
).to(device)

checkpoint = torch.load(
    pretrained_weights,
    map_location="cpu"
)

teacher_model.load_state_dict(
    checkpoint["model_state_dict"]
)

teacher_model.eval()

def predict_logits(model, x):
    """Forward pass of the UNet."""

    x0 = model.enc_1(x)
    x1 = model.enc_2(x0)
    x2 = model.enc_3(x1)
    x3 = model.enc_4(x2)
    x4 = model.enc_5(x3)

    u3 = model.dec_4(x4, x3)
    u2 = model.dec_3(u3, x2)
    u1 = model.dec_2(u2, x1)
    u0 = model.dec_1(u1, x0)

    return model.logits(u0)

##########################################################
# Inference
##########################################################

with torch.no_grad():
    for tile in tqdm(loader):
        img = img.to(device)

        with torch.autocast("cuda"):
            logits = predict_logits(teacher_model, img)
            
        mask = (torch.sigmoid(logits) > 0.5).float().detach().numpy().cpu()

    mask.save(....)
