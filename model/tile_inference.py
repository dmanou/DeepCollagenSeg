import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import kornia.augmentation as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

import dataloader as dataloader
import torch.optim as optim

from unet import UNet
import utils

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

saving_path="/NAS/coolio/dorian/PROJETS/MENINGIOME_SEG/ANAPATH/model_v2/mean_teacher/training/weights/"
results_path="/NAS/coolio/dorian/PROJETS/MENINGIOME_SEG/ANAPATH/model_v2/mean_teacher/results/"

import os
os.makedirs(saving_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

## --------
## Loaders
## --------

ds = dataloader.UnLabeledDataset(tab_unlabel)
dataloader_ds = DataLoader(ds, batch_size = 16, shuffle = False, pin_memory=True, num_workers=8, persistent_workers=True)

teacher_model = UNet(in_channels=3, out_channels=64, num_classes=1).to(device)

checkpoints = torch.load(saving_path + "epochs_49.pth", map_location='cpu')
teacher_model.load_state_dict(checkpoints["model_state_dict"])

teacher_model.eval()

collagene_ratio = []

with torch.no_grad():
    for idx_batch, img in enumerate(tqdm(dataloader_ds)):
        # img, mask = batch[0], batch[1]
        img = img.to(device)
        # mask = mask.to(device)
        with torch.autocast("cuda"):
            x0 = teacher_model.enc_1(img)
            x1 = teacher_model.enc_2(x0)
            x2 = teacher_model.enc_3(x1)
            x3 = teacher_model.enc_4(x2)
            x4 = teacher_model.enc_5(x3)
            # decode
            u3 = teacher_model.dec_4(x4, x3)
            u2 = teacher_model.dec_3(u3, x2)
            u1 = teacher_model.dec_2(u2, x1)
            u0 = teacher_model.dec_1(u1, x0)

            logits = teacher_model.logits(u0)
            # logits, aux1, aux2 = teacher_model(img)
            mask = (torch.sigmoid(logits) > 0.5).float()
            mask_area =  mask.sum(dim=(1, 2, 3))

        collagene_ratio.append(mask_area.detach().cpu())

collagene_full = torch.cat(collagene_ratio, dim = 0).float().numpy()

tab_unlabel['collagene'] = collagene_full

tab_unlabel.to_csv(f"{saving_path}tab_with_collagene.csv", index = False)
