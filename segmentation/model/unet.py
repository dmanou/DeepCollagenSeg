import torch
import torch.nn as nn

from model.blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """U-Net with two auxiliary output heads (deep supervision).

    Only `logits` is used at inference time; `aux1` / `aux2` are kept so
    that checkpoints produced by the mean-teacher training loop load
    without modification.
    """

    def __init__(self, in_channels=3, out_channels=64, num_classes=1):
        super().__init__()

        self.enc_1 = DoubleConv(in_channels, out_channels)
        self.enc_2 = Down(out_channels, out_channels * 2)
        self.enc_3 = Down(out_channels * 2, out_channels * 4)
        self.enc_4 = Down(out_channels * 4, out_channels * 8)
        self.enc_5 = Down(out_channels * 8, out_channels * 16)

        self.dec_4 = Up(out_channels * 8 * 2, out_channels * 8)
        self.dec_3 = Up(out_channels * 4 * 2, out_channels * 4)
        self.dec_2 = Up(out_channels * 2 * 2, out_channels * 2)
        self.dec_1 = Up(out_channels * 2, out_channels)

        self.logits = OutConv(out_channels, num_classes)
        self.aux2 = OutConv(out_channels * 4, num_classes)
        self.aux1 = OutConv(out_channels * 2, num_classes)

    def forward(self, x):
        # encode
        x0 = self.enc_1(x)
        x1 = self.enc_2(x0)
        x2 = self.enc_3(x1)
        x3 = self.enc_4(x2)
        x4 = self.enc_5(x3)

        # decode
        u3 = self.dec_4(x4, x3)
        u2 = self.dec_3(u3, x2)
        u1 = self.dec_2(u2, x1)
        u0 = self.dec_1(u1, x0)

        logits = self.logits(u0)
        aux1 = self.aux1(u1)
        aux2 = self.aux2(u2)

        return logits, aux1, aux2
