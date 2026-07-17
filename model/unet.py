import architecture_blocks as block
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()

        self.enc_1 = block.DoubleConv(in_channels, out_channels)
        self.enc_2 = block.Down(out_channels, out_channels * 2)
        self.enc_3 = block.Down(out_channels * 2, out_channels * 4)
        self.enc_4 = block.Down(out_channels * 4, out_channels * 8)
        self.enc_5 = block.Down(out_channels * 8, out_channels * 16)

        self.dec_4 = block.Up(out_channels * 8 * 2, out_channels * 8)
        self.dec_3 = block.Up(out_channels * 4 * 2, out_channels * 4)
        self.dec_2 = block.Up(out_channels * 2 * 2, out_channels * 2)
        self.dec_1 = block.Up(out_channels * 2, out_channels)

        self.logits = block.OutConv(out_channels, num_classes)
        self.aux2 = block.OutConv(out_channels * 4, num_classes)
        self.aux1 = block.OutConv(out_channels * 2, num_classes)

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
