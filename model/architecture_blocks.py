import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.reduce = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1 = decoder feature
        x2 = encoder skip feature
        """
        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        x1 = self.reduce(x1)

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
    

class OutConv(nn.Module):

    def __init__(self, in_channels, n_class):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_class, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
