"""
3D U-Net baseline for multi-class brain tumor segmentation.

Input: (B, C, D, H, W) where C = number of MRI modalities (T1, T1ce, T2, FLAIR = 4)
Output: (B, num_classes, D, H, W) — one channel per tumor sub-region + background
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers with instance norm + LeakyReLU, the standard block for medical 3D segmentation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling step: strided conv (learned downsampling beats maxpool for this task)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Upsampling step + skip connection concat + conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle off-by-one size mismatches from odd input dims
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Baseline 3D U-Net.

    Args:
        in_channels: number of input MRI modalities (default 4: T1, T1ce, T2, FLAIR)
        num_classes: number of output classes including background (default 4: bg, ET, ED, NCR/NET)
        base_channels: number of channels in the first encoder stage
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.enc2 = ConvBlock(c * 2, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.enc3 = ConvBlock(c * 4, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.enc4 = ConvBlock(c * 8, c * 8)

        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)

        self.out_conv = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)


if __name__ == "__main__":
    # quick smoke test
    model = UNet3D(in_channels=4, num_classes=4)
    x = torch.randn(1, 4, 64, 64, 64)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 4, 64, 64, 64)
    print("UNet3D smoke test passed.")
