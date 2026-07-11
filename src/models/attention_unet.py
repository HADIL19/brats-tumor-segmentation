"""
3D Attention U-Net.

Adds attention gates at each skip connection so the decoder learns to weight
encoder features by relevance instead of passing all of them through uniformly.
This matters for tumor segmentation because tumor sub-regions occupy a small
fraction of the volume — attention helps suppress irrelevant background signal.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," 2018.
"""

import torch
import torch.nn as nn

from src.models.unet import ConvBlock, Down


class AttentionGate(nn.Module):
    """
    Learns a spatial attention map over encoder (skip) features, gated by
    the decoder's coarser semantic signal.

    gate_channels: channels in the decoder feature map (coarser, more semantic)
    skip_channels: channels in the encoder feature map (finer, being gated)
    inter_channels: bottleneck size for the gating computation
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = nn.functional.interpolate(g, size=s.shape[2:], mode="trilinear", align_corners=False)
        attention = self.psi(self.relu(g + s))
        return skip * attention


class AttentionUp(nn.Module):
    """Upsample, apply attention gate to the skip connection, concat, conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(gate_channels=out_channels, skip_channels=skip_channels,
                                   inter_channels=skip_channels // 2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        gated_skip = self.attn(gate=x, skip=skip)
        x = torch.cat([x, gated_skip], dim=1)
        return self.conv(x)


class AttentionUNet3D(nn.Module):
    """Same encoder as UNet3D; decoder uses attention-gated skip connections."""

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

        self.up3 = AttentionUp(c * 8, c * 4, c * 4)
        self.up2 = AttentionUp(c * 4, c * 2, c * 2)
        self.up1 = AttentionUp(c * 2, c, c)

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
    model = AttentionUNet3D(in_channels=4, num_classes=4)
    x = torch.randn(1, 4, 64, 64, 64)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 4, 64, 64, 64)
    print("AttentionUNet3D smoke test passed.")
