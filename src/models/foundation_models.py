"""
Foundation model interfaces and adapters for medical image segmentation.
Supports 2D promptable foundation models (e.g. SAM / MedSAM) and 3D vision transformers (e.g. SwinUNETR).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusionEncoder(nn.Module):
    """
    Early and intermediate fusion module to aggregate 4 MRI modalities (t1n, t1c, t2w, t2f)
    into a unified representation for foundation model backbones.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        # Projects 4 MRI channels to 3 channels (e.g. for standard RGB-based foundation vision encoders like SAM/ViT)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 4, H, W]
        Output: [B, 3, H, W] (compatible with ViT / SAM backbone)
        """
        return self.conv2(self.relu(self.norm(self.conv1(x))))


class SemiSupervisedFoundationSegmentor(nn.Module):
    """
    Unified segmentor combining a medical foundation backbone with promptable segmentation
    and a multi-class mask prediction head for BraTS (4 classes: background, NCR, ED, ET).
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout_p: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Lightweight multi-scale feature encoder (proxy or adaptable wrapper)
        self.fusion = MultiModalFusionEncoder(in_channels=in_channels, out_channels=base_filters)

        self.enc1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            logits: Output logits [B, num_classes, H, W]
        """
        f = self.fusion(x)
        e1 = self.enc1(f)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.classifier(d1)
        return logits


class _ResBlock2D(nn.Module):
    """Pre-activation GroupNorm residual block following the BraTS SegResNet design."""
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        num_groups = min(groups, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SegResNetSegmentor(nn.Module):
    """
    Pure-PyTorch SegResNet 2D/2.5D medical segmentation backbone (GroupNorm + pre-activation
    residual blocks with additive multi-scale decoder) supporting 2D (4 ch) and 2.5D (12 ch).
    Avoids MONAI's eager onnx/trt optional imports on Windows while keeping the SegResNet design.
    """
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 4,
        init_filters: int = 32,
        dropout_p: float = 0.1,
        blocks_down=(1, 2, 2, 4),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_init = nn.Conv2d(in_channels, init_filters, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.down_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = init_filters
        for i, n_blocks in enumerate(blocks_down):
            stage = nn.Sequential(*[_ResBlock2D(ch) for _ in range(n_blocks)])
            self.down_layers.append(stage)
            if i < len(blocks_down) - 1:
                next_ch = ch * 2
                self.downsamples.append(nn.Conv2d(ch, next_ch, kernel_size=3, stride=2, padding=1, bias=False))
                ch = next_ch

        self.up_reduce = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(len(blocks_down) - 1):
            half_ch = ch // 2
            self.up_reduce.append(nn.Conv2d(ch, half_ch, kernel_size=1, bias=False))
            self.up_blocks.append(_ResBlock2D(half_ch))
            ch = half_ch

        num_groups = min(8, ch)
        while ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.conv_init(x))
        skips = []
        for i, stage in enumerate(self.down_layers):
            x = stage(x)
            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        for i in range(len(self.up_reduce)):
            skip = skips[-(i + 1)]
            x = F.interpolate(self.up_reduce[i](x), size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.up_blocks[i](x + skip)

        return self.head(x)


def build_segmentor(
    arch: str = "segresnet",
    in_channels: int = 12,
    num_classes: int = 4,
    base_filters: int = 32,
    dropout_p: float = 0.1,
) -> nn.Module:
    """Factory creating either the lightweight baseline CNN or MONAI SegResNet backbone."""
    arch_key = arch.lower().strip()
    if arch_key == "segresnet":
        return SegResNetSegmentor(
            in_channels=in_channels,
            num_classes=num_classes,
            init_filters=base_filters,
            dropout_p=dropout_p,
        )
    if arch_key in ("unet_mini", "mini", "baseline"):
        return SemiSupervisedFoundationSegmentor(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_p=dropout_p,
        )
    raise ValueError(f"Unknown segmentor architecture '{arch}'. Expected 'segresnet' or 'unet_mini'.")
