"""
Segmentation loss functions and evaluation heads for semi-supervised training.
Supports sample-level weighting for pseudo-label confidence control.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Combined multi-class Dice and Cross-Entropy Loss for multi-modal brain tumor segmentation.
    Supports sample-level weighting so pseudo-labels are weighted individually per sample.
    """
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes, H, W]
            targets: [B, H, W] integer class labels
            sample_weights: Optional [B] or [B, 1, 1] sample-wise weights
        """
        B = logits.shape[0]

        # 1. Cross-entropy computed individually per sample [B]
        ce_loss_raw = F.cross_entropy(logits, targets, reduction='none') # [B, H, W]
        ce_per_sample = ce_loss_raw.mean(dim=(1, 2))                     # [B]

        # 2. Multi-class Dice computed individually per sample [B]
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dice_per_sample = torch.zeros(B, device=logits.device)
        for c in range(1, num_classes):
            p = probs[:, c] # [B, H, W]
            t = targets_onehot[:, c] # [B, H, W]
            intersection = (p * t).sum(dim=(1, 2))
            cardinality = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
            dice_per_sample += (1.0 - dice_c)

        dice_per_sample = dice_per_sample / max(1, num_classes - 1) # [B]

        # Combined per-sample loss
        loss_per_sample = self.ce_weight * ce_per_sample + self.dice_weight * dice_per_sample # [B]

        # 3. Apply sample-specific weights (labeled=1.0, ACCEPT=1.0, REFINE=0.8)
        if sample_weights is not None:
            w = sample_weights.view(B).to(logits.device)
            return (loss_per_sample * w).sum() / (w.sum() + self.eps)

        return loss_per_sample.mean()


class SubregionDiceCELoss(nn.Module):
    """
    Directly optimizes BraTS nested evaluation regions (WT, TC, ET) with batch-level Soft Dice
    combined with cross-entropy, supporting per-sample weights and optional voxel confidence masks.
    """
    def __init__(self, ce_weight: float = 0.7, dice_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def _soft_dice_loss(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor,
    ) -> torch.Tensor:
        inter = (prob * target * weight_map).sum()
        denom = ((prob + target) * weight_map).sum()
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        voxel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = logits.shape[0]
        device = logits.device

        if sample_weights is None:
            sw = torch.ones((B,), device=device, dtype=logits.dtype)
        else:
            sw = sample_weights.view(B).to(device=device, dtype=logits.dtype)

        if voxel_mask is None:
            vm = torch.ones_like(targets, device=device, dtype=logits.dtype)
        else:
            vm = voxel_mask.to(device=device, dtype=logits.dtype)

        # Masked weighted cross-entropy per sample
        ce_map = F.cross_entropy(logits, targets, reduction='none') * vm
        ce_per_sample = ce_map.flatten(1).sum(dim=1) / vm.flatten(1).sum(dim=1).clamp_min(1.0)
        ce_loss = (ce_per_sample * sw).sum() / sw.sum().clamp_min(self.eps)

        # Compute Soft Dice in float32 for numerical stability under AMP
        probs = F.softmax(logits.float(), dim=1)
        weight_map = (vm * sw.view(B, 1, 1)).float()

        p_wt = probs[:, 1] + probs[:, 2] + probs[:, 3]
        p_tc = probs[:, 1] + probs[:, 3]
        p_et = probs[:, 3]

        g_wt = (targets > 0).float()
        g_tc = ((targets == 1) | (targets == 3)).float()
        g_et = (targets == 3).float()

        loss_wt = self._soft_dice_loss(p_wt, g_wt, weight_map)
        loss_tc = self._soft_dice_loss(p_tc, g_tc, weight_map)
        loss_et = self._soft_dice_loss(p_et, g_et, weight_map)
        dice_loss = (loss_wt + 1.2 * loss_tc + 1.2 * loss_et) / 3.4

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
