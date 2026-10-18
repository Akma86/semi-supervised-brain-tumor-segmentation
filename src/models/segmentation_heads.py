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
