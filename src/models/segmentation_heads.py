"""
Segmentation loss functions and evaluation heads for semi-supervised training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Combined multi-class Dice and Cross-Entropy Loss for multi-modal brain tumor segmentation.
    """
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes, H, W]
            targets: [B, H, W] integer class labels
        """
        ce_loss = F.cross_entropy(logits, targets)

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        # Compute dice over foreground classes (1, 2, 3)
        for c in range(1, num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]
            intersection = (p * t).sum(dim=(1, 2))
            cardinality = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
            dice_loss += (1.0 - dice_c.mean())

        dice_loss = dice_loss / max(1, num_classes - 1)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
