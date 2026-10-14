"""
Reliability and uncertainty estimation for semi-supervised brain tumor segmentation.
Computes voxel-wise and regional confidence metrics for Whole Tumor (WT),
Tumor Core (TC), and Enhancing Tumor (ET).
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_entropy_map(probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Shannon entropy across classes for each voxel.
    Normalized to range [0, 1] by dividing by log(num_classes).

    Args:
        probs: Tensor of shape [B, C, H, W] or [B, C, H, W, D] with softmax probabilities.
    Returns:
        Normalized entropy map in [0, 1] where 0 = maximum certainty, 1 = maximum ambiguity.
    """
    num_classes = probs.shape[1]
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1, keepdim=True)
    norm_entropy = entropy / np.log(num_classes)
    return norm_entropy.clamp(0.0, 1.0)


def compute_reliability_score(
    probs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """
    Computes average reliability R in [0, 1] over a region of interest.
    R = 1.0 - mean(normalized_entropy) within the predicted mask.

    Args:
        probs: Probability tensor [B, C, ...]
        mask: Binary foreground mask [B, 1, ...]
    Returns:
        Scalar reliability score in [0.0, 1.0]
    """
    entropy = compute_entropy_map(probs)
    if mask.dim() == entropy.dim() - 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() == entropy.dim() + 1 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    fg_mask = (mask > 0.5)

    if fg_mask.sum() == 0:
        return 0.0

    mean_fg_entropy = entropy[fg_mask].mean().item()
    reliability = 1.0 - mean_fg_entropy
    return float(np.clip(reliability, 0.0, 1.0))


def compute_margin_score(probs: torch.Tensor) -> torch.Tensor:
    """
    Computes prediction margin: the difference between top-1 and top-2 predicted probabilities.
    High margin indicates decisive, reliable predictions; low margin indicates boundary uncertainty.
    """
    top2_vals, _ = torch.topk(probs, k=2, dim=1)
    margin = top2_vals[:, 0:1] - top2_vals[:, 1:2]
    return margin


def evaluate_subregion_reliabilities(
    logits: torch.Tensor,
    wt_mask: torch.Tensor,
    tc_mask: torch.Tensor,
    et_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Computes individual reliability metrics for WT, TC, and ET sub-regions.

    Args:
        logits: Raw network output logits [B, C, ...]
        wt_mask: Whole Tumor binary mask
        tc_mask: Tumor Core binary mask
        et_mask: Enhancing Tumor binary mask
    Returns:
        Dictionary of reliability scores for 'WT', 'TC', and 'ET'.
    """
    probs = F.softmax(logits, dim=1)

    r_wt = compute_reliability_score(probs, wt_mask)
    r_tc = compute_reliability_score(probs, tc_mask)
    r_et = compute_reliability_score(probs, et_mask)

    return {
        'reliability_WT': r_wt,
        'reliability_TC': r_tc,
        'reliability_ET': r_et,
        'reliability_mean': (r_wt + r_tc + r_et) / 3.0
    }
