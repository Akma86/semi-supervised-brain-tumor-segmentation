"""
Evaluation metrics for Brain Tumor Segmentation (BraTS standard):
Dice Similarity Coefficient (DSC) and 95% Hausdorff Distance (HD95).
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def compute_dice_score(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], eps: float = 1e-6) -> float:
    """
    Computes binary Dice Similarity Coefficient (DSC).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_bool = (pred > 0.5).astype(bool)
    target_bool = (target > 0.5).astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    denominator = pred_bool.sum() + target_bool.sum()

    if denominator == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection + eps) / (denominator + eps))


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Approximated 95% Hausdorff Distance between two binary masks.
    Returns 0.0 if both masks are empty, or 373.1 (maximum diagonal) if one is empty.
    """
    pred_pts = np.argwhere(pred > 0.5)
    target_pts = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 and len(target_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 373.1 # Standard BraTS penalty constant for empty prediction

    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return float(max(d1, d2))


def evaluate_brats_subregions(
    pred_seg: Union[np.ndarray, torch.Tensor],
    gt_seg: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluates Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
    in compliance with BraTS challenge guidelines.

    BraTS Labels:
    - 1: Necrotic / Non-enhancing Core (NCR)
    - 2: Edema (ED)
    - 3: Enhancing Tumor (ET)
    """
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.detach().cpu().numpy()
    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.detach().cpu().numpy()

    # Binary representations
    pred_wt = (pred_seg == 1) | (pred_seg == 2) | (pred_seg == 3)
    gt_wt = (gt_seg == 1) | (gt_seg == 2) | (gt_seg == 3)

    pred_tc = (pred_seg == 1) | (pred_seg == 3)
    gt_tc = (gt_seg == 1) | (gt_seg == 3)

    pred_et = (pred_seg == 3)
    gt_et = (gt_seg == 3)

    dice_wt = compute_dice_score(pred_wt, gt_wt)
    dice_tc = compute_dice_score(pred_tc, gt_tc)
    dice_et = compute_dice_score(pred_et, gt_et)
    mean_dice = (dice_wt + dice_tc + dice_et) / 3.0

    return {
        'Dice_WT': dice_wt,
        'Dice_TC': dice_tc,
        'Dice_ET': dice_et,
        'Dice_Mean': mean_dice
    }
