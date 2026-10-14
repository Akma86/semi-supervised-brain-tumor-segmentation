"""
Anatomical hierarchy constraints and violation detection for brain tumor sub-regions.
Enforces the clinical pathological nesting rule:
    Enhancing Tumor (ET) ⊆ Tumor Core (TC) ⊆ Whole Tumor (WT)
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch


def compute_hierarchy_violation_index(
    wt_mask: Union[np.ndarray, torch.Tensor],
    tc_mask: Union[np.ndarray, torch.Tensor],
    et_mask: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6
) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
    """
    Quantifies violations of the nested tumor hierarchy:
        1. ET must be a subset of TC (ET ⊆ TC)
        2. TC must be a subset of WT (TC ⊆ WT)

    Args:
        wt_mask: Whole Tumor binary mask (0 or 1)
        tc_mask: Tumor Core binary mask (0 or 1)
        et_mask: Enhancing Tumor binary mask (0 or 1)
    Returns:
        Dictionary containing:
            - 'hvi_et': Fraction of ET voxels lying outside TC, in [0.0, 1.0]
            - 'hvi_tc': Fraction of TC voxels lying outside WT, in [0.0, 1.0]
            - 'hvi_total': Combined violation penalty
            - 'is_valid': True if hierarchy is strictly respected (HVI < 1e-4)
            - 'et_leak_mask': Binary map of ET voxels outside TC (violating voxels)
            - 'tc_leak_mask': Binary map of TC voxels outside WT (violating voxels)
    """
    is_torch = isinstance(wt_mask, torch.Tensor)

    if is_torch:
        wt_b = (wt_mask > 0.5)
        tc_b = (tc_mask > 0.5)
        et_b = (et_mask > 0.5)

        # ET leaking outside TC: ET and not TC
        et_leak = et_b & (~tc_b)
        et_total = et_b.sum().float().item()
        et_leak_count = et_leak.sum().float().item()
        hvi_et = (et_leak_count / (et_total + eps)) if et_total > 0 else 0.0

        # TC leaking outside WT: TC and not WT
        tc_leak = tc_b & (~wt_b)
        tc_total = tc_b.sum().float().item()
        tc_leak_count = tc_leak.sum().float().item()
        hvi_tc = (tc_leak_count / (tc_total + eps)) if tc_total > 0 else 0.0

        hvi_total = (hvi_et + hvi_tc) / 2.0
        is_valid = (hvi_total < 1e-4)

        return {
            'hvi_et': float(hvi_et),
            'hvi_tc': float(hvi_tc),
            'hvi_total': float(hvi_total),
            'is_valid': bool(is_valid),
            'et_leak_mask': et_leak,
            'tc_leak_mask': tc_leak
        }
    else:
        wt_b = (wt_mask > 0.5).astype(bool)
        tc_b = (tc_mask > 0.5).astype(bool)
        et_b = (et_mask > 0.5).astype(bool)

        et_leak = np.logical_and(et_b, ~tc_b)
        et_total = float(et_b.sum())
        hvi_et = (float(et_leak.sum()) / (et_total + eps)) if et_total > 0 else 0.0

        tc_leak = np.logical_and(tc_b, ~wt_b)
        tc_total = float(tc_b.sum())
        hvi_tc = (float(tc_leak.sum()) / (tc_total + eps)) if tc_total > 0 else 0.0

        hvi_total = (hvi_et + hvi_tc) / 2.0
        is_valid = (hvi_total < 1e-4)

        return {
            'hvi_et': float(hvi_et),
            'hvi_tc': float(hvi_tc),
            'hvi_total': float(hvi_total),
            'is_valid': bool(is_valid),
            'et_leak_mask': et_leak,
            'tc_leak_mask': tc_leak
        }


def enforce_algebraic_hierarchy(
    wt_mask: Union[np.ndarray, torch.Tensor],
    tc_mask: Union[np.ndarray, torch.Tensor],
    et_mask: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Direct algebraic projection onto the valid hierarchy manifold:
        TC_valid = TC ∩ WT
        ET_valid = ET ∩ TC_valid
    Used as an algebraic post-processing baseline.
    """
    if isinstance(wt_mask, torch.Tensor):
        wt_valid = (wt_mask > 0.5).float()
        tc_valid = (tc_mask > 0.5).float() * wt_valid
        et_valid = (et_mask > 0.5).float() * tc_valid
    else:
        wt_valid = (wt_mask > 0.5).astype(np.float32)
        tc_valid = (tc_mask > 0.5).astype(np.float32) * wt_valid
        et_valid = (et_mask > 0.5).astype(np.float32) * tc_valid

    return wt_valid, tc_valid, et_valid
