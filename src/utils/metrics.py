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


def _tight_nonzero_slices(mask: np.ndarray) -> tuple:
    """Computes tight per-axis slice bounds for a non-empty 2D/3D boolean mask without argwhere."""
    slices = []
    for axis in range(mask.ndim):
        reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
        proj = mask.any(axis=reduce_axes)
        idx = np.flatnonzero(proj)
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(slices)


def postprocess_brats_prediction(
    pred_seg: np.ndarray,
    min_wt_size: int = 50,
    min_et_size: int = 25,
) -> np.ndarray:
    """
    Removes isolated false-positive islands in Whole Tumor and suppresses tiny
    Enhancing Tumor specks (converting them to NCR) to protect empty-ET scans.
    """
    from scipy.ndimage import label as cc_label

    cleaned = pred_seg.copy()
    wt_mask = cleaned > 0
    if not wt_mask.any():
        return cleaned

    bbox = _tight_nonzero_slices(wt_mask)
    wt_sub = wt_mask[bbox]
    struct = np.ones((3,) * cleaned.ndim, dtype=np.int32)
    labeled_wt, num_wt = cc_label(wt_sub, structure=struct)
    if num_wt > 0:
        counts = np.bincount(labeled_wt.ravel())
        keep_ids = np.where(counts[1:] >= min_wt_size)[0] + 1
        if len(keep_ids) == 0:
            keep_ids = np.array([int(np.argmax(counts[1:])) + 1])
        keep_mask = np.isin(labeled_wt, keep_ids)
        sub_clean = cleaned[bbox]
        sub_clean[~keep_mask] = 0
        cleaned[bbox] = sub_clean

    et_mask = cleaned == 3
    total_et = int(et_mask.sum())
    if 0 < total_et < min_et_size:
        cleaned[et_mask] = 1

    return cleaned


def predict_probs_tta(
    model,
    imgs: "torch.Tensor",
    use_tta: bool = True,
    use_amp: bool = True,
) -> "torch.Tensor":
    """Runs forward inference with optional 4-view flip Test-Time Augmentation."""
    import torch
    import torch.nn.functional as F

    def _forward_probs(x: torch.Tensor) -> torch.Tensor:
        if use_amp and x.is_cuda:
            with torch.amp.autocast('cuda'):
                logits = model(x)
        else:
            logits = model(x)
        return F.softmax(logits.float(), dim=1)

    probs = _forward_probs(imgs)
    if not use_tta:
        return probs

    probs = probs + torch.flip(_forward_probs(torch.flip(imgs, dims=[3])), dims=[3])
    probs = probs + torch.flip(_forward_probs(torch.flip(imgs, dims=[2])), dims=[2])
    probs = probs + torch.flip(_forward_probs(torch.flip(imgs, dims=[2, 3])), dims=[2, 3])
    return probs / 4.0


def evaluate_dataset_3d(
    model,
    dataset,
    device,
    slice_mode: str = "2.5d",
    batch_size: int = 32,
    use_tta: bool = True,
    postprocess: bool = True,
    slice_stride: int = 1,
) -> Dict[str, float]:
    """Evaluates full 3D patient-level BraTS Dice (WT, TC, ET, Mean) over a dataset."""
    import torch

    model.eval()
    dices_wt, dices_tc, dices_et, dices_mean = [], [], [], []
    use_amp = (device.type == 'cuda')
    stride = max(1, int(slice_stride))

    if use_amp and torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for idx in range(len(dataset)):
            entry = dataset.get_patient_volume_entry(idx)
            H, W, D = entry['orig_shape']
            h0, h1, w0, w1, d0, d1 = entry['bbox']

            pred_vol = np.zeros((H, W, D), dtype=np.uint8)
            brain_slices = entry['brain_slices'][::stride]

            for start in range(0, len(brain_slices), batch_size):
                chunk_z = brain_slices[start:start + batch_size]
                imgs_np = dataset.extract_slices_batch(entry, chunk_z, slice_mode=slice_mode)
                imgs_t = torch.from_numpy(imgs_np).to(device=device, dtype=torch.float32)
                probs = predict_probs_tta(model, imgs_t, use_tta=use_tta, use_amp=use_amp)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                for off in range(stride):
                    z_fill = np.clip(chunk_z + off, 0, D - 1)
                    pred_vol[:, :, z_fill] = np.transpose(preds, (1, 2, 0))

            if postprocess:
                pred_vol = postprocess_brats_prediction(pred_vol)

            gt_vol = np.zeros((H, W, D), dtype=np.uint8)
            gt_vol[h0:h1, w0:w1, d0:d1] = entry['seg_crop']

            # Crop metric evaluation to union of predicted/GT tumor voxels (exact mathematical match)
            union = (pred_vol > 0) | (gt_vol > 0)
            if union.any():
                ubox = _tight_nonzero_slices(union)
                m = evaluate_brats_subregions(pred_vol[ubox], gt_vol[ubox])
            else:
                m = {'Dice_WT': 1.0, 'Dice_TC': 1.0, 'Dice_ET': 1.0, 'Dice_Mean': 1.0}
            dices_wt.append(m['Dice_WT'])
            dices_tc.append(m['Dice_TC'])
            dices_et.append(m['Dice_ET'])
            dices_mean.append(m['Dice_Mean'])

    return {
        'Dice_WT': float(np.mean(dices_wt)),
        'Dice_TC': float(np.mean(dices_tc)),
        'Dice_ET': float(np.mean(dices_et)),
        'Dice_Mean': float(np.mean(dices_mean)),
    }
