"""
Shared training, dataset setup, optimization, and semi-supervised pool helpers.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.data.brats_dataset import BraTSDataset, BraTSMultiSliceDataset
from src.data.transforms import SliceAugmentation2D
from src.models.segmentation_heads import DiceCELoss, SubregionDiceCELoss
from src.utils.metrics import evaluate_dataset_3d


def resolve_dataset_config(cfg: dict, splits_override: Optional[str] = None) -> Tuple[List[str], str, List[str]]:
    """Resolves cohort directories, splits JSON path, and MRI modalities from config."""
    ds_cfg = cfg['dataset']
    root_dir = ds_cfg['root_dir']
    cohorts = ds_cfg.get('cohorts', ['95_Glioma', '51_OtherNeoplasms'])
    cohort_dirs = [os.path.join(root_dir, c) for c in cohorts]
    splits_path = splits_override or ds_cfg.get('splits_file', 'data/splits.json')
    modalities = ds_cfg.get('modalities', ['t1c', 't1n', 't2f', 't2w'])
    return cohort_dirs, splits_path, modalities


def build_loss_fn(loss_name: str = "subregion_dice_ce") -> nn.Module:
    """Builds segmentation loss supporting both labeled and weighted semi-supervised batches."""
    key = loss_name.lower().strip()
    if key == "subregion_dice_ce":
        return SubregionDiceCELoss(ce_weight=0.7, dice_weight=1.0)
    if key == "dice_ce":
        return DiceCELoss(ce_weight=1.0, dice_weight=1.0)
    raise ValueError(f"Unknown loss '{loss_name}'. Expected 'subregion_dice_ce' or 'dice_ce'.")


def compute_loss_compat(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    voxel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatches optional voxel_mask to losses that support entropy confidence masking."""
    if isinstance(criterion, SubregionDiceCELoss):
        return criterion(logits, targets, sample_weights=sample_weights, voxel_mask=voxel_mask)
    return criterion(logits, targets, sample_weights=sample_weights)


def build_train_transform(augment: bool = True) -> Optional[Callable]:
    return SliceAugmentation2D() if augment else None


def wrap_multi_slice_dataset(base_dataset: BraTSDataset, slices_per_volume: int) -> Dataset:
    if slices_per_volume > 1 and base_dataset.mode in ('2d', '2.5d'):
        return BraTSMultiSliceDataset(base_dataset, slices_per_volume=slices_per_volume)
    return base_dataset


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: str,
    desc: str = "Train",
    use_pbar: bool = True,
) -> float:
    """Runs one full training epoch with AMP support and optional sample/voxel weights."""
    from tqdm import tqdm

    model.train()
    epoch_loss = 0.0
    num_batches = max(1, len(loader))
    iterator = tqdm(loader, desc=desc, unit="batch") if use_pbar else loader

    for batch in iterator:
        imgs = batch['image'].to(device).float()
        masks = batch['mask'].to(device).long()
        weights = batch.get('weight')
        if weights is not None:
            weights = weights.to(device).float()
        voxel_mask = batch.get('voxel_mask')
        if voxel_mask is not None:
            voxel_mask = voxel_mask.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = compute_loss_compat(criterion, logits, masks, sample_weights=weights, voxel_mask=voxel_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = compute_loss_compat(criterion, logits, masks, sample_weights=weights, voxel_mask=voxel_mask)
            loss.backward()
            optimizer.step()

        epoch_loss += float(loss.item())
        if use_pbar:
            iterator.set_postfix({'Loss': f"{loss.item():.4f}"})

    return epoch_loss / num_batches


def evaluate_and_checkpoint(
    model: nn.Module,
    val_dataset: BraTSDataset,
    device: str,
    slice_mode: str,
    val_batch_size: int,
    use_tta: bool,
    postprocess: bool,
    best_val_dice: float,
    best_metrics: Dict[str, float],
    save_path: str,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    prefix: str = "Val 3D",
    slice_stride: int = 1,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Runs full 3D volumetric validation and persists the best checkpoint."""
    dev = torch.device(device)
    metrics = evaluate_dataset_3d(
        model=model,
        dataset=val_dataset,
        device=dev,
        slice_mode=slice_mode,
        batch_size=val_batch_size,
        use_tta=use_tta,
        postprocess=postprocess,
        slice_stride=slice_stride,
    )
    print(
        f"  [{prefix} @ Epoch {epoch:02d}] "
        f"WT: {metrics['Dice_WT']*100:.2f}% | "
        f"TC: {metrics['Dice_TC']*100:.2f}% | "
        f"ET: {metrics['Dice_ET']*100:.2f}% | "
        f"Mean: {metrics['Dice_Mean']*100:.2f}%"
    )

    if metrics['Dice_Mean'] > best_val_dice:
        best_val_dice = metrics['Dice_Mean']
        best_metrics = {
            'WT': metrics['Dice_WT'],
            'TC': metrics['Dice_TC'],
            'ET': metrics['Dice_ET'],
        }
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_dice': best_val_dice,
            'metrics': best_metrics,
        }, save_path)
        print(f"  >> Saved new best checkpoint ({best_val_dice*100:.2f}%) -> {save_path}")

    return best_val_dice, best_metrics, metrics


def finalize_val_and_test(
    model: nn.Module,
    save_path: str,
    val_dataset: BraTSDataset,
    cohort_dirs: List[str],
    modalities: List[str],
    splits: dict,
    device: str,
    slice_mode: str,
    use_tta: bool,
    val_tta: bool,
    postprocess: bool,
    best_val_dice: float,
    best_metrics: Dict[str, float],
    best_epoch: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Loads the best checkpoint to run final TTA validation and held-out test evaluation."""
    final_val = {
        'WT': float(best_metrics['WT']),
        'TC': float(best_metrics['TC']),
        'ET': float(best_metrics['ET']),
        'Mean': float(best_val_dice),
    }
    test_metrics: Dict[str, float] = {}
    dev = torch.device(device)

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if use_tta and not val_tta:
            print(f"\n>>> Final 4-view TTA 3D validation on best checkpoint (Epoch {best_epoch})...")
            m_tta = evaluate_dataset_3d(
                model=model,
                dataset=val_dataset,
                device=dev,
                slice_mode=slice_mode,
                batch_size=16,
                use_tta=True,
                postprocess=postprocess,
            )
            final_val = {
                'WT': float(m_tta['Dice_WT']),
                'TC': float(m_tta['Dice_TC']),
                'ET': float(m_tta['Dice_ET']),
                'Mean': float(m_tta['Dice_Mean']),
            }
            print(
                f"  [Val 3D + TTA] WT: {final_val['WT']*100:.2f}% | TC: {final_val['TC']*100:.2f}% | "
                f"ET: {final_val['ET']*100:.2f}% | Mean: {final_val['Mean']*100:.2f}%"
            )

        test_ids = splits.get('test', [])
        if test_ids:
            test_dataset = BraTSDataset(
                data_dir=cohort_dirs,
                modalities=modalities,
                is_labeled=True,
                patient_ids=test_ids,
                mode=slice_mode,
                cache_data=False,
            )
            m_test = evaluate_dataset_3d(
                model=model,
                dataset=test_dataset,
                device=dev,
                slice_mode=slice_mode,
                batch_size=16,
                use_tta=use_tta,
                postprocess=postprocess,
            )
            test_metrics = {
                'WT': float(m_test['Dice_WT']),
                'TC': float(m_test['Dice_TC']),
                'ET': float(m_test['Dice_ET']),
                'Mean': float(m_test['Dice_Mean']),
            }
            print(
                f"  [Held-Out Test 3D] WT: {test_metrics['WT']*100:.2f}% | TC: {test_metrics['TC']*100:.2f}% | "
                f"ET: {test_metrics['ET']*100:.2f}% | Mean: {test_metrics['Mean']*100:.2f}%"
            )

    return final_val, test_metrics


class DynamicSemiSupervisedDataset(Dataset):
    """
    Combines dynamic slice sampling over labeled 3D volumes with memory-efficient
    half-precision pseudo-labeled 2D/2.5D slices and voxel-level confidence masks.
    """
    def __init__(
        self,
        labeled_dataset: BraTSDataset,
        slices_per_labeled_volume: int,
        pseudo_pool: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
    ):
        self.labeled_dataset = labeled_dataset
        self.slices_per_labeled = max(1, int(slices_per_labeled_volume))
        self.num_labeled_items = len(labeled_dataset) * self.slices_per_labeled
        self.pseudo_pool = pseudo_pool
        self.transform = transform

    def __len__(self) -> int:
        return self.num_labeled_items + len(self.pseudo_pool)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.num_labeled_items:
            sample = self.labeled_dataset[idx % len(self.labeled_dataset)]
            sample['weight'] = torch.tensor(1.0, dtype=torch.float32)
            sample['voxel_mask'] = torch.ones_like(sample['mask'], dtype=torch.float32)
            sample['slice_idx'] = int(sample.get('slice_idx', 0))
            sample['is_labeled'] = bool(sample.get('is_labeled', True))
            return sample

        p_item = self.pseudo_pool[idx - self.num_labeled_items]
        sample = {
            'image': torch.from_numpy(p_item['image_fp16'].astype(np.float32)),
            'mask': torch.from_numpy(p_item['mask_u8'].astype(np.int64)),
            'voxel_mask': torch.from_numpy(p_item['voxel_mask_u8'].astype(np.float32)),
            'weight': torch.tensor(float(p_item['weight']), dtype=torch.float32),
            'patient_id': p_item['patient_id'],
            'slice_idx': int(p_item.get('slice_idx', 0)),
            'is_labeled': False,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

