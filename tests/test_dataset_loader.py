"""
Unit tests for BraTS-Africa dataset loader, transformations, and metrics.
"""

import os
import pytest
import torch
import numpy as np

from src.data.brats_dataset import BraTSDataset
from src.utils.metrics import compute_dice_score, evaluate_brats_subregions
from src.models.foundation_models import SemiSupervisedFoundationSegmentor


def test_metrics_dice_score():
    a = np.ones((10, 10))
    b = np.ones((10, 10))
    assert compute_dice_score(a, b) == 1.0

    c = np.zeros((10, 10))
    assert compute_dice_score(a, c) < 1e-4


def test_brats_subregion_evaluation():
    # 0: BG, 1: NCR, 2: ED, 3: ET
    gt = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3])
    res = evaluate_brats_subregions(pred, gt)
    assert res['Dice_WT'] == 1.0
    assert res['Dice_TC'] == 1.0
    assert res['Dice_ET'] == 1.0
    assert res['Dice_Mean'] == 1.0


def test_segmentor_forward_shape():
    model = SemiSupervisedFoundationSegmentor(in_channels=4, num_classes=4, base_filters=16)
    x = torch.randn(2, 4, 64, 64)
    out = model(x)
    assert out.shape == (2, 4, 64, 64)


def test_subregion_loss_and_augmentation():
    from src.data.transforms import SliceAugmentation2D
    from src.models.segmentation_heads import SubregionDiceCELoss
    from src.utils.metrics import postprocess_brats_prediction

    aug = SliceAugmentation2D(flip_prob=1.0, rot90_prob=0.0, intensity_prob=0.0, noise_prob=0.0)
    sample = {
        'image': torch.randn(12, 32, 32),
        'mask': torch.randint(0, 4, (32, 32)),
        'voxel_mask': torch.ones(32, 32),
    }
    out = aug(sample)
    assert out['image'].shape == (12, 32, 32)
    assert out['voxel_mask'].shape == (32, 32)

    loss_fn = SubregionDiceCELoss()
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 32, 32))
    vm = torch.ones(2, 32, 32)
    sw = torch.tensor([1.0, 0.8])
    loss = loss_fn(logits, targets, sample_weights=sw, voxel_mask=vm)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None

    # Tiny WT island (<50) disappears unless it's the only component; tiny ET (<25) converts to NCR (1)
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    pred[2:8, 2:8, 2:8] = 2  # 216 voxels main lesion
    pred[16:18, 16:18, 16:18] = 2  # 8 voxels noise island
    pred[4:6, 4:6, 4:6] = 3  # 8 voxels tiny ET speck
    cleaned = postprocess_brats_prediction(pred, min_wt_size=50, min_et_size=25)
    assert cleaned[16:18, 16:18, 16:18].sum() == 0
    assert (cleaned == 3).sum() == 0
    assert cleaned[4:6, 4:6, 4:6].min() == 1


def test_segresnet_and_25d_dataset_packing(tmp_path):
    from src.models.foundation_models import build_segmentor

    model = build_segmentor(arch="segresnet", in_channels=12, num_classes=4, base_filters=16)
    x = torch.randn(2, 12, 64, 64)
    out = model(x)
    assert out.shape == (2, 4, 64, 64)

    (tmp_path / "TEST_PAT").mkdir()
    ds = BraTSDataset(data_dir=str(tmp_path), is_labeled=True, mode='2.5d', cache_data=True)
    img_vol = np.random.randn(4, 32, 32, 16).astype(np.float32)
    seg_vol = np.zeros((32, 32, 16), dtype=np.int64)
    seg_vol[10:20, 10:20, 6:10] = 3
    entry = ds._pack_volume_entry(img_vol, seg_vol, "TEST_PAT")
    s1 = ds._sample_slice_from_entry(entry)
    assert s1['image'].shape == (12, 32, 32)
    assert s1['mask'].shape == (32, 32)

