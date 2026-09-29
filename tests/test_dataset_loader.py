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
