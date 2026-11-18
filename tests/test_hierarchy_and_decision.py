"""
Unit tests for Anatomical Hierarchy, Reliability, Decision Engine, and Hierarchical Refiner.
"""

import pytest
import torch
import numpy as np

from src.semi_supervised.reliability import compute_entropy_map, compute_reliability_score, evaluate_subregion_reliabilities
from src.semi_supervised.hierarchy import compute_hierarchy_violation_index, enforce_algebraic_hierarchy
from src.semi_supervised.decision_engine import DecisionEngine, PseudoLabelAction
from src.semi_supervised.hierarchical_prompter import HierarchicalSAMRefiner
from src.semi_supervised.hierarchical_pipeline import HierarchicalPseudoLabelPipeline


def test_reliability_calculation():
    # Completely confident prediction: prob = 1.0 for true class
    probs = torch.zeros(1, 4, 10, 10, 10)
    probs[:, 1, :, :, :] = 1.0 # 100% confident class 1
    mask = torch.ones(1, 1, 10, 10, 10)

    score = compute_reliability_score(probs, mask)
    assert score > 0.99, f"Expected high reliability, got {score}"

    # Completely uncertain prediction: uniform distribution 0.25 across 4 classes
    probs_uniform = torch.full((1, 4, 10, 10, 10), 0.25)
    score_unif = compute_reliability_score(probs_uniform, mask)
    assert score_unif < 0.05, f"Expected near zero reliability for uniform entropy, got {score_unif}"


def test_hierarchy_violation_detection():
    # Case 1: Valid hierarchy (ET strictly inside TC, TC strictly inside WT)
    wt = np.zeros((20, 20, 20))
    tc = np.zeros((20, 20, 20))
    et = np.zeros((20, 20, 20))

    wt[2:18, 2:18, 2:18] = 1 # Largest
    tc[5:15, 5:15, 5:15] = 1 # Middle
    et[8:12, 8:12, 8:12] = 1 # Innermost

    stats_valid = compute_hierarchy_violation_index(wt, tc, et)
    assert stats_valid['is_valid'] == True
    assert stats_valid['hvi_et'] == 0.0
    assert stats_valid['hvi_tc'] == 0.0

    # Case 2: Violated hierarchy (ET leaks outside TC)
    et_violated = et.copy()
    et_violated[0:3, 0:3, 0:3] = 1 # Voxel outside TC and WT!

    stats_violated = compute_hierarchy_violation_index(wt, tc, et_violated)
    assert stats_violated['is_valid'] == False
    assert stats_violated['hvi_et'] > 0.0
    assert stats_violated['et_leak_mask'].sum() > 0


def test_decision_engine_actions():
    engine = DecisionEngine(tau_high=0.85, tau_low=0.50, delta_hvi=0.03)

    # 1. Test ACCEPT
    rel_high = {'reliability_mean': 0.92, 'reliability_WT': 0.95, 'reliability_TC': 0.90, 'reliability_ET': 0.91}
    hier_clean = {'hvi_et': 0.0, 'hvi_tc': 0.0, 'hvi_total': 0.0}
    dec_accept = engine.evaluate(rel_high, hier_clean, wt_voxel_count=1000)
    assert dec_accept['action'] == PseudoLabelAction.ACCEPT
    assert dec_accept['weight'] == 1.0

    # 2. Test REFINE due to hierarchy violation
    hier_violated = {'hvi_et': 0.15, 'hvi_tc': 0.0, 'hvi_total': 0.075}
    dec_refine_hier = engine.evaluate(rel_high, hier_violated, wt_voxel_count=1000)
    assert dec_refine_hier['action'] == PseudoLabelAction.REFINE
    assert dec_refine_hier['weight'] > 0.0

    # 3. Test REFINE due to moderate reliability
    rel_mod = {'reliability_mean': 0.70, 'reliability_WT': 0.75, 'reliability_TC': 0.68, 'reliability_ET': 0.67}
    dec_refine_rel = engine.evaluate(rel_mod, hier_clean, wt_voxel_count=1000)
    assert dec_refine_rel['action'] == PseudoLabelAction.REFINE

    # 4. Test REJECT due to low reliability
    rel_low = {'reliability_mean': 0.35, 'reliability_WT': 0.30, 'reliability_TC': 0.35, 'reliability_ET': 0.40}
    dec_reject = engine.evaluate(rel_low, hier_clean, wt_voxel_count=1000)
    assert dec_reject['action'] == PseudoLabelAction.REJECT
    assert dec_reject['weight'] == 0.0


def test_hierarchical_sam_refiner_guarantee():
    refiner = HierarchicalSAMRefiner()

    # Raw masks with artificial violations
    raw_wt = np.zeros((30, 30, 30))
    raw_tc = np.zeros((30, 30, 30))
    raw_et = np.zeros((30, 30, 30))

    raw_wt[5:25, 5:25, 5:25] = 1
    raw_tc[8:22, 8:22, 8:22] = 1
    raw_et[0:5, 0:5, 0:5] = 1 # Severely violating: ET outside WT!

    dummy_img = np.zeros((4, 30, 30, 30))
    refined = refiner.refine_hierarchical(dummy_img, raw_wt, raw_tc, raw_et)

    # After refinement, ET must be 100% inside TC, and TC 100% inside WT
    ref_wt = refined['refined_WT']
    ref_tc = refined['refined_TC']
    ref_et = refined['refined_ET']

    check = compute_hierarchy_violation_index(ref_wt, ref_tc, ref_et)
    assert check['is_valid'] == True, "Hierarchical refiner must guarantee 100% valid hierarchy"
    assert check['hvi_et'] == 0.0
    assert check['hvi_tc'] == 0.0

    # Verify BraTS multi-class mask labels
    assembled = refined['assembled_seg']
    unique_labels = np.unique(assembled)
    for lbl in unique_labels:
        assert lbl in [0, 1, 2, 3], f"Unexpected BraTS label {lbl}"


def test_end_to_end_pipeline():
    pipeline = HierarchicalPseudoLabelPipeline()

    dummy_img = torch.randn(4, 16, 16, 16)
    # Generate mock logits with high WT/TC/ET signal
    dummy_logits = torch.randn(4, 16, 16, 16)
    dummy_logits[1, 4:12, 4:12, 4:12] += 5.0 # NCR
    dummy_logits[2, 2:14, 2:14, 2:14] += 3.0 # ED
    dummy_logits[3, 6:10, 6:10, 6:10] += 6.0 # ET

    out = pipeline.process_unlabeled_sample(dummy_img, dummy_logits)
    assert out['action'] in ['ACCEPT', 'REFINE', 'REJECT']
    assert out['pseudo_label'].shape == (16, 16, 16)
    assert 'reliabilities' in out
    assert 'hierarchy_stats' in out



def test_hierarchical_sam_refiner_soft_blend_and_rgb_clipping():
    import numpy as np
    from src.semi_supervised.hierarchical_prompter import (
        HierarchicalSAMRefiner,
        build_sam_rgb_input,
    )

    # Normal brain MRI signal [1.0..2.0] with a single bright artifact voxel (500.0)
    img = np.zeros((4, 64, 64), dtype=np.float32)
    img[:, 8:56, 8:56] = 1.0
    img[:, 16:48, 16:48] = 2.0
    img[:, 8, 8] = 500.0

    rgb = build_sam_rgb_input(img)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0].sum() == 0
    assert rgb[32, 32].min() > 50

    refiner = HierarchicalSAMRefiner(use_sam=False, sam_version='medsam', device='cpu')
    refiner.has_sam = True
    refiner.medsam_model = object()
    refiner._encode_medsam_slice = lambda rgb_img: None
    refiner._predict_box_prob = lambda img_embed, box, hw: np.where(
        (np.arange(hw[0])[:, None] >= 20) & (np.arange(hw[0])[:, None] < 44)
        & (np.arange(hw[1])[None, :] >= 20) & (np.arange(hw[1])[None, :] < 44),
        0.90,
        0.05,
    ).astype(np.float32)

    raw_wt = np.zeros((64, 64), dtype=np.float32)
    raw_wt[22:42, 22:42] = 1.0
    raw_tc = np.zeros((64, 64), dtype=np.float32)
    raw_tc[24:40, 24:40] = 1.0
    raw_et = np.zeros((64, 64), dtype=np.float32)
    raw_et[28:36, 28:36] = 1.0

    wt_prob = np.full((64, 64), 0.10, dtype=np.float32)
    wt_prob[20:44, 20:44] = 0.40  # 0.65*0.40 + 0.35*0.90 = 0.575 >= 0.50 -> grows to 20:44
    tc_prob = np.full((64, 64), 0.10, dtype=np.float32)
    tc_prob[24:40, 24:40] = 0.85
    et_prob = np.full((64, 64), 0.10, dtype=np.float32)
    et_prob[20:22, 20:22] = 0.90  # deliberately outside TC/WT, must be hierarchy-pruned

    out = refiner.refine_hierarchical(
        image=img,
        raw_wt=raw_wt,
        raw_tc=raw_tc,
        raw_et=raw_et,
        raw_probs={'WT': wt_prob, 'TC': tc_prob, 'ET': et_prob},
        blend_alpha=0.65,
    )

    assert out['hierarchy_guaranteed'] is True
    wt_out = out['refined_WT'] > 0.5
    tc_out = out['refined_TC'] > 0.5
    et_out = out['refined_ET'] > 0.5
    assert wt_out[20, 20]
    assert np.all(np.logical_and(et_out, ~tc_out) == 0)
    assert np.all(np.logical_and(tc_out, ~wt_out) == 0)
    seg = out['assembled_seg']
    assert seg.shape == (64, 64)
    assert set(np.unique(seg).tolist()).issubset({0, 1, 2, 3})



def test_dynamic_semi_supervised_dataset_collation():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data.transforms import SliceAugmentation2D
    from src.models.foundation_models import build_segmentor
    from src.utils.training_helpers import DynamicSemiSupervisedDataset, build_loss_fn, train_one_epoch

    class DummyLabeledDS:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                'image': torch.randn(12, 32, 32, dtype=torch.float32),
                'mask': torch.randint(0, 4, (32, 32), dtype=torch.long),
                'patient_id': f'L_{idx}',
                'slice_idx': 10 + idx,
                'is_labeled': True,
            }

    pseudo_pool = [
        {
            'image_fp16': np.random.randn(12, 32, 32).astype(np.float16),
            'mask_u8': np.random.randint(0, 4, (32, 32), dtype=np.uint8),
            'voxel_mask_u8': np.ones((32, 32), dtype=np.uint8),
            'weight': 0.8,
            'score': 0.72,
            'patient_id': f'U_{i}',
            'slice_idx': 20 + i,
        }
        for i in range(4)
    ]

    ds = DynamicSemiSupervisedDataset(
        labeled_dataset=DummyLabeledDS(),
        slices_per_labeled_volume=2,
        pseudo_pool=pseudo_pool,
        transform=SliceAugmentation2D(),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert b['image'].shape == (4, 12, 32, 32)
        assert b['mask'].shape == (4, 32, 32)
        assert b['voxel_mask'].shape == (4, 32, 32)
        assert b['weight'].shape == (4,)
        assert len(b['slice_idx']) == 4
        assert len(b['is_labeled']) == 4

    model = build_segmentor('segresnet', in_channels=12, num_classes=4, base_filters=8)
    criterion = build_loss_fn('subregion_dice_ce')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_val = train_one_epoch(model, loader, criterion, opt, scaler=None, device='cpu', use_pbar=False)
    assert np.isfinite(loss_val) and loss_val > 0.0
