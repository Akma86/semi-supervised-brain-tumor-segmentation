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
