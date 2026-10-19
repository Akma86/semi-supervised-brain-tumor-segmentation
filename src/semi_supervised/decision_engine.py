"""
Decision Engine for Tri-Way Pseudo-Label Filtering:
Categorizes candidate segmentation masks into ACCEPT, REFINE, or REJECT
based on uncertainty reliability and anatomical hierarchy compliance.
"""

from enum import Enum
from typing import Dict, Any, Optional
import torch


class PseudoLabelAction(str, Enum):
    ACCEPT = "ACCEPT"
    REFINE = "REFINE"
    REJECT = "REJECT"


class DecisionEngine:
    """
    Tri-Way Pseudo-Label Decision Engine:
    - ACCEPT: High reliability and compliant with anatomical hierarchy (ET ⊆ TC ⊆ WT).
      Used directly as high-confidence supervisory pseudo-labels.
    - REFINE: Moderate reliability OR contains hierarchy violations.
      Routed to the Foundation Model (SAM-Med3D) for coarse-to-fine prompt refinement.
    - REJECT: Low reliability or severe irreparable noise.
      Discarded (gradient weight = 0) to prevent confirmation bias and model poisoning.

    Args:
        tau_high: High reliability threshold (default: 0.50, corresponding to ~80% top-class probability)
        tau_low: Low reliability threshold (default: 0.15, corresponding to ~50% top-class probability)
        delta_hvi: Maximum allowable Hierarchy Violation Index (default: 0.03)
        min_tumor_voxels: Minimum tumor size threshold to reject pure empty noise (default: 20)
    """
    def __init__(
        self,
        tau_high: float = 0.50,
        tau_low: float = 0.15,
        delta_hvi: float = 0.03,
        min_tumor_voxels: int = 20
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.delta_hvi = delta_hvi
        self.min_tumor_voxels = min_tumor_voxels

    def evaluate(
        self,
        reliabilities: Dict[str, float],
        hierarchy_stats: Dict[str, Any],
        wt_voxel_count: int
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate prediction and determines action.

        Args:
            reliabilities: Output from evaluate_subregion_reliabilities()
            hierarchy_stats: Output from compute_hierarchy_violation_index()
            wt_voxel_count: Number of foreground tumor voxels in WT
        Returns:
            Dictionary with action, reasoning, and loss weight.
        """
        r_mean = reliabilities.get('reliability_mean', 0.0)
        r_wt = reliabilities.get('reliability_WT', 0.0)
        r_tc = reliabilities.get('reliability_TC', 0.0)
        r_et = reliabilities.get('reliability_ET', 0.0)

        hvi_et = hierarchy_stats.get('hvi_et', 0.0)
        hvi_tc = hierarchy_stats.get('hvi_tc', 0.0)
        hvi_total = hierarchy_stats.get('hvi_total', 0.0)

        # 1. Condition: REJECT (Too small, chaotic, or unreliable)
        if wt_voxel_count < self.min_tumor_voxels:
            return {
                'action': PseudoLabelAction.REJECT,
                'weight': 0.0,
                'score': r_mean,
                'reason': f"Tumor extent too small ({wt_voxel_count} voxels < {self.min_tumor_voxels}). Rejected to avoid false positives."
            }

        if r_mean < self.tau_low or r_wt < self.tau_low:
            return {
                'action': PseudoLabelAction.REJECT,
                'weight': 0.0,
                'score': r_mean,
                'reason': f"Reliability score ({r_mean:.3f}) below rejection threshold ({self.tau_low:.2f}). Irreparable noise."
            }

        # 2. Condition: REFINE (Contains hierarchy violation OR moderate reliability)
        has_hierarchy_violation = (hvi_et > self.delta_hvi) or (hvi_tc > self.delta_hvi)
        # Check moderate reliability on present regions
        has_subregion_uncertainty = (r_tc > 0 and r_tc < self.tau_high) or (r_et > 0 and r_et < self.tau_high)
        is_moderate_reliability = (r_mean < self.tau_high) or has_subregion_uncertainty

        if has_hierarchy_violation:
            reason = (
                f"Anatomical hierarchy violation detected (HVI_ET={hvi_et*100:.1f}%, HVI_TC={hvi_tc*100:.1f}%). "
                f"Routed to SAM-Med3D for hierarchical coarse-to-fine refinement."
            )
            return {
                'action': PseudoLabelAction.REFINE,
                'weight': 0.8, # Scaled weight once refined
                'score': r_mean,
                'reason': reason
            }

        if is_moderate_reliability:
            reason = (
                f"Moderate reliability ({r_mean:.3f} in [{self.tau_low}, {self.tau_high}]). "
                f"Routed to SAM-Med3D for boundary refinement."
            )
            return {
                'action': PseudoLabelAction.REFINE,
                'weight': 0.8,
                'score': r_mean,
                'reason': reason
            }

        # 3. Condition: ACCEPT (High reliability AND strict hierarchy compliance)
        return {
            'action': PseudoLabelAction.ACCEPT,
            'weight': 1.0,
            'score': r_mean,
            'reason': (
                f"High reliability ({r_mean:.3f} >= {self.tau_high}) and compliant with "
                f"hierarchy (HVI={hvi_total*100:.2f}% <= {self.delta_hvi*100:.1f}%). "
                f"Accepted directly as supervisory pseudo-label."
            )
        }
