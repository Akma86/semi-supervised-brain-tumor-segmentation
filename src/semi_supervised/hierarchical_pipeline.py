"""
Complete End-to-End Hierarchical Pseudo-Label Refinement Pipeline:
Orchestrates:
    Unlabeled MRI -> 3D Segmentor -> Reliability & Hierarchy Check ->
    Decision Engine (ACCEPT / REFINE / REJECT) -> SAM-Med3D Coarse-to-Fine Refinement ->
    Final Pseudo-Label Supervision.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reliability import evaluate_subregion_reliabilities, compute_entropy_map
from .hierarchy import compute_hierarchy_violation_index
from .decision_engine import DecisionEngine, PseudoLabelAction
from .hierarchical_prompter import HierarchicalSAMRefiner


class HierarchicalPseudoLabelPipeline:
    """
    End-to-End Hierarchy-Aware Semi-Supervised Pipeline.
    Combines 3D U-Net baseline with SAM-Med3D targeted refinement and
    tri-way filtering.
    """
    def __init__(
        self,
        tau_high: float = 0.85,
        tau_low: float = 0.50,
        delta_hvi: float = 0.03,
        foundation_refiner: Optional[Any] = None
    ):
        self.decision_engine = DecisionEngine(
            tau_high=tau_high,
            tau_low=tau_low,
            delta_hvi=delta_hvi
        )
        self.refiner = HierarchicalSAMRefiner(foundation_segmentor=foundation_refiner)

    def process_unlabeled_sample(
        self,
        image: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Processes a single unlabeled 3D MRI volume through the pipeline.

        Args:
            image: Multi-modal MRI input [C, H, W, D] or [1, C, H, W, D]
            logits: Output logits from 3D U-Net [num_classes, H, W, D] or [1, num_classes, H, W, D]
        Returns:
            Dictionary containing:
                - 'action': 'ACCEPT', 'REFINE', or 'REJECT'
                - 'weight': Loss weight in [0.0, 1.0]
                - 'pseudo_label': Assembled multi-class mask (0: BG, 1: NCR, 2: ED, 3: ET)
                - 'reliabilities': Reliability scores for WT, TC, ET
                - 'hierarchy_stats': HVI violation indexes
                - 'reason': Decision explanation string
        """
        if logits.dim() == 4:
            logits = logits.unsqueeze(0)
        if image.dim() == 4:
            image = image.unsqueeze(0)

        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1) # [1, H, W, D]

        # Decompose into BraTS clinical sub-regions
        # Label 1: NCR, Label 2: ED, Label 3: ET
        wt_mask = (pred_labels == 1) | (pred_labels == 2) | (pred_labels == 3)
        tc_mask = (pred_labels == 1) | (pred_labels == 3)
        et_mask = (pred_labels == 3)

        wt_voxel_count = int(wt_mask.sum().item())

        # 1. Compute Regional Reliability Scores
        reliabilities = evaluate_subregion_reliabilities(
            logits=logits,
            wt_mask=wt_mask.float(),
            tc_mask=tc_mask.float(),
            et_mask=et_mask.float()
        )

        # 2. Check Anatomical Hierarchy Constraints (ET ⊆ TC ⊆ WT)
        hierarchy_stats = compute_hierarchy_violation_index(
            wt_mask=wt_mask[0],
            tc_mask=tc_mask[0],
            et_mask=et_mask[0]
        )

        # 3. Decision Engine: ACCEPT / REFINE / REJECT
        decision = self.decision_engine.evaluate(
            reliabilities=reliabilities,
            hierarchy_stats=hierarchy_stats,
            wt_voxel_count=wt_voxel_count
        )

        action = decision['action']
        weight = decision['weight']

        # 4. Action Execution
        if action == PseudoLabelAction.ACCEPT:
            # High confidence and hierarchy compliant: use directly
            final_pseudo_label = pred_labels[0]

        elif action == PseudoLabelAction.REFINE:
            # Route to SAM-Med3D Coarse-to-Fine Hierarchical Refinement
            refined_dict = self.refiner.refine_hierarchical(
                image=image[0],
                raw_wt=wt_mask[0].float(),
                raw_tc=tc_mask[0].float(),
                raw_et=et_mask[0].float()
            )
            final_pseudo_label = refined_dict['assembled_seg']

        else: # PseudoLabelAction.REJECT
            # Noisy or empty: zeroed mask, zero loss weight
            final_pseudo_label = torch.zeros_like(pred_labels[0])

        return {
            'action': action.value,
            'weight': weight,
            'pseudo_label': final_pseudo_label,
            'reliabilities': reliabilities,
            'hierarchy_stats': hierarchy_stats,
            'reason': decision['reason']
        }
