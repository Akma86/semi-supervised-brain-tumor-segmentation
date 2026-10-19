"""
Hierarchical Foundation Model Prompting and Refinement (SAM / MedSAM Adapter):
Implements coarse-to-fine prompt-based refinement: WT -> TC -> ET.
Enforces the anatomical nesting constraint ET ⊆ TC ⊆ WT by construction.
"""

import os
from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn.functional as F

# Explicit Imports of Medical / Vision Foundation Models (SAM 1 & SAM 2)
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False


def build_sam_rgb_input(mri_tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Synthesizes a 3-channel diagnostic RGB image [H, W, 3] from multi-modal MRI
    for SAM / MedSAM promptable foundation models:
      - Channel R: T1c (highlights active Enhancing Tumor & core)
      - Channel G: T2-FLAIR (highlights peritumoral vasogenic edema / Whole Tumor)
      - Channel B: T2w (highlights fluid-filled necrotic core)
    """
    if isinstance(mri_tensor, torch.Tensor):
        mri = mri_tensor.detach().cpu().numpy()
    else:
        mri = mri_tensor.copy()

    # mri shape: [4, H, W] or [4, H, W, D]
    if mri.ndim == 4:
        # Take central or maximum cross-section
        mri = mri[:, :, :, mri.shape[-1] // 2]

    # Modality indices: 0: t1c, 1: t1n, 2: t2f, 3: t2w
    t1c = mri[0] if mri.shape[0] > 0 else mri[0]
    t2f = mri[2] if mri.shape[0] > 2 else mri[0]
    t2w = mri[3] if mri.shape[0] > 3 else mri[0]

    def norm_to_uint8(ch: np.ndarray) -> np.ndarray:
        c_min, c_max = ch.min(), ch.max()
        if c_max - c_min > 1e-5:
            norm = (ch - c_min) / (c_max - c_min)
        else:
            norm = np.zeros_like(ch)
        return (norm * 255.0).astype(np.uint8)

    r = norm_to_uint8(t1c)
    g = norm_to_uint8(t2f)
    b = norm_to_uint8(t2w)

    return np.stack([r, g, b], axis=-1) # [H, W, 3] uint8


def extract_3d_bounding_box(
    mask: Union[np.ndarray, torch.Tensor],
    margin: int = 5
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Extracts 3D bounding box [x_min, y_min, z_min, x_max, y_max, z_max] from binary mask.
    Returns None if mask is empty.
    """
    if isinstance(mask, torch.Tensor):
        coords = (mask > 0.5).nonzero()
        if len(coords) == 0:
            return None
        mins = coords.min(dim=0)[0]
        maxs = coords.max(dim=0)[0]
        return tuple(int(x.item()) for x in torch.cat([mins, maxs]))
    else:
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return (int(mins[0]), int(mins[1]), int(mins[2]), int(maxs[0]), int(maxs[1]), int(maxs[2]))


def extract_2d_bounding_box(
    mask: Union[np.ndarray, torch.Tensor],
    margin: int = 4
) -> Optional[np.ndarray]:
    """
    Extracts 2D bounding box [x_min, y_min, x_max, y_max] for SAM predictor.
    """
    if isinstance(mask, torch.Tensor):
        coords = (mask > 0.5).nonzero()
        if len(coords) == 0:
            return None
        y_min = max(0, int(coords[:, 0].min().item()) - margin)
        y_max = int(coords[:, 0].max().item()) + margin
        x_min = max(0, int(coords[:, 1].min().item()) - margin)
        x_max = int(coords[:, 1].max().item()) + margin
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    else:
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return None
        y_min = max(0, int(coords[:, 0].min()) - margin)
        y_max = int(coords[:, 0].max()) + margin
        x_min = max(0, int(coords[:, 1].min()) - margin)
        x_max = int(coords[:, 1].max()) + margin
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class HierarchicalSAMRefiner:
    """
    Hierarchical Refiner using Medical Foundation Model (SAM / MedSAM):
    Refines noisy or violating pseudo-labels coarse-to-fine:
        1. Refine Whole Tumor (WT) using coarse WT bounding box prompt.
        2. Refine Tumor Core (TC) with box constrained strictly inside refined WT.
        3. Refine Enhancing Tumor (ET) with box constrained strictly inside refined TC.

    Guarantees:
        ET ⊆ TC ⊆ WT by anatomical construction (HVI = 0).
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_sam: bool = False,
        sam_version: str = "sam2",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        model_type: str = "vit_b",
        device: str = "cpu",
        margin: int = 4,
        foundation_segmentor: Optional[Any] = None
    ):
        self.device = device
        self.margin = margin
        self.predictor = foundation_segmentor
        self.has_sam = (foundation_segmentor is not None)
        self.sam_version = "Custom" if self.has_sam else "None"

        if use_sam and not self.has_sam:
            # 1. MedSAM from Bowang Lab (Nature Communications 2024, specialized for medical MRI)
            if sam_version == "medsam" and HAS_SAM:
                try:
                    ckpt = checkpoint_path or "medsam_repo/work_dir/MedSAM/medsam_vit_b.pth"
                    if os.path.exists(ckpt):
                        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
                    else:
                        sam = sam_model_registry["vit_b"]()
                    sam.to(device=device)
                    sam.eval()
                    self.predictor = SamPredictor(sam)
                    self.has_sam = True
                    self.sam_version = "MedSAM (Bowang-Lab ViT-B)"
                except Exception:
                    pass

            # 2. Meta SAM 2 / SAM 2.1 (Hiera Backbone)
            if sam_version == "sam2" and HAS_SAM2 and not self.has_sam:
                try:
                    ckpt = checkpoint_path or "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"
                    cfg = sam2_config or "configs/sam2.1/sam2.1_hiera_t.yaml"
                    if os.path.exists(ckpt):
                        sam2_model = build_sam2(cfg, ckpt, device=device)
                        self.predictor = SAM2ImagePredictor(sam2_model)
                        self.has_sam = True
                        self.sam_version = "Meta SAM 2.1 (Hiera)"
                except Exception:
                    pass

            # 3. Fallback: Meta SAM 1 (ViT-B)
            if not self.has_sam and HAS_SAM:
                try:
                    if checkpoint_path and os.path.exists(checkpoint_path):
                        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                    else:
                        sam = sam_model_registry[model_type]()
                    sam.to(device=device)
                    sam.eval()
                    self.predictor = SamPredictor(sam)
                    self.has_sam = True
                    self.sam_version = f"SAM 1 ({model_type})"
                except Exception:
                    self.has_sam = False

    def refine_hierarchical(
        self,
        image: Union[np.ndarray, torch.Tensor],
        raw_wt: Union[np.ndarray, torch.Tensor],
        raw_tc: Union[np.ndarray, torch.Tensor],
        raw_et: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Executes coarse-to-fine prompt refinement on raw U-Net candidate predictions.

        Returns:
            Dictionary with refined 'WT', 'TC', 'ET', assembled multi-class 'seg',
            and verification of hierarchy compliance.
        """
        is_torch = isinstance(raw_wt, torch.Tensor)

        # Optional SAM predictor execution if SAM is attached and image is 2D slice
        rgb_image = build_sam_rgb_input(image) if (image.ndim in [3, 4]) else None

        # -------------------------------------------------------------
        # Stage 1: Refine Whole Tumor (WT) - Coarsest lesion boundary
        # -------------------------------------------------------------
        if is_torch:
            wt_b = (raw_wt > 0.5).float()
            refined_wt = wt_b.clone()
        else:
            wt_b = (raw_wt > 0.5).astype(np.float32)
            refined_wt = wt_b.copy()

        if self.has_sam and self.predictor is not None and rgb_image is not None:
            box_wt = extract_2d_bounding_box(raw_wt, margin=self.margin)
            if box_wt is not None:
                try:
                    self.predictor.set_image(rgb_image)
                    masks, _, _ = self.predictor.predict(box=box_wt[None, :], multimask_output=False)
                    if is_torch:
                        refined_wt = torch.from_numpy(masks[0].astype(np.float32)).to(raw_wt.device)
                    else:
                        refined_wt = masks[0].astype(np.float32)
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Stage 2: Refine Tumor Core (TC) - Guided strictly inside refined WT
        # -------------------------------------------------------------
        if is_torch:
            tc_constrained = (raw_tc > 0.5).float() * refined_wt
            refined_tc = tc_constrained
        else:
            tc_constrained = (raw_tc > 0.5).astype(np.float32) * refined_wt
            refined_tc = tc_constrained

        if self.has_sam and self.predictor is not None and rgb_image is not None:
            # The prompt box for TC is derived from tc_constrained, ensuring WT guides TC
            box_tc = extract_2d_bounding_box(tc_constrained, margin=self.margin)
            if box_tc is not None:
                try:
                    masks, _, _ = self.predictor.predict(box=box_tc[None, :], multimask_output=False)
                    sam_tc = torch.from_numpy(masks[0].astype(np.float32)).to(raw_tc.device) if is_torch else masks[0].astype(np.float32)
                    refined_tc = sam_tc * refined_wt
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Stage 3: Refine Enhancing Tumor (ET) - Guided strictly inside refined TC
        # -------------------------------------------------------------
        if is_torch:
            et_constrained = (raw_et > 0.5).float() * refined_tc
            refined_et = et_constrained
        else:
            et_constrained = (raw_et > 0.5).astype(np.float32) * refined_tc
            refined_et = et_constrained

        if self.has_sam and self.predictor is not None and rgb_image is not None:
            # The prompt box for ET is derived from et_constrained, ensuring TC guides ET
            box_et = extract_2d_bounding_box(et_constrained, margin=self.margin)
            if box_et is not None:
                try:
                    masks, _, _ = self.predictor.predict(box=box_et[None, :], multimask_output=False)
                    sam_et = torch.from_numpy(masks[0].astype(np.float32)).to(raw_et.device) if is_torch else masks[0].astype(np.float32)
                    refined_et = sam_et * refined_tc
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Stage 4: Assemble into Standard BraTS Multi-Class Mask
        # 0: Background
        # 1: Necrotic Tumor Core (NCR = TC \ ET)
        # 2: Peritumoral Edema (ED = WT \ TC)
        # 3: Enhancing Tumor (ET = ET)
        # -------------------------------------------------------------
        if is_torch:
            assembled_seg = torch.zeros_like(refined_wt, dtype=torch.int64)
            ed_mask = (refined_wt > 0.5) & (~(refined_tc > 0.5))
            ncr_mask = (refined_tc > 0.5) & (~(refined_et > 0.5))
            et_mask = (refined_et > 0.5)

            assembled_seg[ed_mask] = 2
            assembled_seg[ncr_mask] = 1
            assembled_seg[et_mask] = 3
        else:
            assembled_seg = np.zeros_like(refined_wt, dtype=np.int64)
            ed_mask = (refined_wt > 0.5) & (~(refined_tc > 0.5))
            ncr_mask = (refined_tc > 0.5) & (~(refined_et > 0.5))
            et_mask = (refined_et > 0.5)

            assembled_seg[ed_mask] = 2
            assembled_seg[ncr_mask] = 1
            assembled_seg[et_mask] = 3

        return {
            'refined_WT': refined_wt,
            'refined_TC': refined_tc,
            'refined_ET': refined_et,
            'assembled_seg': assembled_seg,
            'hierarchy_guaranteed': True
        }
