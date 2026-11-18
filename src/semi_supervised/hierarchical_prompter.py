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
        bg_val = ch.min()
        fg = ch[ch > bg_val + 1e-4]
        if len(fg) > 20:
            lo, hi = np.percentile(fg, 0.5), np.percentile(fg, 99.5)
            if hi - lo > 1e-5:
                norm = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
                norm[ch <= bg_val + 1e-4] = 0.0
                return (norm * 255.0).astype(np.uint8)
        c_max = ch.max()
        if c_max - bg_val > 1e-5:
            norm = (ch - bg_val) / (c_max - bg_val)
        else:
            norm = np.zeros_like(ch)
        return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)

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
        self.medsam_model = None
        self.has_sam = (foundation_segmentor is not None)
        self.sam_version = "Custom" if self.has_sam else "None"

        if use_sam and not self.has_sam:
            # MedSAM (Bowang-Lab) uses [0, 1] 1024x1024 scaling rather than SamPredictor ImageNet stats
            if sam_version == "medsam" and HAS_SAM:
                try:
                    ckpt = checkpoint_path or "medsam_repo/work_dir/MedSAM/medsam_vit_b.pth"
                    if os.path.exists(ckpt):
                        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
                    else:
                        sam = sam_model_registry["vit_b"]()
                    sam.to(device=device)
                    sam.eval()
                    self.medsam_model = sam
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

    def _encode_medsam_slice(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Encodes an RGB slice with official MedSAM [0, 1] 1024x1024 scaling (no ImageNet normalization)."""
        inp = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        inp_1024 = F.interpolate(inp, size=(1024, 1024), mode="bilinear", align_corners=False).to(self.device)
        use_amp = str(self.device).startswith("cuda")
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    return self.medsam_model.image_encoder(inp_1024)
            return self.medsam_model.image_encoder(inp_1024)

    def _predict_box_prob(
        self,
        medsam_emb: Optional[torch.Tensor],
        box_xyxy: np.ndarray,
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        """Predicts foreground probability [H, W] inside a 2D xyxy prompt box."""
        H, W = orig_hw
        if medsam_emb is not None and self.medsam_model is not None:
            box_1024 = box_xyxy[None, :] / np.array([W, H, W, H], dtype=np.float32) * 1024.0
            box_t = torch.as_tensor(box_1024, dtype=torch.float32, device=medsam_emb.device)[:, None, :]
            use_amp = medsam_emb.is_cuda
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        sp, dp = self.medsam_model.prompt_encoder(points=None, boxes=box_t, masks=None)
                        lr, _ = self.medsam_model.mask_decoder(
                            image_embeddings=medsam_emb.float(),
                            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp,
                            dense_prompt_embeddings=dp,
                            multimask_output=False,
                        )
                else:
                    sp, dp = self.medsam_model.prompt_encoder(points=None, boxes=box_t, masks=None)
                    lr, _ = self.medsam_model.mask_decoder(
                        image_embeddings=medsam_emb,
                        image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dp,
                        multimask_output=False,
                    )
                prob = F.interpolate(torch.sigmoid(lr.float()), size=(H, W), mode="bilinear", align_corners=False)
            return prob[0, 0].cpu().numpy().astype(np.float32)

        masks, _, _ = self.predictor.predict(box=box_xyxy[None, :], multimask_output=False)
        return masks[0].astype(np.float32)

    @staticmethod
    def _to_binary_np(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        np_arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
        return (np_arr > 0.5).astype(np.float32)

    @staticmethod
    def _assemble_multiclass(wt: np.ndarray, tc: np.ndarray, et: np.ndarray) -> np.ndarray:
        seg = np.zeros_like(wt, dtype=np.int64)
        wt_b, tc_b, et_b = wt > 0.5, tc > 0.5, et > 0.5
        seg[wt_b & (~tc_b)] = 2
        seg[tc_b & (~et_b)] = 1
        seg[et_b] = 3
        return seg

    def refine_hierarchical(
        self,
        image: Union[np.ndarray, torch.Tensor],
        raw_wt: Union[np.ndarray, torch.Tensor],
        raw_tc: Union[np.ndarray, torch.Tensor],
        raw_et: Union[np.ndarray, torch.Tensor],
        raw_probs: Optional[Dict[str, np.ndarray]] = None,
        blend_alpha: float = 0.65,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Executes coarse-to-fine prompt refinement on candidate predictions:
        WT -> TC -> ET, optionally soft-blending student and MedSAM probabilities
        while enforcing ET <= TC <= WT by construction.
        """
        is_torch = isinstance(raw_wt, torch.Tensor)
        src_device = raw_wt.device if is_torch else None

        wt_np = self._to_binary_np(raw_wt)
        tc_np = self._to_binary_np(raw_tc)
        et_np = self._to_binary_np(raw_et)

        rgb_image = build_sam_rgb_input(image) if (self.has_sam and image.ndim in [3, 4]) else None
        medsam_emb = None
        predictor_ready = False

        if self.has_sam and rgb_image is not None:
            try:
                if self.medsam_model is not None:
                    medsam_emb = self._encode_medsam_slice(rgb_image)
                    predictor_ready = True
                elif self.predictor is not None:
                    self.predictor.set_image(rgb_image)
                    predictor_ready = True
            except Exception:
                predictor_ready = False

        orig_hw = (wt_np.shape[0], wt_np.shape[1])

        def _refine_region(region_key: str, cond_mask: np.ndarray, parent_mask: Optional[np.ndarray]) -> np.ndarray:
            base = cond_mask * parent_mask if parent_mask is not None else cond_mask
            if not predictor_ready:
                return base
            box = extract_2d_bounding_box(base, margin=self.margin)
            if box is None:
                return base
            try:
                sam_prob = self._predict_box_prob(medsam_emb, box, orig_hw)
                if raw_probs is not None and region_key in raw_probs:
                    fused = blend_alpha * raw_probs[region_key] + (1.0 - blend_alpha) * sam_prob
                    out = (fused > 0.5).astype(np.float32)
                else:
                    out = (sam_prob > 0.5).astype(np.float32)
                return out * parent_mask if parent_mask is not None else out
            except Exception:
                return base

        refined_wt = _refine_region("WT", wt_np, None)
        refined_tc = _refine_region("TC", tc_np, refined_wt)
        refined_et = _refine_region("ET", et_np, refined_tc)
        assembled = self._assemble_multiclass(refined_wt, refined_tc, refined_et)

        if is_torch:
            return {
                'refined_WT': torch.from_numpy(refined_wt).to(src_device),
                'refined_TC': torch.from_numpy(refined_tc).to(src_device),
                'refined_ET': torch.from_numpy(refined_et).to(src_device),
                'assembled_seg': torch.from_numpy(assembled).to(src_device),
                'hierarchy_guaranteed': True,
            }

        return {
            'refined_WT': refined_wt,
            'refined_TC': refined_tc,
            'refined_ET': refined_et,
            'assembled_seg': assembled,
            'hierarchy_guaranteed': True,
        }
