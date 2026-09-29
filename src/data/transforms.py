"""
Data transforms, augmentations, and prompt generation for medical foundation models.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


class CropForeground:
    """Crops the zero-background margin around the brain to reduce volume size."""
    def __init__(self, margin: int = 4):
        self.margin = margin

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = sample['image'] # [C, H, W, D]
        mask = sample.get('mask')

        # Find non-zero bounds across all modalities
        nonzero = (image.abs().sum(dim=0) > 1e-4).nonzero()
        if len(nonzero) == 0:
            return sample

        min_coords = nonzero.min(dim=0)[0]
        max_coords = nonzero.max(dim=0)[0]

        h_start = max(0, min_coords[0].item() - self.margin)
        h_end = min(image.shape[1], max_coords[0].item() + self.margin + 1)
        w_start = max(0, min_coords[1].item() - self.margin)
        w_end = min(image.shape[2], max_coords[1].item() + self.margin + 1)
        d_start = max(0, min_coords[2].item() - self.margin)
        d_end = min(image.shape[3], max_coords[2].item() + self.margin + 1)

        sample['image'] = image[:, h_start:h_end, w_start:w_end, d_start:d_end]
        if mask is not None:
            sample['mask'] = mask[h_start:h_end, w_start:w_end, d_start:d_end]

        return sample


class RandomSpatialCrop3D:
    """Randomly crops a fixed 3D patch (e.g. 128x128x128) with tumor foreground bias."""
    def __init__(self, roi_size: Tuple[int, int, int] = (128, 128, 128), pos_ratio: float = 0.7):
        self.roi_size = roi_size
        self.pos_ratio = pos_ratio

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = sample['image']
        mask = sample.get('mask')
        C, H, W, D = image.shape
        rH, rW, rD = self.roi_size

        if H <= rH or W <= rW or D <= rD:
            # Pad if smaller than roi_size
            pad_h = max(0, rH - H)
            pad_w = max(0, rW - W)
            pad_d = max(0, rD - D)
            image = F.pad(image, (0, pad_d, 0, pad_w, 0, pad_h))
            if mask is not None:
                mask = F.pad(mask.unsqueeze(0), (0, pad_d, 0, pad_w, 0, pad_h)).squeeze(0)
            H, W, D = image.shape[1:]

        # Choose center: foreground-biased or uniform random
        is_pos = (torch.rand(1).item() < self.pos_ratio) and (mask is not None) and (mask > 0).any()
        if is_pos:
            fg_indices = (mask > 0).nonzero()
            chosen_idx = fg_indices[torch.randint(0, len(fg_indices), (1,)).item()]
            cH, cW, cD = chosen_idx[0].item(), chosen_idx[1].item(), chosen_idx[2].item()
            sH = np.clip(cH - rH // 2, 0, H - rH)
            sW = np.clip(cW - rW // 2, 0, W - rW)
            sD = np.clip(cD - rD // 2, 0, D - rD)
        else:
            sH = np.random.randint(0, H - rH + 1)
            sW = np.random.randint(0, W - rW + 1)
            sD = np.random.randint(0, D - rD + 1)

        sample['image'] = image[:, sH:sH + rH, sW:sW + rW, sD:sD + rD]
        if mask is not None:
            sample['mask'] = mask[sH:sH + rH, sW:sW + rW, sD:sD + rD]

        return sample


class FoundationModelPromptGenerator:
    """
    Extracts 2D slices containing tumor and generates prompt bounding boxes / point coordinates
    suitable for promptable Medical Foundation Models (SAM, MedSAM, SAM-Med2D).
    """
    def __init__(self, perturbation_range: int = 5):
        self.perturbation_range = perturbation_range

    def get_bounding_box_prompt(self, mask_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes a bounding box [x_min, y_min, x_max, y_max] from a 2D mask,
        optionally perturbed to simulate clinical uncertainty or weak bounding box annotations.
        """
        y_indices, x_indices = np.where(mask_2d > 0)
        if len(y_indices) == 0:
            return None

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Add jitter/perturbation
        if self.perturbation_range > 0:
            H, W = mask_2d.shape
            x_min = max(0, x_min - np.random.randint(0, self.perturbation_range + 1))
            y_min = max(0, y_min - np.random.randint(0, self.perturbation_range + 1))
            x_max = min(W - 1, x_max + np.random.randint(0, self.perturbation_range + 1))
            y_max = min(H - 1, y_max + np.random.randint(0, self.perturbation_range + 1))

        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
