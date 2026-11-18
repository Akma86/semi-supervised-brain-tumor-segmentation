"""
BraTS-Africa Dataset Loader for 3D Volumes and 2D Foundation Model Prompts.
Supports semi-supervised loading where a subset of samples are labeled,
and the remainder are treated as unlabeled data.
"""

import os
import glob
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class BraTSDataset(Dataset):
    """
    BraTS-Africa Multi-Modal MRI Dataset.
    Loads multi-parametric MRI modalities (t1n, t1c, t2w, t2f) and segmentation ground truth.

    Args:
        data_dir: Path to directory containing patient folders (e.g. 'data/BraTS-Africa/95_Glioma')
        modalities: List of modality keys to load (default: ['t1c', 't1n', 't2f', 't2w'])
        is_labeled: If False, ground truth segmentation is withheld (used for semi-supervised training)
        patient_ids: Optional explicit list of patient IDs to include
        transform: Optional callable transform to apply to images and masks
        mode: '3d' for full volumetric cubes, '2d' for slice-by-slice prompting
    """

    MODALITY_PATTERNS = {
        't1c': ['-t1c.nii.gz', '_t1ce.nii.gz', '_t1ce.nii', '-t1c.nii'],
        't1n': ['-t1n.nii.gz', '_t1.nii.gz', '_t1.nii', '-t1n.nii'],
        't2f': ['-t2f.nii.gz', '_flair.nii.gz', '_flair.nii', '-t2f.nii'],
        't2w': ['-t2w.nii.gz', '_t2.nii.gz', '_t2.nii', '-t2w.nii'],
        'seg': ['-seg.nii.gz', '_seg.nii.gz', '_seg.nii', '-seg.nii']
    }

    @classmethod
    def _resolve_file_path(cls, folder: str, patient_id: str, modality_key: str) -> Optional[str]:
        patterns = cls.MODALITY_PATTERNS.get(modality_key, [])
        for pat in patterns:
            cand = os.path.join(folder, f"{patient_id}{pat}")
            if os.path.exists(cand):
                return cand
        return None

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        modalities: Optional[List[str]] = None,
        is_labeled: bool = True,
        patient_ids: Optional[List[str]] = None,
        transform=None,
        mode: str = '3d',
        cache_data: bool = False
    ):
        self.modalities = modalities or ['t1c', 't1n', 't2f', 't2w']
        self.is_labeled = is_labeled
        self.transform = transform
        self.mode = mode
        self.cache_data = cache_data
        self._cache = {}

        # Resolve patient paths
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = data_dir

        self.patient_folders = []
        for d in data_dirs:
            if not os.path.exists(d):
                continue
            candidates = sorted([
                os.path.join(d, f) for f in os.listdir(d)
                if os.path.isdir(os.path.join(d, f)) and not f.startswith(".")
            ])
            self.patient_folders.extend(candidates)

        if patient_ids is not None:
            self.patient_folders = [
                p for p in self.patient_folders
                if os.path.basename(p) in patient_ids
            ]

        if len(self.patient_folders) == 0:
            raise ValueError(f"No patient folders found in provided directories: {data_dirs}")

    def __len__(self) -> int:
        return len(self.patient_folders)

    @staticmethod
    def load_nifti_image(file_path: str) -> np.ndarray:
        """Loads a NIfTI image file into a NumPy array."""
        nii = nib.load(file_path)
        return nii.get_fdata().astype(np.float32)

    @staticmethod
    def normalize_intensity(volume: np.ndarray) -> np.ndarray:
        """
        Z-score intensity normalization computed over brain parenchyma (non-zero voxels).
        Clips extreme outliers outside [0.5, 99.5] percentiles.
        """
        mask = volume > 0
        if np.any(mask):
            p_low, p_high = np.percentile(volume[mask], (0.5, 99.5))
            volume = np.clip(volume, p_low, p_high)
            mean = volume[mask].mean()
            std = volume[mask].std()
            if std > 1e-6:
                volume = (volume - mean) / std
        return volume

    @staticmethod
    def compute_subregions(seg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Converts multi-class discrete segmentation to standard BraTS sub-regions:
        - WT (Whole Tumor): labels 1, 2, 3
        - TC (Tumor Core): labels 1, 3
        - ET (Enhancing Tumor): label 3
        """
        wt = (seg == 1) | (seg == 2) | (seg == 3)
        tc = (seg == 1) | (seg == 3)
        et = (seg == 3)
        return {
            'WT': wt.astype(np.float32),
            'TC': tc.astype(np.float32),
            'ET': et.astype(np.float32)
        }

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if self.mode in ('2d', '2.5d'):
            cached = self._get_cached_volume(idx) if self.cache_data else None
            if cached is not None:
                return self._sample_slice_from_entry(cached)
        elif idx in self._cache:
            return self._cache[idx]

        image, seg, patient_id = self._load_raw_patient_volume(idx)

        if self.mode in ('2d', '2.5d'):
            entry = self._pack_volume_entry(image, seg, patient_id)
            if self.cache_data:
                self._put_cached_volume(idx, entry)
            return self._sample_slice_from_entry(entry)
        else:
            sample = {
                'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(seg).long(),
                'patient_id': patient_id,
                'is_labeled': self.is_labeled
            }

        if self.transform is not None:
            sample = self.transform(sample)

        if self.cache_data:
            self._cache[idx] = sample

        return sample

    def _get_vol_cache(self) -> dict:
        if not hasattr(self, '_vol_cache'):
            self._vol_cache = {}
            self._max_vol_cache = 110
        return self._vol_cache

    def _put_cached_volume(self, idx: int, entry: dict) -> None:
        cache = self._get_vol_cache()
        if idx in cache:
            cache.pop(idx)
        elif len(cache) >= self._max_vol_cache:
            # Evict least-recently-used volume to stay within 16GB system RAM
            cache.pop(next(iter(cache)))
        cache[idx] = entry

    def _get_cached_volume(self, idx: int) -> Optional[dict]:
        cache = self._get_vol_cache()
        if idx not in cache:
            return None
        entry = cache.pop(idx)
        cache[idx] = entry
        return entry

    def _pack_volume_entry(self, image: np.ndarray, seg: np.ndarray, patient_id: str) -> dict:
        H, W, D = image.shape[1], image.shape[2], image.shape[3]
        fg = (np.abs(image).sum(axis=0) > 1e-4) | (seg > 0)
        coords = np.argwhere(fg)
        if len(coords) > 0:
            h0, w0, d0 = np.maximum(0, coords.min(axis=0) - 2)
            h1, w1, d1 = np.minimum([H, W, D], coords.max(axis=0) + 3)
        else:
            h0, w0, d0, h1, w1, d1 = 0, H, 0, W, 0, D

        tumor_slices = (
            np.where(seg.sum(axis=(0, 1)) > 0)[0]
            if (self.is_labeled and (seg > 0).any())
            else np.array([], dtype=np.int64)
        )
        brain_slices = np.where(np.abs(image).sum(axis=(0, 1, 2)) > 10.0)[0]
        if len(brain_slices) == 0:
            d_lo = min(10, max(0, D - 1))
            d_hi = max(d_lo + 1, D - d_lo)
            brain_slices = np.arange(d_lo, d_hi, dtype=np.int64)

        return {
            'patient_id': patient_id,
            'orig_shape': (H, W, D),
            'bbox': (int(h0), int(h1), int(w0), int(w1), int(d0), int(d1)),
            'image_crop': image[:, h0:h1, w0:w1, d0:d1].astype(np.float16),
            'seg_crop': seg[h0:h1, w0:w1, d0:d1].astype(np.uint8),
            'tumor_slices': tumor_slices,
            'brain_slices': brain_slices,
        }

    def _load_raw_patient_volume(self, idx: int) -> tuple:
        folder = self.patient_folders[idx]
        patient_id = os.path.basename(folder)

        modality_volumes = []
        for m in self.modalities:
            file_path = self._resolve_file_path(folder, patient_id, m)
            if file_path is None or not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing modality {m} for patient {patient_id} in {folder}")
            vol = self.load_nifti_image(file_path)
            vol = self.normalize_intensity(vol)
            modality_volumes.append(vol)

        image = np.stack(modality_volumes, axis=0)

        if self.is_labeled:
            seg_file = self._resolve_file_path(folder, patient_id, 'seg')
            if seg_file and os.path.exists(seg_file):
                seg = self.load_nifti_image(seg_file).astype(np.int64)
                seg[seg == 4] = 3
            else:
                seg = np.zeros(image.shape[1:], dtype=np.int64)
        else:
            seg = np.zeros(image.shape[1:], dtype=np.int64)

        return image, seg, patient_id

    def get_patient_volume_entry(self, idx: int) -> dict:
        """Loads and optionally LRU-caches the foreground-cropped 3D patient volume."""
        cached = self._get_cached_volume(idx) if self.cache_data else None
        if cached is not None:
            return cached

        image, seg, patient_id = self._load_raw_patient_volume(idx)
        entry = self._pack_volume_entry(image, seg, patient_id)
        if self.cache_data:
            self._put_cached_volume(idx, entry)
        return entry

    def extract_slice(self, entry: dict, z: int, slice_mode: Optional[str] = None) -> tuple:
        """
        Reconstructs a 2D [4, H, W] or 2.5D [12, H, W] image and [H, W] seg slice.
        2.5D orders channels as [z, z-1, z+1] so channels 0:4 hold the center slice.
        """
        mode = slice_mode or self.mode
        H, W, D = entry['orig_shape']
        h0, h1, w0, w1, d0, d1 = entry['bbox']
        img_crop = entry['image_crop']
        seg_crop = entry['seg_crop']

        z_indices = [z] if mode != '2.5d' else [z, max(0, z - 1), min(D - 1, z + 1)]
        img_out = np.zeros((4 * len(z_indices), H, W), dtype=np.float32)
        for s_idx, zs in enumerate(z_indices):
            if d0 <= zs < d1:
                c0, c1 = s_idx * 4, (s_idx + 1) * 4
                img_out[c0:c1, h0:h1, w0:w1] = img_crop[:, :, :, zs - d0].astype(np.float32)

        seg_out = np.zeros((H, W), dtype=np.int64)
        if d0 <= z < d1:
            seg_out[h0:h1, w0:w1] = seg_crop[:, :, z - d0].astype(np.int64)
        return img_out, seg_out

    def extract_slices_batch(self, entry: dict, z_list, slice_mode: Optional[str] = None) -> np.ndarray:
        """Vectorized extraction of a batch of 2D [B, 4, H, W] or 2.5D [B, 12, H, W] slices."""
        mode = slice_mode or self.mode
        H, W, D = entry['orig_shape']
        h0, h1, w0, w1, d0, d1 = entry['bbox']
        img_crop = entry['image_crop']

        offsets = (0,) if mode != '2.5d' else (0, -1, 1)
        z_arr = np.asarray(z_list, dtype=np.int64)
        imgs_out = np.zeros((len(z_arr), 4 * len(offsets), H, W), dtype=np.float32)

        for s_idx, off in enumerate(offsets):
            zs = np.clip(z_arr + off, 0, D - 1)
            valid = (zs >= d0) & (zs < d1)
            if valid.any():
                c0, c1 = s_idx * 4, (s_idx + 1) * 4
                rel = zs[valid] - d0
                imgs_out[valid, c0:c1, h0:h1, w0:w1] = np.transpose(
                    img_crop[:, :, :, rel], (3, 0, 1, 2)
                ).astype(np.float32)

        return imgs_out

    def _sample_slice_from_entry(self, entry: dict) -> Dict[str, Union[torch.Tensor, str]]:
        tumor_slices = entry['tumor_slices']
        brain_slices = entry['brain_slices']

        if len(tumor_slices) > 0 and np.random.rand() < 0.85:
            z = int(np.random.choice(tumor_slices))
        else:
            z = int(np.random.choice(brain_slices))

        image_slice, seg_slice = self.extract_slice(entry, z)
        sample = {
            'image': torch.from_numpy(image_slice).float(),
            'mask': torch.from_numpy(seg_slice).long(),
            'patient_id': entry['patient_id'],
            'slice_idx': z,
            'is_labeled': self.is_labeled
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class BraTSMultiSliceDataset(Dataset):
    """Repeats patient indexing across an epoch so dynamic slice sampling covers full 3D brains."""
    def __init__(self, base_dataset: BraTSDataset, slices_per_volume: int = 8):
        self.base_dataset = base_dataset
        self.slices_per_volume = max(1, int(slices_per_volume))

    def __len__(self) -> int:
        return len(self.base_dataset) * self.slices_per_volume

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.base_dataset[idx % len(self.base_dataset)]
