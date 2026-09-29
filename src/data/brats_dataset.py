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

    MODALITY_SUFFIXES = {
        't1c': '-t1c.nii.gz',
        't1n': '-t1n.nii.gz',
        't2f': '-t2f.nii.gz',
        't2w': '-t2w.nii.gz',
        'seg': '-seg.nii.gz'
    }

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
                if os.path.isdir(os.path.join(d, f)) and f.startswith("BraTS-SSA")
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
        if idx in self._cache:
            return self._cache[idx]

        folder = self.patient_folders[idx]
        patient_id = os.path.basename(folder)

        # Load multi-modal MRI scans
        modality_volumes = []
        for m in self.modalities:
            file_name = f"{patient_id}{self.MODALITY_SUFFIXES[m]}"
            file_path = os.path.join(folder, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing modality {m} for patient {patient_id}: {file_path}")
            vol = self.load_nifti_image(file_path)
            vol = self.normalize_intensity(vol)
            modality_volumes.append(vol)

        # Stack into [C, H, W, D] where C = len(modalities) (typically 4)
        image = np.stack(modality_volumes, axis=0)

        # Load Ground Truth segmentation mask if labeled
        if self.is_labeled:
            seg_file = os.path.join(folder, f"{patient_id}{self.MODALITY_SUFFIXES['seg']}")
            if os.path.exists(seg_file):
                seg = self.load_nifti_image(seg_file).astype(np.int64)
            else:
                seg = np.zeros(image.shape[1:], dtype=np.int64)
        else:
            # For unlabeled cases in semi-supervised training
            seg = np.zeros(image.shape[1:], dtype=np.int64)

        sample = {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(seg),
            'patient_id': patient_id,
            'is_labeled': self.is_labeled
        }

        if self.transform is not None:
            sample = self.transform(sample)

        if self.cache_data:
            self._cache[idx] = sample

        return sample
