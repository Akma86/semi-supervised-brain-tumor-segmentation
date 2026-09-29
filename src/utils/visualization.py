"""
Visualization utilities for Multi-Modal MRI and Brain Tumor Segmentation masks.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_brats_colormap():
    """
    Returns discrete colormap for BraTS labels:
    0: Transparent (Background)
    1: Red (Necrotic and Non-Enhancing Core - NCR)
    2: Green (Peritumoral Edema - ED)
    3: Yellow (Enhancing Tumor - ET)
    """
    colors = [
        (0.0, 0.0, 0.0, 0.0), # 0: Background (Fully transparent)
        (0.9, 0.2, 0.2, 0.7), # 1: NCR (Red)
        (0.2, 0.8, 0.3, 0.7), # 2: ED (Green)
        (1.0, 0.85, 0.1, 0.8) # 3: ET (Yellow)
    ]
    return mcolors.ListedColormap(colors)


def plot_multimodal_slices(
    modalities_dict: dict,
    slice_idx: int,
    seg_mask: Optional[np.ndarray] = None,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (18, 5)
):
    """
    Plots the 4 MRI sequences side-by-side with optional segmentation mask overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    modality_keys = ['t1n', 't1c', 't2w', 't2f']
    modality_titles = [
        'T1 Native (t1n)',
        'T1 Contrast (t1c)',
        'T2 Weighted (t2w)',
        'T2 FLAIR (t2f)'
    ]
    cmap_seg = get_brats_colormap()

    for idx, (m_key, m_title) in enumerate(zip(modality_keys, modality_titles)):
        ax = axes[idx]
        if m_key in modalities_dict:
            img_slice = modalities_dict[m_key][:, :, slice_idx]
            ax.imshow(img_slice.T, cmap='gray', origin='lower')

            if seg_mask is not None:
                mask_slice = seg_mask[:, :, slice_idx]
                ax.imshow(mask_slice.T, cmap=cmap_seg, origin='lower', vmin=0, vmax=3, interpolation='nearest')

            ax.set_title(f"{m_title}\n(Slice {slice_idx})", fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"{title_prefix} Multi-Modal MRI Slice Inspection", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_triplanar_view(
    volume: np.ndarray,
    seg_mask: Optional[np.ndarray] = None,
    center_coords: Optional[Tuple[int, int, int]] = None,
    title: str = "Tri-Planar Orthogonal View",
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plots Axial (Z), Coronal (Y), and Sagittal (X) orthogonal cross-sections.
    """
    if center_coords is None:
        if seg_mask is not None and (seg_mask > 0).any():
            coords = np.argwhere(seg_mask > 0)
            center = coords.mean(axis=0).astype(int)
            cx, cy, cz = center[0], center[1], center[2]
        else:
            cx, cy, cz = volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2
    else:
        cx, cy, cz = center_coords

    cmap_seg = get_brats_colormap()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Axial View (Z slice)
    ax = axes[0]
    ax.imshow(volume[:, :, cz].T, cmap='gray', origin='lower')
    if seg_mask is not None:
        ax.imshow(seg_mask[:, :, cz].T, cmap=cmap_seg, origin='lower', vmin=0, vmax=3, interpolation='nearest')
    ax.set_title(f"Axial View (Z = {cz})", fontweight='bold')
    ax.axis('off')

    # 2. Coronal View (Y slice)
    ax = axes[1]
    ax.imshow(volume[:, cy, :].T, cmap='gray', origin='lower', aspect=volume.shape[2]/volume.shape[0]*1.2)
    if seg_mask is not None:
        ax.imshow(seg_mask[:, cy, :].T, cmap=cmap_seg, origin='lower', vmin=0, vmax=3, interpolation='nearest', aspect=volume.shape[2]/volume.shape[0]*1.2)
    ax.set_title(f"Coronal View (Y = {cy})", fontweight='bold')
    ax.axis('off')

    # 3. Sagittal View (X slice)
    ax = axes[2]
    ax.imshow(volume[cx, :, :].T, cmap='gray', origin='lower', aspect=volume.shape[2]/volume.shape[1]*1.2)
    if seg_mask is not None:
        ax.imshow(seg_mask[cx, :, :].T, cmap=cmap_seg, origin='lower', vmin=0, vmax=3, interpolation='nearest', aspect=volume.shape[2]/volume.shape[1]*1.2)
    ax.set_title(f"Sagittal View (X = {cx})", fontweight='bold')
    ax.axis('off')

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig
