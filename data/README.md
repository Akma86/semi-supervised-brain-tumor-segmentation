# BraTS-Africa (Sub-Saharan Africa - SSA) Dataset

This directory contains the multi-parametric Magnetic Resonance Imaging (mpMRI) scans from the **BraTS-Africa** challenge cohort.

## Dataset Structure

```text
data/
└── BraTS-Africa/
    ├── 51_OtherNeoplasms/       # 51 cases of non-glioma intracranial neoplasms
    │   ├── BraTS-SSA-00009-000/
    │   └── ...
    └── 95_Glioma/               # 95 cases of adult-type diffuse gliomas
        ├── BraTS-SSA-00002-000/
        │   ├── BraTS-SSA-00002-000-seg.nii.gz  # Multi-class ground-truth segmentation mask
        │   ├── BraTS-SSA-00002-000-t1c.nii.gz  # T1-weighted post-contrast (T1CE / T1Gd)
        │   ├── BraTS-SSA-00002-000-t1n.nii.gz  # T1-weighted native pre-contrast
        │   ├── BraTS-SSA-00002-000-t2f.nii.gz  # T2-weighted FLAIR
        │   └── BraTS-SSA-00002-000-t2w.nii.gz  # T2-weighted
        └── ...
```

## Image Characteristics

- **Format:** Neuroimaging Informatics Technology Initiative (NIfTI, compressed `.nii.gz`)
- **Spatial Resolution:** Standardized 1 mm³ isotropic resolution.
- **Volume Dimensions:** $240 \times 240 \times 155$ voxels ($X \times Y \times Z$).
- **Coregistration:** All four modalities for a given patient are rigidly aligned to the same anatomical space.
- **Skull-stripping:** Scans have undergone standardized brain extraction.

## Multi-Modal MRI Sequences

| File Suffix | Modality Name | Clinical Significance in Brain Tumor Diagnostics |
| :--- | :--- | :--- |
| `-t1n.nii.gz` | **T1-Native** (pre-contrast) | Delineates general brain anatomy and boundaries of normal gray/white matter structures. |
| `-t1c.nii.gz` | **T1-Contrast** (post-gadolinium) | Shows breakdown of the blood-brain barrier (BBB); critical for demarcating actively proliferating tumor margins. |
| `-t2w.nii.gz` | **T2-Weighted** | Highlights high-water-content regions, edema, and ventricles (CSF appears bright). |
| `-t2f.nii.gz` | **T2-FLAIR** (Fluid-Attenuated) | Suppresses fluid (CSF) signals, allowing clear visualization of peritumoral infiltration and edema. |

## Ground Truth Annotation Labels (`seg.nii.gz`)

The segmentation labels adhere to the updated BraTS standardization:

| Voxel Value | Label Description | Histopathological Correlate |
| :---: | :--- | :--- |
| **0** | Background | Healthy brain parenchyma or non-brain background voxels. |
| **1** | Necrotic Tumor Core (NCR) | Non-enhancing, avascular necrotic tumor debris. |
| **2** | Peritumoral Edematous Tissue (ED) | Infiltrating edematous zone surrounding the solid tumor. |
| **3** | Enhancing Tumor (ET) | Hyper-vascularized, viable active neoplastic tumor tissue. |

### Standard BraTS Evaluation Sub-Regions

In clinical benchmarks and challenge evaluations, models are evaluated on nested, overlapping composite regions:

1. **Whole Tumor (WT):** Labels `1 + 2 + 3` (Complete pathological extent).
2. **Tumor Core (TC):** Labels `1 + 3` (Solid resectable tumor core).
3. **Enhancing Tumor (ET):** Label `3` (Active neoplastic core).
