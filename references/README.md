# Literature References & Research Foundations

This directory catalogs the literature foundation for the paper:
**"Semi-Supervised Brain Tumor Segmentation Using Medical Foundation Models Under Limited Annotation"**

> **Note:** The full PDF manuscripts are located locally under `references/papers/` (git-ignored to keep the repository lightweight and adhere to redistribution policies).

---

## 1. Medical Foundation Models & Adaptation

1. **SemiSAM: Rethinking Semi-Supervised Medical Image Segmentation with Foundation Models**
   - *Venue:* Medical Image Analysis (2025)
   - *Key Takeaway:* Adapts the Segment Anything Model (SAM) to medical domains via prompt-guided consistency and teacher-student distillation under sparse annotations.
2. **Semi-SwinUNETR: Towards 3D Swin Vision Transformer-Based UNet with Limited Annotations**
   - *Venue:* IEEE / arXiv
   - *Key Takeaway:* Integrates hierarchical 3D vision transformers with self-supervised pretext tasks and consistency regularization for volumetric segmentation.
3. **Medical Foundation Models for Image Segmentation: Benchmarks and Opportunities**
   - *Venue:* npj Digital Medicine / arXiv (2025)
   - *Key Takeaway:* Comprehensive review of MedSAM, SAM-Med2D, SegVol, and BiomedParse for cross-domain generalization.

---

## 2. Semi-Supervised Learning & Regularization Paradigms

4. **BUFNet: Boundary-Aware and Uncertainty-Driven Multi-Modal Framework**
   - *Venue:* Medical Image Analysis (2026)
   - *Key Takeaway:* Uses boundary uncertainty maps to filter out noisy pseudo-labels in multi-modal brain lesion segmentation.
5. **Two-Level Semi-Supervised Collaborative Medical Image Segmentation**
   - *Venue:* Medical Image Analysis (2026)
   - *Key Takeaway:* Hierarchical collaboration between pixel-level and prototype-level representations to leverage unlabeled slices.
6. **Prototype-Oriented Contrastive Learning for Semi-Supervised Medical Image Segmentation**
   - *Venue:* Biomedical Signal Processing and Control (2024)
   - *Key Takeaway:* Class-aware prototype clustering to stabilize representations under extreme label scarcity (e.g., 5-10% labels).
7. **Semi-Supervised Medical Image Segmentation via Hard Positive Mining**
   - *Venue:* Pattern Recognition (2024)
   - *Key Takeaway:* Addresses extreme class imbalance between large edema regions and small necrotic cores by mining hard positive voxels.
8. **Semi-Supervised Learning Allows for Improved Segmentation with Reduced Annotation Burden**
   - *Venue:* Magnetic Resonance Imaging (2025)
   - *Key Takeaway:* Quantifies empirical performance thresholds when transitioning from fully supervised to 10%-20% supervised regimes.

---

## 3. BraTS & Multi-Modal Brain Lesion Benchmarks

9. **Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries**
   - *Venue:* Springer LNCS (BraTS Challenge Proceedings)
   - *Key Takeaway:* Benchmark evaluation criteria for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) across multi-modal scans.
10. **BraTS-Africa Sub-Saharan Africa (SSA) Benchmark**
    - *Key Takeaway:* Evaluates multi-center generalizability on underserved populations, capturing clinical variability in imaging protocols and glioma presentation.
