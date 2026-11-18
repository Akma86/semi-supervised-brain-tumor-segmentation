# Ringkasan Eksekutif & Dokumentasi Eksperimen
## Semi-Supervised Brain Tumor Segmentation with MedSAM Foundation Model
**Dataset**: BraTS 2019 (Multimodal MRI: T1, T1ce, T2, FLAIR)  
**Terakhir Diperbarui**: November 2026  
**Status**: Target Val Dice `80%+` Berhasil Tercapai (**80.60% Mean Val Dice**, **76.93% Held-Out Test**)

---

## 1. Executive Summary

Eksperimen ini bertujuan mengatasi tantangan utama segmentasi tumor otak pada regime data berlabel terbatas (*semi-supervised learning* / SSL). Model dilatih menggunakan hanya sebagian kecil data berlabel (misal 40% label = 94 pasien berlabel dari total 235 pasien latih), sementara sisa volume diproses tanpa anotasi (*unlabeled* = 141 pasien).

### Lompatan Hasil:
* **Baseline Awal (Pipeline v1 - MiniUNet 2D)**:
  * Pada rasio 40% label, performa stagnan di kisaran **67.05%** (Supervised) dan **72.35%** (Semi-Supervised). Bahkan dengan 80% data berlabel sekalipun, baseline v1 hanya mencapai **68.34%**.
* **Pipeline Baru (Pipeline v2 - SegResNet 2.5D + Subregion Loss + MedSAM SoftBlend)**:
  * Berhasil melesat ke **80.60% Mean Val Dice** (WT **87.63%**, TC **80.24%**, ET **73.94%**).
  * Terjadi kenaikan absolut sebesar **+13.55%** dibanding baseline v1 supervised pada rasio yang sama.
  * Pada pengujian objektif **Held-Out Test (50 pasien asing)**, metode Semi-Supervised terbukti secara konsisten mengalahkan model Supervised murni di seluruh 6 metrik (Test Mean naik **+0.95%**, Tumor Core naik **+1.41%**, Enhancing Tumor naik **+1.33%**).

---

## 2. Tabel Komparasi Hasil Lengkap (BraTS 2019)

### A. Komparasi Antar Rasio Data Berlabel (Baseline v1 vs v2)

| Pipeline | Model / Metode | Rasio Label | Pasien Labeled / Unlabeled | Val WT (%) | Val TC (%) | Val ET (%) | Val Mean Dice (%) | Best Epoch |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1 Baseline** | Supervised (MiniUNet 2D) | 10% | 24 / 211 | 65.82 | 61.56 | 68.03 | **65.14** | 100 |
| **v1 Baseline** | Semi-Supervised (MedSAM v1) | 10% | 24 / 211 | 56.36 | 52.56 | 73.52 | **60.81** | 95 |
| **v1 Baseline** | Supervised (MiniUNet 2D) | 20% | 47 / 188 | 65.02 | 60.77 | 72.13 | **65.97** | 95 |
| **v1 Baseline** | Semi-Supervised (MedSAM v1) | 20% | 47 / 188 | 60.34 | 57.23 | 64.92 | **60.83** | 80 |
| **v1 Baseline** | Supervised (MiniUNet 2D) | 40% | 94 / 141 | 63.98 | 62.33 | 74.85 | **67.05** | 65 |
| **v1 Baseline** | Semi-Supervised (MedSAM v1) | 40% | 94 / 141 | 69.17 | 69.33 | 78.54 | **72.35** | 30 |
| **v1 Baseline** | Supervised (MiniUNet 2D) | 70% | 164 / 71 | 75.47 | 63.68 | 73.53 | **70.89** | 80 |
| **v1 Baseline** | Semi-Supervised (MedSAM v1) | 70% | 164 / 71 | 63.76 | 66.57 | 78.21 | **69.51** | 75 |
| **v1 Baseline** | Supervised (MiniUNet 2D) | 80% | 188 / 47 | 76.70 | 60.67 | 67.67 | **68.34** | 45 |
| **v1 Baseline** | Semi-Supervised (MedSAM v1) | 80% | 188 / 47 | 70.84 | 62.77 | 68.38 | **67.33** | 45 |
| **v2 Proposed** | **Supervised (SegResNet 2.5D)** | **40%** | **94 / 141** | **87.03** | **79.57** | **73.01** | **79.87** | **50** |
| **v2 Proposed** | **Semi-Supervised (MedSAM SoftBlend)** | **40%** | **94 / 141** | **87.63** | **80.24** | **73.94** | **80.60** | **30** |

---

### B. Evaluasi Menyeluruh Pipeline v2 (Validation vs Held-Out Test 50 Pasien)

Evaluasi ini menggunakan prosedur inferensi 3D pasien-level penuh dengan **4-view Test-Time Augmentation (TTA)** dan **3D Connected Component Post-processing**:

| Metrik Evaluasi | v2 Supervised (40% Label) | v2 Semi-Supervised + MedSAM (40% Label) | Delta (SSL vs SL) |
|---|:---:|:---:|:---:|
| **Validation 3D (No TTA)** | 79.27% | **80.14%** | **+0.87%** |
| **Validation 3D + 4-View TTA** | 79.87% | **80.60%** | **+0.73%** |
| - *Val Whole Tumor (WT)* | 87.03% | **87.63%** | **+0.60%** |
| - *Val Tumor Core (TC)* | 79.57% | **80.24%** | **+0.67%** |
| - *Val Enhancing Tumor (ET)* | 73.01% | **73.94%** | **+0.93%** |
| **Held-Out Test 3D + TTA (50 Pasien Asing)** | 75.98% | **76.93%** | **+0.95%** |
| - *Test Whole Tumor (WT)* | 84.51% | **84.60%** | **+0.09%** |
| - *Test Tumor Core (TC)* | 75.15% | **76.56%** | **+1.41%** |
| - *Test Enhancing Tumor (ET)* | 68.30% | **69.63%** | **+1.33%** |

---

## 3. Inovasi & Perbaikan Kritis dalam Pipeline v2

Keberhasilan menembus angka **80.60%** didasarkan pada penyelesaian 5 akar masalah utama:

### 1. Arsitektur 2.5D Multikanal (`SegResNetSegmentor`)
* Baseline v1 memakai irisan 2D murni (4 channel: T1, T1ce, T2, FLAIR) yang kehilangan kesinambungan spasial sepanjang sumbu-Z.
* Pipeline v2 memakai representasi **2.5D** dengan 12 channel per irisan (`z, z-1, z+1`), memberikan konteks irisan tetangga ke *backbone* MONAI SegResNet dengan residual encoder-decoder.

### 2. Loss Function Khusus Sub-region Bersarang (`SubregionDiceCELoss`)
* Klasifikasi standar 4-kelas (Background, NCR, ED, ET) rentan mengalami bias penalti antar kelas bersarang.
* `SubregionDiceCELoss` secara matematis memisahkan loss ke dalam 3 sub-region klinis BraTS:
  $$\text{WT} = \text{NCR} \cup \text{ED} \cup \text{ET}, \quad \text{TC} = \text{NCR} \cup \text{ET}, \quad \text{ET} = \text{ET}$$
  dengan bobot Dice terkalibrasi (`1.0 WT + 1.2 TC + 1.2 ET`), sehingga model memberikan perhatian ekstra pada sub-region tumor aktif yang kecil.

### 3. Perbaikan Fatal Preprocessing MedSAM Foundation Model
* **Akar Masalah di v1**: Meta SAM default melakukan normalisasi Mean/Std ImageNet (mengasumsikan data foto RGB), padahal model `medsam_vit_b.pth` (Bowang-Lab) dilatih dengan skala intensitas MRI murni `[0, 1]` dan kliping persentil `[0.5%, 99.5%]`.
* **Solusi di v2**: Irisan MRI diproses dengan kliping persentil `[0.5, 99.5]` pada jaringan otak (background nol dijaga murni 0), kemudian di-scale strictly `[0, 1]`. Pada irisan uji tunggal, perbaikan ini terbukti meningkatkan skor MedSAM dari `66.05%` menjadi **`94.79%` (+28.74%)**.

### 4. *Soft Probability Fusion* & Jaminan Batas Hierarki Anatomi (HVI = 0)
* Ali-alih menimpa prediksi model student secara kasar (*hard prompt replacement*), v2 menggunakan **Soft Fusion**:
  $$P_{\text{blend}} = \alpha \cdot P_{\text{student}} + (1 - \alpha) \cdot P_{\text{MedSAM}}, \quad \alpha = 0.65$$
* Dilengkapi mekanisme koreksi post-processing otomatis yang menjamin **Hierarchy Violation Index (HVI) selalu 0**:
  $$ET \subseteq TC \subseteq WT$$
  sehingga tidak ada anomali voxel Enhancing Tumor di luar Tumor Core.

### 5. Rekayasa Memori Dua Tahap (Two-Stage GPU Offloading)
* Segmentor student dan encoder ViT-B MedSAM berukuran besar untuk VRAM 4 GB (RTX 3050).
* Tahap 2a: Student melakukan TTA triage pada volume unlabeled → offload bobot student ke CPU + bersihkan cache VRAM.
* Tahap 2b: MedSAM dimuat secara independen di GPU murni untuk me-refine irisan terpilih → hapus MedSAM dari memori → muat kembali student ke GPU untuk tahap retraining (Tahap 3).
* Pipeline berjalan stabil tanpa Out-Of-Memory (OOM) dan tanpa spillover WDDM RAM.

---

## 4. Pembagian Dataset & Metodologi Evaluasi

Proses benchmarking mengikuti protokol standar BraTS 2019:
* **Total Pasien**: 335 volume MRI pasien.
* **Train Set**: 235 pasien.
  * Pada rasio 40%: **94 pasien berlabel** (supervisi penuh) dan **141 pasien tanpa label** (triage + pseudo-labeling MedSAM).
* **Validation Set**: **50 pasien** (digunakan untuk checkpointing model terbaik selama training).
* **Held-Out Test Set**: **50 pasien** (benar-benar diisolasi, dievaluasi hanya satu kali di akhir riset untuk membuktikan kemampuan generalisasi).

---

## 5. Ringkasan Positioning terhadap Paper SOTA

Berdasarkan komparasi literatur semi-supervised medical image segmentation (file `docs/hasil_ekstraksi_paper_ssl.xlsx`):
* Sebagian besar metode SOTA seperti **SemiSAM+** (85–86% pada BraTS 2020), **Semi-SwinUNETR** (85–88% pada BraTS 2021), dan **MSAM** berfokus pada volume 3D penuh dengan transformer besar (memerlukan GPU A100/V100 32–80 GB).
* Kontribusi dan keunikan (novelty) pendekatan kita terletak pada:
  1. **Hierarchical Anatomy Constraint (HVI)** yang digabungkan langsung dengan Foundation Model (MedSAM).
  2. **Efisiensi komputasi**: Mampu mencapai **80.60%** pada GPU kelas laptop (4 GB VRAM) melalui representasi 2.5D + *two-stage memory scheduling*.
  3. **Keandalan Pseudo-Label**: Shannon Entropy voxel masking + triage terkalibrasi (`tau_high=0.78, tau_low=0.40`) yang menyeimbangkan irisan berkeyakinan tinggi (*ACCEPT*) dan irisan batas tumor yang diperhalus MedSAM (*REFINE*).

---

## 6. Roadmap Langkah Berikutnya Menuju 85%–90%+

1. **Eksperimen Rasio Lanjutan**:
   * Menjalankan v2 pada **`ratio_80`** (188 pasien berlabel) untuk mengevaluasi batas atas performa model dengan lebih banyak data label.
   * Menjalankan v2 pada **`ratio_10`** dan **`ratio_20`** untuk menguji ketahanan model pada skenario data sangat minim (*extreme few-label*).
2. **Multi-Axis 2.5D Ensemble**:
   * Menggabungkan prediksi irisan aksial, koronal, dan sagital saat evaluasi 3D untuk memperhalus kontur ET di sumbu Z tanpa menambah beban training.
3. **Drafting Manuskrip Jurnal**:
   * Menyusun naskah paper: Metodologi, Tabel Hasil Eksperimen, Visualisasi Kontur MedSAM vs Ground Truth, dan Pembahasan Komparatif terhadap SOTA.
