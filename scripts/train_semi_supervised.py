"""
Hierarchy-Aware Semi-Supervised Brain Tumor Segmentation Pipeline (NO TEACHER).
Implements pure Self-Training with Anatomical Constraint Verification and SAM-Med3D Refinement:

Alur Penelitian:
1. Warm-up 3D U-Net pada data berlabel (10%, 20%, 30%).
2. 3D U-Net memprediksi data unlabeled.
3. Hitung Reliability (Shannon Entropy) & Cek Hierarki Anatomis (ET ⊆ TC ⊆ WT).
4. Decision Engine Tri-Way Triage:
     • ACCEPT : Langsung sah jadi pseudo-label (weight 1.0).
     • REFINE : Diperbaiki oleh Hierarchical SAM Refiner coarse-to-fine WT -> TC -> ET (weight 0.8).
     • REJECT : Dibuang (noise/uncertainty tinggi, weight 0.0).
5. Retrain 3D U-Net dengan gabungan data labeled + pseudo-label terverifikasi.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import SemiSupervisedFoundationSegmentor as BrainTumorUNet
from src.models.segmentation_heads import DiceCELoss
from src.semi_supervised.reliability import evaluate_subregion_reliabilities, compute_entropy_map
from src.semi_supervised.hierarchy import compute_hierarchy_violation_index
from src.semi_supervised.decision_engine import DecisionEngine, PseudoLabelAction
from src.semi_supervised.hierarchical_prompter import HierarchicalSAMRefiner
from src.utils.metrics import evaluate_brats_subregions


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class PseudoLabeledDataset(Dataset):
    """Dataset combining original images with generated & verified pseudo-labels."""
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(description="Hierarchy-Aware Semi-Supervised Segmentation (Single U-Net + SAM Refiner)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ratio", type=str, default="ratio_10",
                        choices=["ratio_10", "ratio_20", "ratio_30", "ratio_40", "ratio_50", "ratio_70", "ratio_80", "ratio_100"],
                        help="Labeled split ratio key")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs on labeled data only (default: 10)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval for validation evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default="runs/experiments", help="Base directory for saving logs")
    parser.add_argument("--use_sam", type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True, help="Enable SAM foundation model refinement in Phase 2 (default: True)")
    parser.add_argument("--sam_version", type=str, default="medsam", choices=["medsam", "sam2", "sam1"], help="SAM model: medsam (Bowang-Lab ViT-B for MRI), sam2 (Meta SAM 2.1 Hiera), or sam1 (default: medsam)")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml", help="SAM 2 config path")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Path to SAM / MedSAM weights checkpoint")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", help="SAM 1 model type: vit_b, vit_l, vit_h (default: vit_b)")
    parser.add_argument("--tau_high", type=float, default=0.50, help="High reliability threshold for ACCEPT (default: 0.50)")
    parser.add_argument("--tau_low", type=float, default=0.15, help="Low reliability threshold for REJECT (default: 0.15)")
    parser.add_argument("--max_pseudo_ratio", type=float, default=None, help="Optional max ratio of pseudo-labels to labeled samples (e.g. 2.0 to prevent drowning in ratio_10)")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits JSON file (default: from config or data/splits.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n=======================================================")
    print("[FRAMEWORK] Hierarchy-Aware 3D U-Net + SAM Refiner Pipeline")
    print("=======================================================")
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"Device: {device} ({device_name})")
    print("Architecture: Single 3D U-Net (NO Teacher Network)")

    # Resolve paths & partitions
    splits_file = args.splits or cfg['dataset'].get('splits_file', 'data/splits.json')
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found at {splits_file}. Run scripts/prepare_splits.py first.")

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    partition = splits['semi_supervised_partitions'][args.ratio]
    print(f"\nAnnotation Regime [{args.ratio}]:")
    print(f"  - Labeled patients   : {partition['num_labeled']}")
    print(f"  - Unlabeled patients : {partition['num_unlabeled']}")
    print(f"  - Validation patients: {len(splits['val'])}")

    cohort_dirs = [
        os.path.join(cfg['dataset']['root_dir'], c)
        for c in cfg['dataset']['cohorts']
    ]

    # Initialize Datasets (2D slice mode with RAM caching for speed)
    labeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        mode='2d',
        cache_data=True,
        patient_ids=partition['labeled_patients']
    )
    unlabeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=False,
        mode='2d',
        cache_data=False,
        patient_ids=partition['unlabeled_patients'] if partition['num_unlabeled'] > 0 else partition['labeled_patients']
    )
    val_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        mode='2d',
        cache_data=True,
        patient_ids=splits['val']
    )

    batch_size = args.batch_size
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(labeled_dataset) >= batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Single 3D U-Net Model (No Teacher!)
    model = BrainTumorUNet(in_channels=4, num_classes=4).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Modules for Hierarchy, Reliability & SAM Refinement (Calibrated 4-class entropy scale)
    decision_engine = DecisionEngine(tau_high=args.tau_high, tau_low=args.tau_low, delta_hvi=0.03)

    # Resolve default SAM checkpoints if none provided explicitly
    sam_ckpt = args.sam_checkpoint
    if sam_ckpt is None:
        if args.sam_version == "medsam":
            default_medsam = "medsam_repo/work_dir/MedSAM/medsam_vit_b.pth"
            if os.path.exists(default_medsam):
                sam_ckpt = default_medsam
        elif args.sam_version == "sam2":
            default_sam2 = "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"
            if os.path.exists(default_sam2):
                sam_ckpt = default_sam2
        else:
            for default_ckpt in ["medsam_repo/work_dir/MedSAM/medsam_vit_b.pth", "checkpoints/medsam_vit_b.pth", "checkpoints/sam_vit_b_01ec64.pth"]:
                if os.path.exists(default_ckpt):
                    sam_ckpt = default_ckpt
                    break

    sam_refiner = HierarchicalSAMRefiner(
        checkpoint_path=sam_ckpt,
        use_sam=args.use_sam,
        sam_version=args.sam_version,
        sam2_config=args.sam2_config,
        model_type=args.sam_model_type,
        device=device
    )
    sam_status_str = f"Active [{sam_refiner.sam_version} on {device.upper()}]" if sam_refiner.has_sam else "Geometric Hierarchy Constraint Fallback"
    print(f"SAM Refiner Status: {sam_status_str}")

    save_base = args.save_dir if args.save_dir != "runs/semi_supervised" else cfg.get('logging', {}).get('save_dir', 'runs/semi_supervised')
    exp_dir = os.path.join(save_base, args.ratio)
    os.makedirs(exp_dir, exist_ok=True)

    epochs = args.epochs
    warmup_epochs = min(args.warmup_epochs, epochs // 2)
    best_val_dice = 0.0

    print(f"\nTraining Plan:")
    print(f"  - Phase 1 (Warmup on Labeled)       : Epoch 1 to {warmup_epochs}")
    print(f"  - Phase 2 (Triage & SAM Refinement) : Executed at Epoch {warmup_epochs}")
    print(f"  - Phase 3 (Retrain with Pseudo)     : Epoch {warmup_epochs + 1} to {epochs}")

    # =========================================================================
    # PHASE 1: Supervised Warm-Up on Labeled Patients (e.g. 10%)
    # =========================================================================
    print(f"\n>>> [PHASE 1] Starting Warm-Up on Labeled Data (Epoch 1 - {warmup_epochs})...")
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(labeled_loader, desc=f"Warmup [{epoch:02d}/{warmup_epochs:02d}]", unit="batch")
        for batch in pbar:
            imgs = batch['image'].to(device).float()
            masks = batch['mask'].to(device).long()

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # Validate warm-up baseline
    print("\n[VALIDATION] Warm-up Baseline Evaluation:")
    model.eval()
    warmup_dices = []
    with torch.no_grad():
        for v_batch in val_loader:
            v_imgs = v_batch['image'].to(device).float()
            v_masks = v_batch['mask'].to(device).long()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    v_logits = model(v_imgs)
            else:
                v_logits = model(v_imgs)
            v_preds = torch.argmax(v_logits, dim=1)
            for b in range(v_preds.shape[0]):
                m = evaluate_brats_subregions(v_preds[b].cpu().numpy(), v_masks[b].cpu().numpy())
                warmup_dices.append(m['Dice_Mean'])
    print(f"   >> Warm-up Mean Dice: {np.mean(warmup_dices)*100:.2f}%\n")

    # =========================================================================
    # PHASE 2: Predict Unlabeled -> Reliability & Hierarchy Check -> Decision Engine -> SAM Refine
    # =========================================================================
    print("=========================================================================")
    print(">>> [PHASE 2] Auditing 92 Unlabeled Cases with Decision Engine & SAM Refiner")
    print("=========================================================================")
    model.eval()
    pseudo_labeled_pool = []
    counts = {'ACCEPT': 0, 'REFINE': 0, 'REJECT': 0}

    unlabeled_loader_eval = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader_eval, desc="Triage & Refine", unit="patient"):
            img = batch['image'].to(device).float() # [1, 4, H, W]

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(img)
            else:
                logits = model(img)

            probs = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1) # [1, H, W]

            wt_mask = (pred_labels == 1) | (pred_labels == 2) | (pred_labels == 3)
            tc_mask = (pred_labels == 1) | (pred_labels == 3)
            et_mask = (pred_labels == 3)

            wt_voxels = int(wt_mask.sum().item())

            # 1. Compute Reliability
            reliabilities = evaluate_subregion_reliabilities(
                logits=logits,
                wt_mask=wt_mask.float(),
                tc_mask=tc_mask.float(),
                et_mask=et_mask.float()
            )

            # 2. Check Hierarchy Constraints (ET ⊆ TC ⊆ WT)
            hierarchy_stats = compute_hierarchy_violation_index(
                wt_mask=wt_mask[0].cpu().numpy(),
                tc_mask=tc_mask[0].cpu().numpy(),
                et_mask=et_mask[0].cpu().numpy()
            )

            # 3. Decision Engine: ACCEPT / REFINE / REJECT
            decision = decision_engine.evaluate(
                reliabilities=reliabilities,
                hierarchy_stats=hierarchy_stats,
                wt_voxel_count=wt_voxels
            )

            action = decision['action']
            weight = decision['weight']
            score = decision.get('score', 0.0)

            if action == PseudoLabelAction.ACCEPT:
                counts['ACCEPT'] += 1
                pseudo_labeled_pool.append({
                    'image': batch['image'].squeeze(0),
                    'mask': pred_labels[0].cpu(),
                    'weight': 1.0,
                    'score': score,
                    'patient_id': batch['patient_id'][0]
                })

            elif action == PseudoLabelAction.REFINE:
                counts['REFINE'] += 1
                # SAM-Med3D Coarse-to-Fine Refinement (WT -> TC -> ET)
                refined = sam_refiner.refine_hierarchical(
                    image=batch['image'][0],
                    raw_wt=wt_mask[0].cpu().float(),
                    raw_tc=tc_mask[0].cpu().float(),
                    raw_et=et_mask[0].cpu().float()
                )
                pseudo_labeled_pool.append({
                    'image': batch['image'].squeeze(0),
                    'mask': refined['assembled_seg'].clone().detach() if isinstance(refined['assembled_seg'], torch.Tensor) else torch.from_numpy(refined['assembled_seg']),
                    'weight': 0.8,
                    'score': score,
                    'patient_id': batch['patient_id'][0]
                })

            else:
                counts['REJECT'] += 1

    total_unlabeled = len(unlabeled_dataset)
    print("\n[TRIAGE SUMMARY RESULTS]:")
    print(f"  • 🟢 ACCEPT (Direct Pseudo-Label) : {counts['ACCEPT']:2d} / {total_unlabeled} ({counts['ACCEPT']/total_unlabeled*100:.1f}%)")
    print(f"  • 🟡 REFINE (Refined by SAM-Med3D): {counts['REFINE']:2d} / {total_unlabeled} ({counts['REFINE']/total_unlabeled*100:.1f}%)")
    print(f"  • 🔴 REJECT (Discarded Noise)     : {counts['REJECT']:2d} / {total_unlabeled} ({counts['REJECT']/total_unlabeled*100:.1f}%)")
    print(f"  >> Total Validated Pseudo-Labels  : {len(pseudo_labeled_pool)} cases generated.")

    # Optional Smart Balancing: Prevent Pseudo-Label Drowning in Low-Data Regimes
    if args.max_pseudo_ratio is not None and len(pseudo_labeled_pool) > 0:
        max_allowed = max(1, int(len(labeled_dataset) * args.max_pseudo_ratio))
        if len(pseudo_labeled_pool) > max_allowed:
            pseudo_labeled_pool.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            print(f"  ⚡ [BALANCER] Retaining Top-{max_allowed} highest-confidence pseudo-labels (Ratio {args.max_pseudo_ratio}x labeled count: {len(labeled_dataset)} cases) to prevent confirmation bias drift.")
            pseudo_labeled_pool = pseudo_labeled_pool[:max_allowed]
    print(f"  >> Final Active Pseudo-Label Pool: {len(pseudo_labeled_pool)} cases added to training pool!\n")

    # =========================================================================
    # PHASE 3: Retrain 3D U-Net on Labeled + Verified Pseudo-Labels
    # =========================================================================
    print(f">>> [PHASE 3] Retraining 3D U-Net on Augmented Dataset (Epoch {warmup_epochs + 1} - {epochs})...")

    # Combine labeled dataset items with verified pseudo-labels (clean matching keys for DataLoader collate)
    augmented_samples = []
    for i in range(len(labeled_dataset)):
        sample = labeled_dataset[i]
        augmented_samples.append({
            'image': sample['image'],
            'mask': sample['mask'],
            'weight': 1.0,
            'patient_id': sample['patient_id']
        })
    for p_item in pseudo_labeled_pool:
        augmented_samples.append({
            'image': p_item['image'],
            'mask': p_item['mask'],
            'weight': p_item['weight'],
            'patient_id': p_item['patient_id']
        })

    augmented_dataset = PseudoLabeledDataset(augmented_samples)
    augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
    print(f"Enriched Dataset Size: {len(augmented_dataset)} samples (Original: {len(labeled_dataset)})\n")

    for epoch in range(warmup_epochs + 1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(augmented_loader, desc=f"Retrain [{epoch:02d}/{epochs:02d}]", unit="batch")
        for batch in pbar:
            imgs = batch['image'].to(device).float()
            masks = batch['mask'].to(device).long()
            weights = batch['weight'].to(device).float()

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    # Rigorous sample-wise loss weighting (no batch bleeding)
                    loss = criterion(logits, masks, sample_weights=weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, masks, sample_weights=weights)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Periodic Validation
        if (epoch % args.val_interval == 0) or (epoch == epochs):
            model.eval()
            dices_wt, dices_tc, dices_et, dices_mean = [], [], [], []

            with torch.no_grad():
                for v_batch in val_loader:
                    v_imgs = v_batch['image'].to(device).float()
                    v_masks = v_batch['mask'].to(device).long()

                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            v_logits = model(v_imgs)
                    else:
                        v_logits = model(v_imgs)

                    v_preds = torch.argmax(v_logits, dim=1)
                    for b in range(v_preds.shape[0]):
                        m = evaluate_brats_subregions(v_preds[b].cpu().numpy(), v_masks[b].cpu().numpy())
                        dices_wt.append(m['Dice_WT'])
                        dices_tc.append(m['Dice_TC'])
                        dices_et.append(m['Dice_ET'])
                        dices_mean.append(m['Dice_Mean'])

            mean_wt = float(np.mean(dices_wt))
            mean_tc = float(np.mean(dices_tc))
            mean_et = float(np.mean(dices_et))
            mean_dice = float(np.mean(dices_mean))

            print(f"\n[VALIDATION] Epoch {epoch:03d}:")
            print(f"   WT Dice: {mean_wt*100:.2f}% | TC Dice: {mean_tc*100:.2f}% | ET Dice: {mean_et*100:.2f}%")
            print(f"   >> Mean Dice: {mean_dice*100:.2f}%")

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                ckpt_path = os.path.join(exp_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mean_dice': best_val_dice,
                    'metrics': {'WT': mean_wt, 'TC': mean_tc, 'ET': mean_et, 'Mean': mean_dice},
                    'ratio': args.ratio,
                    'config': cfg
                }, ckpt_path)
                print(f"   [SAVED] New best model checkpoint: {ckpt_path}\n")

    # Save final metrics summary for master reporting
    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ratio': args.ratio,
            'best_mean_dice': best_val_dice,
            'epochs': epochs,
            'warmup_epochs': warmup_epochs,
            'checkpoint': os.path.join(exp_dir, "best_model.pth")
        }, f, indent=2)

    print("\n=======================================================")
    print("[COMPLETE] Pipeline Finished Successfully!")
    print(f"  • Best Validation Mean Dice : {best_val_dice*100:.2f}%")
    print(f"  • Model Checkpoint Saved To : {exp_dir}")
    print(f"  • Metrics Summary Saved To  : {summary_path}")
    print("=======================================================\n")


if __name__ == '__main__':
    main()
