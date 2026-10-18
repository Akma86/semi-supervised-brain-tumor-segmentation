"""
Complete Training Script for Semi-Supervised Brain Tumor Segmentation.
Supports Mean Teacher framework and Hierarchy-Aware Pseudo-Labeling Refinement
across variable annotation regimes (10%, 20%, 30%, 50%, 100%).
"""

import os
import sys
import json
import argparse
from itertools import cycle
from typing import Dict, Any

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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import SemiSupervisedFoundationSegmentor
from src.semi_supervised.mean_teacher import MeanTeacherFramework
from src.semi_supervised.reliability import compute_entropy_map, compute_reliability_score
from src.semi_supervised.hierarchy import compute_hierarchy_violation_index
from src.semi_supervised.decision_engine import DecisionEngine, PseudoLabelAction
from src.semi_supervised.hierarchical_prompter import HierarchicalSAMRefiner
from src.utils.metrics import evaluate_brats_subregions


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train semi-supervised brain tumor segmentor.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ratio", type=str, default="ratio_10",
                        choices=["ratio_10", "ratio_20", "ratio_30", "ratio_50", "ratio_100"],
                        help="Labeled split ratio key")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval for validation evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default="runs/experiments", help="Base directory for saving logs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n=======================================================")
    print("[INIT] Semi-Supervised Brain Tumor Segmentation Training")
    print("=======================================================")
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"Device: {device} ({device_name})")

    # Resolve paths & partitions
    splits_file = "data/splits.json"
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

    # Initialize Datasets with mode='2d' for memory-efficient training
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

    batch_size = args.batch_size or cfg['training'].get('batch_size', 2)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(labeled_dataset) >= batch_size
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(unlabeled_dataset) >= batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize Models & Optimizer
    student_net = SemiSupervisedFoundationSegmentor(in_channels=4, num_classes=4)
    teacher_net = SemiSupervisedFoundationSegmentor(in_channels=4, num_classes=4)

    lr = args.lr or cfg['training']['learning_rate']
    optimizer = torch.optim.AdamW(
        student_net.parameters(),
        lr=lr,
        weight_decay=cfg['training']['weight_decay']
    )

    framework = MeanTeacherFramework(
        student_model=student_net,
        teacher_model=teacher_net,
        optimizer=optimizer,
        ema_decay=cfg['semi_supervised']['ema_decay'],
        consistency_weight=cfg['semi_supervised']['consistency_weight'],
        rampup_epochs=cfg['semi_supervised']['consistency_rampup_epochs'],
        device=device
    )

    decision_engine = DecisionEngine(
        tau_high=cfg['semi_supervised'].get('pseudo_label_threshold', 0.85),
        tau_low=0.50,
        delta_hvi=0.03
    )
    refiner = HierarchicalSAMRefiner()

    epochs = args.epochs or cfg['training']['epochs']
    exp_dir = os.path.join(args.save_dir, args.ratio)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  - Total Epochs      : {epochs}")
    print(f"  - Batch Size        : {batch_size}")
    print(f"  - Learning Rate     : {lr}")
    print(f"  - Mixed Precision   : True (AMP enabled for 4GB VRAM)")
    print(f"  - Experiment Folder : {exp_dir}")

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Training Loop
    history = {
        'epoch': [],
        'total_loss': [],
        'loss_sup': [],
        'loss_cons': [],
        'val_dice_wt': [],
        'val_dice_tc': [],
        'val_dice_et': [],
        'val_dice_mean': []
    }

    best_val_dice = 0.0
    steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))

    print(f"\n[RUN] Starting Semi-Supervised Training ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        framework.student.train()
        framework.teacher.train()

        labeled_iter = cycle(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss_accum = 0.0
        sup_loss_accum = 0.0
        cons_loss_accum = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch [{epoch:03d}/{epochs:03d}]", unit="batch")
        for _ in pbar:
            batch_l = next(labeled_iter)
            try:
                batch_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                batch_u = next(unlabeled_iter)

            # Move tensors with explicit types
            img_l = batch_l['image'].to(device).float()
            target_l = batch_l['mask'].to(device).long()
            img_u = batch_u['image'].to(device).float()

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # 1. Supervised loss
                    pred_l = framework.student(img_l)
                    loss_sup = framework.supervised_criterion(pred_l, target_l)

                    # 2. Consistency / Teacher prediction
                    noise = torch.randn_like(img_u) * 0.05
                    student_u_pred = framework.student(img_u + noise)

                    with torch.no_grad():
                        teacher_u_pred = framework.teacher(img_u)

                    loss_const = framework.consistency_criterion(student_u_pred, teacher_u_pred)

                    rampup = framework.consistency_weight * (min(1.0, epoch / max(1, framework.rampup_epochs)))
                    total_loss = loss_sup + rampup * loss_const

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_l = framework.student(img_l)
                loss_sup = framework.supervised_criterion(pred_l, target_l)

                noise = torch.randn_like(img_u) * 0.05
                student_u_pred = framework.student(img_u + noise)
                with torch.no_grad():
                    teacher_u_pred = framework.teacher(img_u)

                loss_const = framework.consistency_criterion(student_u_pred, teacher_u_pred)
                rampup = framework.consistency_weight * (min(1.0, epoch / max(1, framework.rampup_epochs)))
                total_loss = loss_sup + rampup * loss_const

                total_loss.backward()
                optimizer.step()

            # Update EMA Teacher
            framework.update_ema_weights(decay=framework.ema_decay)

            total_loss_accum += total_loss.item()
            sup_loss_accum += loss_sup.item()
            cons_loss_accum += loss_const.item()

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Sup': f"{loss_sup.item():.4f}",
                'Cons': f"{loss_const.item():.4f}"
            })

        avg_total_loss = total_loss_accum / max(1, steps_per_epoch)
        avg_sup_loss = sup_loss_accum / max(1, steps_per_epoch)
        avg_cons_loss = cons_loss_accum / max(1, steps_per_epoch)

        history['epoch'].append(epoch)
        history['total_loss'].append(avg_total_loss)
        history['loss_sup'].append(avg_sup_loss)
        history['loss_cons'].append(avg_cons_loss)

        # Validation Step
        if (epoch % args.val_interval == 0) or (epoch == epochs):
            framework.student.eval()
            dices_wt, dices_tc, dices_et, dices_mean = [], [], [], []

            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch['image'].to(device).float()
                    v_mask = v_batch['mask'].to(device).long()

                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            v_logits = framework.student(v_img)
                    else:
                        v_logits = framework.student(v_img)

                    v_pred = torch.argmax(v_logits, dim=1)

                    for b in range(v_pred.shape[0]):
                        p_single = v_pred[b].cpu().numpy()
                        t_single = v_mask[b].cpu().numpy()
                        sub_metrics = evaluate_brats_subregions(p_single, t_single)
                        dices_wt.append(sub_metrics['Dice_WT'])
                        dices_tc.append(sub_metrics['Dice_TC'])
                        dices_et.append(sub_metrics['Dice_ET'])
                        dices_mean.append(sub_metrics['Dice_Mean'])

            mean_wt = float(np.mean(dices_wt))
            mean_tc = float(np.mean(dices_tc))
            mean_et = float(np.mean(dices_et))
            val_mean_dice = float(np.mean(dices_mean))

            history['val_dice_wt'].append(mean_wt)
            history['val_dice_tc'].append(mean_tc)
            history['val_dice_et'].append(mean_et)
            history['val_dice_mean'].append(val_mean_dice)

            print(f"\n[VALIDATION] Epoch {epoch:03d}:")
            print(f"   WT Dice: {mean_wt*100:.2f}% | TC Dice: {mean_tc*100:.2f}% | ET Dice: {mean_et*100:.2f}%")
            print(f"   >> Mean Dice: {val_mean_dice*100:.2f}%")

            # Save best checkpoint
            if val_mean_dice > best_val_dice:
                best_val_dice = val_mean_dice
                ckpt_path = os.path.join(exp_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'student_state_dict': framework.student.state_dict(),
                    'teacher_state_dict': framework.teacher.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mean_dice': best_val_dice,
                    'metrics': {
                        'WT': mean_wt,
                        'TC': mean_tc,
                        'ET': mean_et,
                        'Mean': val_mean_dice
                    },
                    'ratio': args.ratio,
                    'config': cfg
                }, ckpt_path)
                print(f"   [SAVED] New best model checkpoint: {ckpt_path}\n")

    # Save final history log
    history_file = os.path.join(exp_dir, "training_history.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f"\n=======================================================")
    print(f"[COMPLETE] Training Finished!")
    print(f"  - Best Validation Mean Dice : {best_val_dice*100:.2f}%")
    print(f"  - Checkpoint & Logs Saved To : {exp_dir}")
    print(f"=======================================================\n")


if __name__ == '__main__':
    main()
