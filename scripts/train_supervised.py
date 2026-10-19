"""
Pure Supervised Benchmark Training for Brain Tumor Segmentation (BraTS-Africa):
Trains a single 3D U-Net using ONLY the labeled partition (without semi-supervised pseudo-labels).
Serves as the empirical lower-bound baseline across annotation regimes:
  - ratio_10:  10 labeled patients
  - ratio_20:  20 labeled patients
  - ratio_40:  41 labeled patients
  - ratio_100: 102 labeled patients (Fully-Supervised Upper Bound)

Usage:
    python scripts/train_supervised.py --ratio ratio_10 --epochs 200
    python scripts/train_supervised.py --ratio ratio_20 --epochs 200
    python scripts/train_supervised.py --ratio ratio_40 --epochs 200
    python scripts/train_supervised.py --ratio ratio_100 --epochs 200
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import SemiSupervisedFoundationSegmentor as BrainTumorUNet
from src.models.segmentation_heads import DiceCELoss
from src.utils.metrics import evaluate_brats_subregions


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Pure Supervised Training Benchmark (Single 3D U-Net)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ratio", type=str, default="ratio_10",
                        choices=["ratio_10", "ratio_20", "ratio_30", "ratio_40", "ratio_50", "ratio_80", "ratio_100"],
                        help="Labeled split ratio key")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval for validation (default: 5)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default="runs/supervised", help="Base directory for saving checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'

    print("\n" + "=" * 65)
    print(" [BENCHMARK] PURE SUPERVISED TRAINING (LABELED-ONLY BASELINE)")
    print("=" * 65)
    print(f" Device       : {device} ({device_name})")
    print(f" Architecture : Single 3D U-Net")
    print(f" Regime       : {args.ratio}")
    print(f" Epochs       : {args.epochs}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" Learning Rate: {args.lr}")

    # Resolve paths & partitions
    splits_file = "data/splits.json"
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found at {splits_file}. Run scripts/prepare_splits.py first.")

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    partition = splits['semi_supervised_partitions'][args.ratio]
    print(f"\nDataset Statistics [{args.ratio}]:")
    print(f"  - Labeled patients   : {partition['num_labeled']}")
    print(f"  - Validation patients: {len(splits['val'])}")
    print(f"  - Supervised status  : 100% Ground-Truth Supervision (No Pseudo-Labels)")

    cohort_dirs = [
        os.path.join(cfg['dataset']['root_dir'], c)
        for c in cfg['dataset']['cohorts']
    ]

    # Initialize Datasets (2D slice mode with RAM caching for maximum GPU throughput)
    labeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        mode='2d',
        cache_data=True,
        patient_ids=partition['labeled_patients']
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

    # Initialize 3D U-Net model and loss
    model = BrainTumorUNet(in_channels=4, num_classes=4).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Setup directories
    exp_dir = os.path.join(args.save_dir, args.ratio)
    os.makedirs(exp_dir, exist_ok=True)

    best_val_dice = 0.0
    best_metrics = {}
    best_epoch = 0
    start_time = time.time()

    print(f"\n>>> Starting Supervised Training ({args.epochs} Epochs)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(labeled_loader, desc=f"Epoch [{epoch:03d}/{args.epochs:03d}]", unit="batch")
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

        # Periodic Validation
        if (epoch % args.val_interval == 0) or (epoch == args.epochs):
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
                best_epoch = epoch
                best_metrics = {'WT': mean_wt, 'TC': mean_tc, 'ET': mean_et, 'Mean': mean_dice}
                ckpt_path = os.path.join(exp_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mean_dice': best_val_dice,
                    'metrics': best_metrics,
                    'ratio': args.ratio,
                    'is_supervised': True
                }, ckpt_path)
                print(f"   🔥 [CHECKPOINT] New best supervised validation Dice ({best_val_dice*100:.2f}%) saved to: {ckpt_path}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    print("\n" + "=" * 65)
    print(" [SUPERVISED TRAINING COMPLETE]")
    print("=" * 65)
    print(f" Total Duration            : {minutes:02d}m {seconds:02d}s")
    print(f" Best Supervised Mean Dice : {best_val_dice*100:.2f}% (Epoch {best_epoch})")
    if best_metrics:
        print(f"   • WT Dice: {best_metrics.get('WT', 0.0)*100:.2f}%")
        print(f"   • TC Dice: {best_metrics.get('TC', 0.0)*100:.2f}%")
        print(f"   • ET Dice: {best_metrics.get('ET', 0.0)*100:.2f}%")
    print(f" Checkpoint Saved          : {os.path.join(exp_dir, 'best_model.pth')}")
    print("=" * 65 + "\n")

    # Save summary json
    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ratio': args.ratio,
            'is_supervised': True,
            'best_mean_dice': best_val_dice,
            'best_epoch': best_epoch,
            'metrics': best_metrics,
            'epochs': args.epochs,
            'checkpoint': os.path.join(exp_dir, "best_model.pth")
        }, f, indent=2)


if __name__ == "__main__":
    main()
