"""
Supervised Benchmark Training for Brain Tumor Segmentation (BraTS-Africa & BraTS 2019).
Uses MONAI SegResNet (or baseline CNN) with 2.5D multi-slice context, dynamic volume slicing,
direct BraTS subregion loss, cosine annealing, TTA, and full 3D volumetric evaluation.
"""

import argparse
import json
import os
import sys
import time

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import build_segmentor
from src.utils.training_helpers import (
    build_loss_fn,
    build_train_transform,
    evaluate_and_checkpoint,
    finalize_val_and_test,
    resolve_dataset_config,
    train_one_epoch,
    wrap_multi_slice_dataset,
)


def str_to_bool(val: str) -> bool:
    return str(val).lower() in ('true', '1', 'yes')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Supervised Brain Tumor Segmentation Training.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--ratio", type=str, default="ratio_10",
                        help="Partition key in splits JSON (e.g., ratio_10, ratio_20, ratio_40, ratio_80)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/supervised")
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--arch", type=str, default="segresnet", choices=["segresnet", "unet_mini"])
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--slice_mode", type=str, default="2.5d", choices=["2d", "2.5d"])
    parser.add_argument("--slices_per_volume", type=int, default=8)
    parser.add_argument("--loss", type=str, default="subregion_dice_ce", choices=["subregion_dice_ce", "dice_ce"])
    parser.add_argument("--tta", type=str_to_bool, default=True)
    parser.add_argument("--val_tta", type=str_to_bool, default=False)
    parser.add_argument("--val_stride", type=int, default=2)
    parser.add_argument("--postprocess", type=str_to_bool, default=True)
    parser.add_argument("--augment", type=str_to_bool, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    cohort_dirs, splits_path, modalities = resolve_dataset_config(cfg, args.splits)

    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file missing: {splits_path}")
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    partitions = splits['semi_supervised_partitions']
    if args.ratio not in partitions:
        raise ValueError(f"Ratio '{args.ratio}' not in {list(partitions.keys())}")

    labeled_ids = partitions[args.ratio]['labeled_patients']
    val_ids = splits['val']

    exp_dir = os.path.join(args.save_dir, args.ratio)
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "best_model.pth")

    print(f"\n[Supervised Training] ratio={args.ratio} ({len(labeled_ids)} train, {len(val_ids)} val) | "
          f"arch={args.arch} | mode={args.slice_mode} | device={device}")

    train_base = BraTSDataset(
        data_dir=cohort_dirs,
        modalities=modalities,
        is_labeled=True,
        patient_ids=labeled_ids,
        transform=build_train_transform(args.augment),
        mode=args.slice_mode,
        cache_data=True,
    )
    train_dataset = wrap_multi_slice_dataset(train_base, args.slices_per_volume)

    val_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        modalities=modalities,
        is_labeled=True,
        patient_ids=val_ids,
        mode=args.slice_mode,
        cache_data=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    in_channels = 12 if args.slice_mode == "2.5d" else 4
    model = build_segmentor(
        arch=args.arch,
        in_channels=in_channels,
        num_classes=4,
        base_filters=args.base_filters,
        dropout_p=0.1,
    ).to(device)
    criterion = build_loss_fn(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_val_dice = -1.0
    best_epoch = 0
    best_metrics = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            desc=f"Sup [{epoch:02d}/{args.epochs:02d}]",
        )
        scheduler.step()

        if (epoch % args.val_interval == 0) or (epoch == args.epochs):
            prev_best = best_val_dice
            best_val_dice, best_metrics, _ = evaluate_and_checkpoint(
                model=model,
                val_dataset=val_dataset,
                device=device,
                slice_mode=args.slice_mode,
                val_batch_size=16,
                use_tta=args.val_tta,
                postprocess=args.postprocess,
                best_val_dice=best_val_dice,
                best_metrics=best_metrics,
                save_path=save_path,
                epoch=epoch,
                optimizer=optimizer,
                prefix=f"Sup Loss={avg_loss:.4f} | Val 3D",
                slice_stride=args.val_stride,
            )
            if best_val_dice > prev_best:
                best_epoch = epoch

    final_val_metrics, test_metrics = finalize_val_and_test(
        model=model,
        save_path=save_path,
        val_dataset=val_dataset,
        cohort_dirs=cohort_dirs,
        modalities=modalities,
        splits=splits,
        device=device,
        slice_mode=args.slice_mode,
        use_tta=args.tta,
        val_tta=args.val_tta,
        postprocess=args.postprocess,
        best_val_dice=best_val_dice,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
    )

    elapsed = time.time() - start_time
    summary = {
        'ratio': args.ratio,
        'is_supervised': True,
        'arch': args.arch,
        'slice_mode': args.slice_mode,
        'tta': args.tta,
        'postprocess': args.postprocess,
        'best_mean_dice': float(final_val_metrics['Mean']),
        'best_val_notta_dice': float(best_val_dice),
        'best_epoch': best_epoch,
        'metrics': final_val_metrics,
        'test_metrics': test_metrics,
        'epochs': args.epochs,
        'elapsed_sec': round(elapsed, 2),
        'checkpoint': save_path,
    }
    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[Supervised Done in {elapsed:.1f}s] Best Mean Dice @ Ep {best_epoch}: {final_val_metrics['Mean']*100:.2f}% "
        f"(WT={final_val_metrics['WT']*100:.2f}%, TC={final_val_metrics['TC']*100:.2f}%, ET={final_val_metrics['ET']*100:.2f}%)"
    )


if __name__ == '__main__':
    main()
