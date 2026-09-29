"""
Training script for semi-supervised brain tumor segmentation.
Supports Mean Teacher framework with variable labeled ratios (10%, 20%, 50%, 100%).
"""

import os
import sys
import json
import argparse
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import SemiSupervisedFoundationSegmentor
from src.semi_supervised.mean_teacher import MeanTeacherFramework
from src.utils.metrics import evaluate_brats_subregions


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train semi-supervised brain tumor segmentor.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ratio", type=str, default="ratio_10", choices=["ratio_10", "ratio_20", "ratio_50", "ratio_100"], help="Labeled split ratio key")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load splits
    splits_file = "data/splits.json"
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found at {splits_file}. Run scripts/prepare_splits.py first.")

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    partition = splits['semi_supervised_partitions'][args.ratio]
    print(f"\nTraining with partition '{args.ratio}':")
    print(f"  Labeled patients  : {partition['num_labeled']}")
    print(f"  Unlabeled patients: {partition['num_unlabeled']}")

    cohort_dirs = [
        os.path.join(cfg['dataset']['root_dir'], c)
        for c in cfg['dataset']['cohorts']
    ]

    # Initialize Datasets
    labeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        patient_ids=partition['labeled_patients']
    )
    unlabeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=False,
        patient_ids=partition['unlabeled_patients'] if partition['num_unlabeled'] > 0 else partition['labeled_patients']
    )
    val_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        patient_ids=splits['val']
    )

    print(f"Dataset successfully loaded: {len(labeled_dataset)} labeled, {len(unlabeled_dataset)} unlabeled, {len(val_dataset)} validation.")

    # Initialize Models
    student_net = SemiSupervisedFoundationSegmentor(in_channels=4, num_classes=4)
    teacher_net = SemiSupervisedFoundationSegmentor(in_channels=4, num_classes=4)

    optimizer = torch.optim.AdamW(
        student_net.parameters(),
        lr=cfg['training']['learning_rate'],
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

    print("\nMean Teacher framework initialized.")
    print("Ready for semi-supervised training pipeline.")


if __name__ == '__main__':
    main()
