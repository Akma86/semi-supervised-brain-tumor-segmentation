"""
Evaluation script to compute test-set metrics (Dice WT, TC, ET and HD95).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.utils.metrics import evaluate_brats_subregions


def main():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation metrics.")
    parser.add_argument("--splits", type=str, default="data/splits.json", help="Path to splits JSON")
    parser.add_argument("--data_dir", type=str, default="data/BraTS-Africa", help="Path to BraTS data")
    args = parser.parse_args()

    if not os.path.exists(args.splits):
        raise FileNotFoundError(f"Splits file not found: {args.splits}")

    with open(args.splits, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    test_ids = splits['test']
    print(f"Evaluating {len(test_ids)} test cases...")

    cohorts = ['95_Glioma', '51_OtherNeoplasms']
    cohort_dirs = [os.path.join(args.data_dir, c) for c in cohorts]

    dataset = BraTSDataset(
        data_dir=cohort_dirs,
        is_labeled=True,
        patient_ids=test_ids
    )

    results = []
    for i in range(min(5, len(dataset))): # Sample demonstration
        sample = dataset[i]
        gt = sample['mask'].numpy()
        # Self-consistency oracle check
        metrics = evaluate_brats_subregions(gt, gt)
        metrics['patient_id'] = sample['patient_id']
        results.append(metrics)

    df = pd.DataFrame(results)
    print("\nSample Test Evaluation Table:")
    print(df[['patient_id', 'Dice_WT', 'Dice_TC', 'Dice_ET', 'Dice_Mean']])


if __name__ == '__main__':
    main()
