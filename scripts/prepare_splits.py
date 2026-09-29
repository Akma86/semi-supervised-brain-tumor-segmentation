"""
Prepares reproducible dataset splits for the BraTS-Africa cohort.
Partitions patients into Train (70%), Validation (15%), and Test (15%) subsets,
and creates semi-supervised training partitions (10%, 20%, 50%, 100% labeled ratios).
"""

import os
import json
import argparse
import random
from typing import Dict, List


def discover_patients(data_dir: str) -> Dict[str, List[str]]:
    """Scans and discovers patient folder IDs in each cohort."""
    cohorts = ['95_Glioma', '51_OtherNeoplasms']
    patients_by_cohort = {}

    for cohort in cohorts:
        cohort_dir = os.path.join(data_dir, cohort)
        if not os.path.exists(cohort_dir):
            print(f"Warning: Directory {cohort_dir} does not exist. Skipping.")
            continue
        patient_ids = sorted([
            f for f in os.listdir(cohort_dir)
            if os.path.isdir(os.path.join(cohort_dir, f)) and f.startswith("BraTS-SSA")
        ])
        patients_by_cohort[cohort] = patient_ids
        print(f"Discovered {len(patient_ids)} patients in cohort: {cohort}")

    return patients_by_cohort


def create_splits(
    patients_by_cohort: Dict[str, List[str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict:
    """Creates stratified train/val/test splits maintaining cohort distribution."""
    random.seed(seed)

    train_all = []
    val_all = []
    test_all = []

    for cohort, pids in patients_by_cohort.items():
        shuffled = list(pids)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test

        train_pids = shuffled[:n_train]
        val_pids = shuffled[n_train:n_train + n_val]
        test_pids = shuffled[n_train + n_val:]

        train_all.extend(train_pids)
        val_all.extend(val_pids)
        test_all.extend(test_pids)

    # Semi-supervised partitions for train set
    random.shuffle(train_all)
    n_train_total = len(train_all)

    ratios = [0.10, 0.20, 0.50, 1.00]
    semi_supervised_partitions = {}

    for r in ratios:
        n_labeled = max(1, int(round(n_train_total * r)))
        labeled_ids = train_all[:n_labeled]
        unlabeled_ids = train_all[n_labeled:]
        semi_supervised_partitions[f"ratio_{int(r*100)}"] = {
            'labeled_ratio': r,
            'num_labeled': len(labeled_ids),
            'num_unlabeled': len(unlabeled_ids),
            'labeled_patients': labeled_ids,
            'unlabeled_patients': unlabeled_ids
        }

    splits = {
        'seed': seed,
        'summary': {
            'total_patients': sum(len(v) for v in patients_by_cohort.values()),
            'train_count': len(train_all),
            'val_count': len(val_all),
            'test_count': len(test_all)
        },
        'train': sorted(train_all),
        'val': sorted(val_all),
        'test': sorted(test_all),
        'semi_supervised_partitions': semi_supervised_partitions
    }

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate BraTS-Africa dataset splits.")
    parser.add_argument("--data_dir", type=str, default="data/BraTS-Africa", help="Path to BraTS-Africa directory")
    parser.add_argument("--output_path", type=str, default="data/splits.json", help="Path to output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Scanning patients in: {args.data_dir}")
    patients = discover_patients(args.data_dir)

    splits = create_splits(patients, seed=args.seed)
    print("\nDataset Split Summary:")
    print(f"  Train : {splits['summary']['train_count']} cases")
    print(f"  Val   : {splits['summary']['val_count']} cases")
    print(f"  Test  : {splits['summary']['test_count']} cases")
    for k, v in splits['semi_supervised_partitions'].items():
        print(f"  {k:10}: {v['num_labeled']} labeled | {v['num_unlabeled']} unlabeled")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplits successfully saved to: {args.output_path}")


if __name__ == '__main__':
    main()
