"""
Combined Master Benchmark Runner: Pure Supervised vs. MedSAM Semi-Supervised.
Runs all 4 annotation regimes sequentially:
  - ratio_10: 10 labeled patients
  - ratio_20: 20 labeled patients
  - ratio_40: 41 labeled patients
  - ratio_80: 82 labeled patients

Executes:
  Stage 1: Pure Supervised Baseline (Single 3D U-Net on labeled only)
  Stage 2: MedSAM Semi-Supervised Pipeline (Warmup + Calibrated Triage + MedSAM Refinement + Retrain)
  Stage 3: Generates unified comparative benchmark table with absolute gain (Δ)

Usage:
    python scripts/run_combined_benchmarks.py
    python scripts/run_combined_benchmarks.py --ratios ratio_10 ratio_20 ratio_40 ratio_80 --epochs 200 --warmup_epochs 40
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import timedelta

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_command_stream(cmd: list, log_file: str) -> int:
    """Executes subprocess and streams output to console and log simultaneously."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Master Unified Runner: Supervised vs. Semi-Supervised")
    parser.add_argument("--ratios", nargs="+", default=["ratio_10", "ratio_20", "ratio_40", "ratio_80"],
                        help="List of ratio splits to run sequentially (default: ratio_10 ratio_20 ratio_40 ratio_80)")
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs per experiment (default: 200)")
    parser.add_argument("--warmup_epochs", type=int, default=40, help="Warmup epochs for semi-supervised (default: 40)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--sam_version", type=str, default="medsam", choices=["medsam", "sam2", "sam1"],
                        help="SAM foundation model: medsam, sam2, or sam1 (default: medsam)")
    parser.add_argument("--max_pseudo_ratio", type=float, default=2.0,
                        help="Maximum ratio of pseudo-labels to labeled samples (default: 2.0 to prevent drowning in ratio_10)")
    parser.add_argument("--skip_supervised", action="store_true", help="Skip pure supervised stage")
    parser.add_argument("--skip_semi_supervised", action="store_true", help="Skip semi-supervised stage")
    args = parser.parse_args()

    total_start_time = time.time()

    print("\n" + "=" * 80)
    print(" 🚀 UNIFIED RESEARCH BENCHMARK: SUPERVISED BASELINE VS. MEDSAM SEMI-SUPERVISED")
    print("=" * 80)
    print(f" Regimes to Run   : {args.ratios}")
    print(f" Epochs           : {args.epochs} (Semi-Sup Warmup: {args.warmup_epochs})")
    print(f" Foundation Model : {args.sam_version.upper()}")
    print(f" Pseudo Cap Ratio : {args.max_pseudo_ratio}x labeled count")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1: PURE SUPERVISED BASELINES
    # =========================================================================
    if not args.skip_supervised:
        print("\n" + "#" * 80)
        print(" [STAGE 1/2] RUNNING PURE SUPERVISED BASELINES (LABELED-ONLY)")
        print("#" * 80)

        for idx, ratio in enumerate(args.ratios, start=1):
            print(f"\n>>> [SUPERVISED {idx}/{len(args.ratios)}] Training {ratio.upper()}...")
            log_path = os.path.join("runs", "supervised", ratio, "train.log")
            cmd = [
                sys.executable,
                "scripts/train_supervised.py",
                "--ratio", ratio,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--val_interval", "5",
                "--save_dir", "runs/supervised"
            ]
            ret = run_command_stream(cmd, log_path)
            if ret != 0:
                print(f"❌ Error during supervised training of {ratio} (Code {ret})")

    # =========================================================================
    # STAGE 2: SEMI-SUPERVISED PIPELINE (MEDSAM / SAM 2)
    # =========================================================================
    if not args.skip_semi_supervised:
        print("\n" + "#" * 80)
        print(f" [STAGE 2/2] RUNNING SEMI-SUPERVISED PIPELINE WITH {args.sam_version.upper()}")
        print("#" * 80)

        for idx, ratio in enumerate(args.ratios, start=1):
            print(f"\n>>> [SEMI-SUPERVISED {idx}/{len(args.ratios)}] Training {ratio.upper()}...")
            log_path = os.path.join("runs", "experiments", ratio, "train.log")
            cmd = [
                sys.executable,
                "scripts/train_semi_supervised.py",
                "--ratio", ratio,
                "--epochs", str(args.epochs),
                "--warmup_epochs", str(args.warmup_epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--val_interval", "5",
                "--sam_version", args.sam_version,
                "--save_dir", "runs/experiments"
            ]
            if args.max_pseudo_ratio is not None:
                cmd.extend(["--max_pseudo_ratio", str(args.max_pseudo_ratio)])

            ret = run_command_stream(cmd, log_path)
            if ret != 0:
                print(f"❌ Error during semi-supervised training of {ratio} (Code {ret})")

    total_duration = str(timedelta(seconds=int(time.time() - total_start_time)))

    # =========================================================================
    # STAGE 3: COMPILE MASTER COMPARATIVE REPORT
    # =========================================================================
    with open("data/splits.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    comparison_results = {}
    for ratio in args.ratios:
        part = splits_data['semi_supervised_partitions'].get(ratio, {})
        n_labeled = part.get('num_labeled', '?')
        n_unlabeled = part.get('num_unlabeled', '?')

        # Load Supervised Summary
        sup_path = os.path.join("runs", "supervised", ratio, "metrics_summary.json")
        sup_dice = 0.0
        sup_metrics = {}
        if os.path.exists(sup_path):
            with open(sup_path, "r", encoding="utf-8") as f:
                sup_data = json.load(f)
                sup_dice = sup_data.get("best_mean_dice", 0.0)
                sup_metrics = sup_data.get("metrics", {})

        # Load Semi-Supervised Summary
        semi_path = os.path.join("runs", "experiments", ratio, "metrics_summary.json")
        semi_dice = 0.0
        semi_metrics = {}
        if os.path.exists(semi_path):
            with open(semi_path, "r", encoding="utf-8") as f:
                semi_data = json.load(f)
                semi_dice = semi_data.get("best_mean_dice", 0.0)
                semi_metrics = semi_data.get("metrics", {})

        gain = semi_dice - sup_dice

        comparison_results[ratio] = {
            "labeled_patients": n_labeled,
            "unlabeled_patients": n_unlabeled,
            "supervised_mean_dice": sup_dice,
            "supervised_metrics": sup_metrics,
            "semi_supervised_mean_dice": semi_dice,
            "semi_supervised_metrics": semi_metrics,
            "absolute_gain": gain
        }

    # Save Unified Master Comparative Report
    master_comp_file = "runs/master_comparative_table.json"
    with open(master_comp_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "sam_version": args.sam_version,
            "comparison": comparison_results
        }, f, indent=2)

    # Print Final Grand Summary Table
    print("\n" + "=" * 88)
    print(" 🏁 MASTER COMPARATIVE BENCHMARK: SUPERVISED BASELINE VS. SEMI-SUPERVISED")
    print("=" * 88)
    print(f" Total Duration : {total_duration}")
    print(f" Model Used     : {args.sam_version.upper()}")
    print(f" Report Saved   : {master_comp_file}")
    print("-" * 88)
    header = f"{'Regime':<10} | {'Labeled':<9} | {'Supervised Dice':<17} | {'Semi-Supervised':<17} | {'Gain (Δ)':<10}"
    print(header)
    print("-" * 88)

    for ratio, d in comparison_results.items():
        sup_str = f"{d['supervised_mean_dice']*100:.2f}%" if d['supervised_mean_dice'] > 0 else "N/A"
        semi_str = f"{d['semi_supervised_mean_dice']*100:.2f}%" if d['semi_supervised_mean_dice'] > 0 else "N/A"
        gain_val = d['absolute_gain'] * 100
        gain_str = f"+{gain_val:.2f}%" if gain_val >= 0 else f"{gain_val:.2f}%"
        print(f"{ratio:<10} | {d['labeled_patients']:<2} cases   | {sup_str:<17} | {semi_str:<17} | {gain_str:<10}")

    print("=" * 88 + "\n")

    # Automatically Export to Styled Excel Workbook
    try:
        from scripts.export_to_excel import build_excel_report
        excel_out = "runs/benchmark_master_results.xlsx"
        build_excel_report(excel_out)
        print(f"📊 [EXCEL] Complete benchmark report exported to: {excel_out}\n")
    except Exception as e:
        print(f"Warning: Excel export failed: {e}")


if __name__ == "__main__":
    main()
