"""
Unified Master Benchmark Runner for BraTS 2019:
Pure Supervised vs. MedSAM Semi-Supervised.

Executes:
  Stage 1: Pure Supervised Baseline on BraTS 2019 (labeled only)
  Stage 2: MedSAM Semi-Supervised Pipeline on BraTS 2019 (Warmup + Triage + MedSAM Pseudo + Retrain)
  Stage 3: Compiles unified comparative report & Excel for BraTS 2019

Usage:
  python scripts/run_brats2019_benchmarks.py --ratios ratio_10 ratio_20 --epochs 100
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
    """Executes subprocess and streams output to console and log file."""
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
    parser = argparse.ArgumentParser(description="Master Unified Runner for BraTS 2019")
    parser.add_argument("--ratios", nargs="+", default=["ratio_10", "ratio_20", "ratio_40", "ratio_70", "ratio_80"],
                        choices=["ratio_10", "ratio_20", "ratio_30", "ratio_40", "ratio_50", "ratio_70", "ratio_80", "ratio_100"],
                        help="List of ratios to benchmark sequentially")
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs (default: 150)")
    parser.add_argument("--warmup_epochs", type=int, default=30, help="Warmup epochs for semi-supervised (default: 30)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--sam_version", type=str, default="medsam", choices=["medsam", "sam2", "sam1"])
    parser.add_argument("--skip_supervised", action="store_true", help="Skip supervised baseline stage")
    parser.add_argument("--skip_semi", action="store_true", help="Skip semi-supervised stage")
    parser.add_argument("--config", type=str, default="configs/brats2019_config.yaml", help="Config file path")
    parser.add_argument("--splits", type=str, default="data/splits_brats2019.json", help="Splits file path")
    args = parser.parse_args()

    total_start_time = time.time()
    out_dir = "runs/brats2019"
    sup_dir = os.path.join(out_dir, "supervised")
    semi_dir = os.path.join(out_dir, "semi_supervised")
    os.makedirs(sup_dir, exist_ok=True)
    os.makedirs(semi_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" 🚀 STARTING MASTER UNIFIED BENCHMARK ON BraTS 2019")
    print("=" * 80)
    print(f" Config         : {args.config}")
    print(f" Splits         : {args.splits}")
    print(f" Regimes        : {args.ratios}")
    print(f" Epochs         : {args.epochs} (Warmup: {args.warmup_epochs})")
    print(f" Batch Size     : {args.batch_size}")
    print(f" Foundation SAM : {args.sam_version.upper()}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1: PURE SUPERVISED BASELINE
    # =========================================================================
    if not args.skip_supervised:
        print("\n" + "#" * 80)
        print("  STAGE 1: PURE SUPERVISED BASELINES (BraTS 2019 Labeled Only)")
        print("#" * 80)
        for i, ratio in enumerate(args.ratios, 1):
            print(f"\n>>> [{i}/{len(args.ratios)}] Executing Supervised Baseline for {ratio}...")
            log_path = os.path.join(sup_dir, ratio, "training.log")
            cmd = [
                sys.executable, "scripts/train_supervised.py",
                "--config", args.config,
                "--splits", args.splits,
                "--ratio", ratio,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--val_interval", "5",
                "--save_dir", sup_dir
            ]
            ret = run_command_stream(cmd, log_path)
            if ret != 0:
                print(f"❌ Error during supervised training of {ratio} (Code {ret})")

    # =========================================================================
    # STAGE 2: MEDSAM SEMI-SUPERVISED PIPELINE
    # =========================================================================
    if not args.skip_semi:
        print("\n" + "#" * 80)
        print("  STAGE 2: MEDSAM SEMI-SUPERVISED PIPELINE (BraTS 2019 Labeled + Pseudo)")
        print("#" * 80)
        for i, ratio in enumerate(args.ratios, 1):
            print(f"\n>>> [{i}/{len(args.ratios)}] Executing MedSAM Semi-Supervised for {ratio}...")
            log_path = os.path.join(semi_dir, ratio, "pipeline.log")
            cmd = [
                sys.executable, "scripts/train_semi_supervised.py",
                "--config", args.config,
                "--splits", args.splits,
                "--ratio", ratio,
                "--epochs", str(args.epochs),
                "--warmup_epochs", str(args.warmup_epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--val_interval", "5",
                "--sam_version", args.sam_version,
                "--max_pseudo_ratio", "1.5",
                "--save_dir", semi_dir
            ]
            ret = run_command_stream(cmd, log_path)
            if ret != 0:
                print(f"❌ Error during semi-supervised training of {ratio} (Code {ret})")

    total_duration = str(timedelta(seconds=int(time.time() - total_start_time)))

    # =========================================================================
    # STAGE 3: COMPILE MASTER COMPARATIVE REPORT
    # =========================================================================
    with open(args.splits, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    comparison_results = {}
    for ratio in args.ratios:
        part = splits_data['semi_supervised_partitions'].get(ratio, {})
        n_labeled = part.get('num_labeled', '?')
        n_unlabeled = part.get('num_unlabeled', '?')

        # Load Supervised Summary
        sup_path = os.path.join(sup_dir, ratio, "metrics_summary.json")
        sup_dice = 0.0
        sup_metrics = {}
        if os.path.exists(sup_path):
            with open(sup_path, "r", encoding="utf-8") as f:
                sup_data = json.load(f)
                sup_dice = sup_data.get("best_mean_dice", 0.0)
                sup_metrics = sup_data.get("metrics", {})

        # Load Semi-Supervised Summary
        semi_path = os.path.join(semi_dir, ratio, "metrics_summary.json")
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
    master_comp_file = os.path.join(out_dir, "master_comparative_table.json")
    with open(master_comp_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "BraTS-2019",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "sam_version": args.sam_version,
            "comparison": comparison_results
        }, f, indent=2)

    # Print Final Summary Table
    print("\n" + "=" * 88)
    print(" 🏁 MASTER COMPARATIVE BENCHMARK: BraTS 2019 (SUPERVISED VS. SEMI-SUPERVISED)")
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
        excel_out = os.path.join(out_dir, "benchmark_master_results.xlsx")
        build_excel_report(
            output_path=excel_out,
            splits_path=args.splits,
            sup_dir=sup_dir,
            semi_dir=semi_dir,
            dataset_name="BraTS-2019"
        )
        print(f"📊 [EXCEL] Complete BraTS 2019 benchmark report exported to: {excel_out}\n")
    except Exception as e:
        print(f"Warning: Excel export failed: {e}")


if __name__ == "__main__":
    main()
