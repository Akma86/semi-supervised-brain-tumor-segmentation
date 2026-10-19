"""
Master Benchmark Runner for Pure Supervised Baselines.
Sequentially trains single 3D U-Net models on purely labeled partitions:
  - ratio_10:  10 labeled patients
  - ratio_20:  20 labeled patients
  - ratio_40:  41 labeled patients
  - ratio_100: 102 labeled patients (Fully Supervised Oracle / Upper Bound)

Usage:
    python scripts/run_supervised_benchmarks.py
    python scripts/run_supervised_benchmarks.py --ratios ratio_10 ratio_20 ratio_40 --epochs 200
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


def main():
    parser = argparse.ArgumentParser(description="Master Automated Runner for Supervised Baseline Experiments")
    parser.add_argument("--ratios", nargs="+", default=["ratio_10", "ratio_20", "ratio_40", "ratio_80"],
                        help="List of ratio splits to run sequentially (default: ratio_10 ratio_20 ratio_40 ratio_80)")
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs per run (default: 200)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (default: 5)")
    parser.add_argument("--save_dir", type=str, default="runs/supervised", help="Base directory for saving logs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print(" 🚀 MASTER BENCHMARK RUNNER: PURE SUPERVISED BASELINES (LABELED-ONLY)")
    print("=" * 76)
    print(f" Target Ratios   : {args.ratios}")
    print(f" Epochs per Run  : {args.epochs}")
    print(f" Batch Size      : {args.batch_size}")
    print(f" Save Directory  : {args.save_dir}")
    print("=" * 76 + "\n")

    overall_start_time = time.time()
    results = {}

    for idx, ratio in enumerate(args.ratios, start=1):
        print("\n" + "#" * 76)
        print(f" [{idx}/{len(args.ratios)}] STARTING SUPERVISED REGIME: {ratio.upper()}")
        print("#" * 76)

        exp_dir = os.path.join(args.save_dir, ratio)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, "train.log")

        cmd = [
            sys.executable,
            "scripts/train_supervised.py",
            "--ratio", ratio,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--val_interval", str(args.val_interval),
            "--save_dir", args.save_dir
        ]

        print(f"Executing: {' '.join(cmd)}")
        print(f"Logging stdout/stderr to: {log_file}\n")

        start_time = time.time()

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

        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if proc.returncode == 0:
            print(f"\n✅ [{idx}/{len(args.ratios)}] {ratio} COMPLETED SUCCESSFULLY in {elapsed_str}!")
            summary_path = os.path.join(exp_dir, "metrics_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                best_dice = summary.get("best_mean_dice", 0.0)
                metrics = summary.get("metrics", {})
            else:
                best_dice = 0.0
                metrics = {}

            results[ratio] = {
                "status": "SUCCESS",
                "best_mean_dice": best_dice,
                "metrics": metrics,
                "elapsed_time": elapsed_str,
                "checkpoint": os.path.join(exp_dir, "best_model.pth")
            }
        else:
            print(f"\n❌ [{idx}/{len(args.ratios)}] {ratio} FAILED with return code {proc.returncode}!")
            results[ratio] = {
                "status": "FAILED",
                "best_mean_dice": 0.0,
                "metrics": {},
                "elapsed_time": elapsed_str,
                "checkpoint": None
            }

    total_elapsed = str(timedelta(seconds=int(time.time() - overall_start_time)))

    # Save Master Supervised Report
    master_report_path = os.path.join(args.save_dir, "master_supervised_report.json")
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_time": total_elapsed,
        "results": results
    }
    with open(master_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Print Pretty Summary Table
    print("\n" + "=" * 76)
    print(" 🏁 SUPERVISED BENCHMARK RUNNER: FINAL SUMMARY TABLE")
    print("=" * 76)
    print(f" Total Duration : {total_elapsed}")
    print(f" Master Report  : {master_report_path}")
    print("-" * 76)
    print(f"{'Regime':<15} | {'Status':<10} | {'Best Mean Dice':<18} | {'Duration'}")
    print("-" * 76)
    for ratio, info in results.items():
        dice_str = f"{info['best_mean_dice']*100:.2f}%" if info['status'] == 'SUCCESS' else "N/A"
        print(f"{ratio:<15} | {info['status']:<10} | {dice_str:<18} | {info['elapsed_time']}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
