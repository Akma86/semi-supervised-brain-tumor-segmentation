"""
Master Automated Benchmark Runner for Semi-Supervised Brain Tumor Segmentation.
Sequentially trains and evaluates across annotation regimes:
  - ratio_10 (10% Labeled, 90% Unlabeled)
  - ratio_20 (20% Labeled, 80% Unlabeled)
  - ratio_40 (40% Labeled, 60% Unlabeled)

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --ratios ratio_10 ratio_20 ratio_40 --epochs 30 --warmup_epochs 10
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
    parser = argparse.ArgumentParser(description="Master Automated Runner for Semi-Supervised Experiments")
    parser.add_argument("--ratios", nargs="+", default=["ratio_10", "ratio_20", "ratio_40", "ratio_80"],
                        help="List of ratio splits to run sequentially (default: ratio_10 ratio_20 ratio_40 ratio_80)")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs per run (default: 30)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs per run (default: 10)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--save_dir", type=str, default="runs/experiments", help="Base directory for saving logs")
    parser.add_argument("--sam_version", type=str, default="medsam", choices=["medsam", "sam2", "sam1"], help="SAM model: medsam (Bowang-Lab ViT-B for MRI), sam2 (Meta SAM 2.1), or sam1 (default: medsam)")
    parser.add_argument("--max_pseudo_ratio", type=float, default=None, help="Optional max ratio of pseudo-labels to labeled samples (e.g. 2.0 to prevent drowning in ratio_10)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print(" 🚀 MASTER BENCHMARK RUNNER: 3D U-NET + HIERARCHICAL SAM REFINER")
    print("=" * 76)
    print(f" Target Ratios   : {args.ratios}")
    print(f" SAM Version     : {args.sam_version.upper()}")
    print(f" Epochs per Run  : {args.epochs} (Warm-Up: {args.warmup_epochs})")
    print(f" Batch Size      : {args.batch_size}")
    print(f" Save Directory  : {args.save_dir}")
    print("=" * 76 + "\n")

    overall_start_time = time.time()
    results = {}

    for idx, ratio in enumerate(args.ratios, start=1):
        print("\n" + "#" * 76)
        print(f" [{idx}/{len(args.ratios)}] STARTING REGIME: {ratio.upper()}")
        print("#" * 76)

        exp_dir = os.path.join(args.save_dir, ratio)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, "train.log")

        cmd = [
            sys.executable,
            "scripts/train_semi_supervised.py",
            "--ratio", ratio,
            "--epochs", str(args.epochs),
            "--warmup_epochs", str(args.warmup_epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--sam_version", args.sam_version,
            "--save_dir", args.save_dir
        ]
        if args.max_pseudo_ratio is not None:
            cmd.extend(["--max_pseudo_ratio", str(args.max_pseudo_ratio)])

        print(f"Executing: {' '.join(cmd)}")
        print(f"Logging stdout/stderr to: {log_file}\n")

        start_time = time.time()

        # Run process and tee output to both console and log file
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
            # Load metrics summary
            summary_path = os.path.join(exp_dir, "metrics_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    results[ratio] = {
                        'status': 'SUCCESS',
                        'best_mean_dice': summary.get('best_mean_dice', 0.0),
                        'elapsed_time': elapsed_str,
                        'checkpoint': summary.get('checkpoint', '')
                    }
            else:
                results[ratio] = {'status': 'SUCCESS', 'elapsed_time': elapsed_str}
        else:
            print(f"\n❌ [{idx}/{len(args.ratios)}] {ratio} FAILED with return code {proc.returncode}!")
            results[ratio] = {'status': 'FAILED', 'elapsed_time': elapsed_str}

    total_elapsed = str(timedelta(seconds=int(time.time() - overall_start_time)))

    # Save Master Benchmark Report
    master_report_path = os.path.join(args.save_dir, "master_benchmark_report.json")
    with open(master_report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_elapsed_time': total_elapsed,
            'results': results
        }, f, indent=2)

    # Print Final Summary Table
    print("\n\n" + "=" * 76)
    print(" 🏆 ALL EXPERIMENTS COMPLETED! MASTER BENCHMARK SUMMARY")
    print("=" * 76)
    print(f" Total Duration : {total_elapsed}")
    print(f" Master Report  : {master_report_path}")
    print("-" * 76)
    print(f"{'Regime':<15} | {'Status':<10} | {'Best Mean Dice':<18} | {'Duration':<12}")
    print("-" * 76)

    for ratio, res in results.items():
        status = res.get('status', 'UNKNOWN')
        dice = res.get('best_mean_dice', None)
        dice_str = f"{dice*100:.2f}%" if dice is not None else "N/A"
        dur = res.get('elapsed_time', 'N/A')
        print(f"{ratio:<15} | {status:<10} | {dice_str:<18} | {dur:<12}")

    print("=" * 76 + "\n")


if __name__ == '__main__':
    main()
