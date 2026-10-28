"""
Export Benchmark Results to a Professional Styled Excel Workbook (.xlsx).
Compiles Supervised Baselines, MedSAM Semi-Supervised, and Comparative Gains
across all annotation regimes (ratio_10, ratio_20, ratio_40, ratio_80).

Usage:
    python scripts/export_to_excel.py
    python scripts/export_to_excel.py --output runs/BraTS_Africa_Benchmark_Report.xlsx
"""

import os
import sys
import json
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_excel_report(
    output_path: str = "runs/benchmark_master_results.xlsx",
    splits_path: str = "data/splits.json",
    sup_dir: str = "runs/supervised",
    semi_dir: str = "runs/experiments",
    dataset_name: str = "BraTS-Africa"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    target_order = ["ratio_10", "ratio_20", "ratio_40", "ratio_70", "ratio_80", "ratio_30", "ratio_50", "ratio_100"]
    available_parts = splits_data.get('semi_supervised_partitions', {})
    ratios = [r for r in target_order if r in available_parts]
    r_labels = {
        r: f"{int(available_parts[r]['labeled_ratio']*100)}% Labeled ({available_parts[r]['num_labeled']} Cases)"
        for r in ratios
    }

    wb = Workbook()

    # -------------------------------------------------------------
    # Style Definitions
    # -------------------------------------------------------------
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    sub_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    gain_pos_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid") # light green
    gain_neg_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid") # light red
    zebra_fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")

    font_title = Font(name="Calibri", size=16, bold=True, color="1F4E79")
    font_subtitle = Font(name="Calibri", size=11, italic=True, color="595959")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_sub_header = Font(name="Calibri", size=11, bold=True, color="1F4E79")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_regular = Font(name="Calibri", size=11)
    font_gain_pos = Font(name="Calibri", size=11, bold=True, color="276A3C")
    font_gain_neg = Font(name="Calibri", size=11, bold=True, color="C00000")

    thin_border_side = Side(border_style="thin", color="D3D3D3")
    border_all = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    # -------------------------------------------------------------
    # SHEET 1: Master Comparative Benchmark
    # -------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "Master Benchmark"
    ws1.views.sheetView[0].showGridLines = True

    # Title
    ws1.merge_cells("A1:K1")
    ws1["A1"] = "Brain Tumor Segmentation Benchmark: Supervised vs. MedSAM Semi-Supervised"
    ws1["A1"].font = font_title
    ws1["A1"].alignment = align_left

    ws1.merge_cells("A2:K2")
    ws1["A2"] = "Dataset: BraTS-Africa (Sub-Saharan Africa Cohort) | Backbone: Single 3D U-Net | Refiner: MedSAM (Bowang-Lab ViT-B)"
    ws1["A2"].font = font_subtitle
    ws1["A2"].alignment = align_left

    # Table Multi-level Headers
    # Row 4: Super headers
    headers_row4 = [
        ("Regime", "A4:B4"),
        ("Dataset Distribution", "C4:D4"),
        ("Pure Supervised Baseline (Lower Bound)", "E4:H4"),
        ("MedSAM Semi-Supervised (Our Method)", "I4:L4"),
        ("Performance Gain (Δ)", "M4:M4")
    ]
    for text, rng in headers_row4:
        ws1.merge_cells(rng)
        top_left = rng.split(":")[0]
        ws1[top_left] = text
        ws1[top_left].font = font_header
        ws1[top_left].alignment = align_center
        # Fill cells in range
        start_col, start_row = rng.split(":")[0][0], int(rng.split(":")[0][1:])
        end_col, end_row = rng.split(":")[1][0], int(rng.split(":")[1][1:])
        for r in range(start_row, end_row + 1):
            for c in range(ord(start_col) - 64, ord(end_col) - 64 + 1):
                cell = ws1.cell(row=r, column=c)
                cell.fill = header_fill
                cell.border = border_all

    # Row 5: Column headers
    cols_row5 = [
        "Regime Key", "Regime Description",
        "Labeled Cases", "Unlabeled Cases",
        "WT Dice", "TC Dice", "ET Dice", "Mean Dice",
        "WT Dice", "TC Dice", "ET Dice", "Mean Dice",
        "Absolute Gain (Δ)"
    ]
    for idx, c_name in enumerate(cols_row5, start=1):
        cell = ws1.cell(row=5, column=idx)
        cell.value = c_name
        cell.font = font_sub_header
        cell.fill = sub_fill
        cell.alignment = align_center
        cell.border = border_all

    # Data Rows
    current_row = 6
    for ratio in ratios:
        part = splits_data['semi_supervised_partitions'].get(ratio, {})
        n_labeled = part.get('num_labeled', 0)
        n_unlabeled = part.get('num_unlabeled', 0)

        # Supervised Data
        sup_file = os.path.join(sup_dir, ratio, "metrics_summary.json")
        sup_wt, sup_tc, sup_et, sup_mean = 0.0, 0.0, 0.0, 0.0
        if os.path.exists(sup_file):
            with open(sup_file) as sf:
                sd = json.load(sf)
                m = sd.get('metrics', {})
                sup_wt = m.get('WT', 0.0) * 100
                sup_tc = m.get('TC', 0.0) * 100
                sup_et = m.get('ET', 0.0) * 100
                sup_mean = sd.get('best_mean_dice', 0.0) * 100

        # Semi-Supervised Data
        semi_file = os.path.join(semi_dir, ratio, "metrics_summary.json")
        semi_wt, semi_tc, semi_et, semi_mean = 0.0, 0.0, 0.0, 0.0
        if os.path.exists(semi_file):
            with open(semi_file) as smf:
                smd = json.load(smf)
                m = smd.get('metrics', {})
                semi_wt = m.get('WT', 0.0) * 100
                semi_tc = m.get('TC', 0.0) * 100
                semi_et = m.get('ET', 0.0) * 100
                semi_mean = smd.get('best_mean_dice', 0.0) * 100

        gain = semi_mean - sup_mean if (semi_mean > 0 and sup_mean > 0) else None

        # Populate row
        vals = [
            (ratio, align_center, font_bold),
            (r_labels.get(ratio, ratio), align_left, font_regular),
            (n_labeled, align_center, font_regular),
            (n_unlabeled, align_center, font_regular),
            (f"{sup_wt:.2f}%" if sup_wt > 0 else "-", align_right, font_regular),
            (f"{sup_tc:.2f}%" if sup_tc > 0 else "-", align_right, font_regular),
            (f"{sup_et:.2f}%" if sup_et > 0 else "-", align_right, font_regular),
            (f"{sup_mean:.2f}%" if sup_mean > 0 else "-", align_right, font_bold),
            (f"{semi_wt:.2f}%" if semi_wt > 0 else "-", align_right, font_regular),
            (f"{semi_tc:.2f}%" if semi_tc > 0 else "-", align_right, font_regular),
            (f"{semi_et:.2f}%" if semi_et > 0 else "-", align_right, font_regular),
            (f"{semi_mean:.2f}%" if semi_mean > 0 else "-", align_right, font_bold),
        ]

        for col_idx, (val, alignment, font) in enumerate(vals, start=1):
            cell = ws1.cell(row=current_row, column=col_idx)
            cell.value = val
            cell.alignment = alignment
            cell.font = font
            cell.border = border_all
            if current_row % 2 == 1:
                cell.fill = zebra_fill

        # Gain Cell
        gain_cell = ws1.cell(row=current_row, column=13)
        if gain is not None:
            gain_cell.value = f"{'+' if gain >= 0 else ''}{gain:.2f}%"
            gain_cell.font = font_gain_pos if gain >= 0 else font_gain_neg
            gain_cell.fill = gain_pos_fill if gain >= 0 else gain_neg_fill
        else:
            gain_cell.value = "In Progress..."
            gain_cell.font = font_subtitle
        gain_cell.alignment = align_right
        gain_cell.border = border_all

        current_row += 1

    # Auto-adjust column widths
    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = max(max_len + 4, 12)

    # -------------------------------------------------------------
    # SHEET 2: Supervised Baselines Details
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Supervised Baselines")
    ws2.views.sheetView[0].showGridLines = True
    ws2.merge_cells("A1:G1")
    ws2["A1"] = "Pure Supervised Training Details (Ground-Truth Only)"
    ws2["A1"].font = font_title
    ws2["A1"].alignment = align_left

    cols_sup = ["Regime", "Labeled Patients", "Best Epoch", "WT Dice", "TC Dice", "ET Dice", "Mean Dice"]
    for idx, c_name in enumerate(cols_sup, start=1):
        cell = ws2.cell(row=3, column=idx)
        cell.value = c_name
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = align_center
        cell.border = border_all

    s_row = 4
    for ratio in ratios:
        part = splits_data['semi_supervised_partitions'].get(ratio, {})
        n_labeled = part.get('num_labeled', 0)
        sup_file = os.path.join(sup_dir, ratio, "metrics_summary.json")
        if os.path.exists(sup_file):
            with open(sup_file) as sf:
                sd = json.load(sf)
                m = sd.get('metrics', {})
                ws2.cell(row=s_row, column=1, value=ratio).font = font_bold
                ws2.cell(row=s_row, column=2, value=n_labeled).alignment = align_center
                ws2.cell(row=s_row, column=3, value=sd.get('best_epoch', '-')).alignment = align_center
                ws2.cell(row=s_row, column=4, value=f"{m.get('WT', 0)*100:.2f}%").alignment = align_right
                ws2.cell(row=s_row, column=5, value=f"{m.get('TC', 0)*100:.2f}%").alignment = align_right
                ws2.cell(row=s_row, column=6, value=f"{m.get('ET', 0)*100:.2f}%").alignment = align_right
                ws2.cell(row=s_row, column=7, value=f"{sd.get('best_mean_dice', 0)*100:.2f}%").font = font_bold
                ws2.cell(row=s_row, column=7).alignment = align_right
                for c in range(1, 8):
                    ws2.cell(row=s_row, column=c).border = border_all
                s_row += 1

    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = max(max_len + 4, 14)

    # Save Excel Workbook
    wb.save(output_path)
    print(f"✅ Excel benchmark report successfully generated at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Benchmark Results to Styled Excel (.xlsx)")
    parser.add_argument("--output", type=str, default="runs/benchmark_master_results.xlsx",
                        help="Path for saving the Excel report (.xlsx)")
    parser.add_argument("--splits", type=str, default="data/splits.json", help="Path to splits JSON")
    parser.add_argument("--sup_dir", type=str, default="runs/supervised", help="Supervised results directory")
    parser.add_argument("--semi_dir", type=str, default="runs/experiments", help="Semi-supervised results directory")
    parser.add_argument("--dataset_name", type=str, default="BraTS-Africa", help="Dataset name")
    args = parser.parse_args()

    build_excel_report(
        output_path=args.output,
        splits_path=args.splits,
        sup_dir=args.sup_dir,
        semi_dir=args.semi_dir,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main()
