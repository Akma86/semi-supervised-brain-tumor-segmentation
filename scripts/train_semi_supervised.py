"""
Hierarchy-Aware Semi-Supervised Brain Tumor Segmentation Pipeline:
  Phase 1: Warm-up backbone (MONAI SegResNet or baseline CNN) on labeled scans.
  Phase 2: Full-volume TTA pseudo-label generation with Shannon Entropy voxel filtering,
           Hierarchy Violation Index (ET <= TC <= WT), Decision Engine triage, and MedSAM refinement.
  Phase 3: Cosine-annealed retraining on labeled + entropy-masked pseudo-labeled slices.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset
from src.models.foundation_models import build_segmentor
from src.semi_supervised.decision_engine import DecisionEngine, PseudoLabelAction
from src.semi_supervised.hierarchical_prompter import HierarchicalSAMRefiner
from src.semi_supervised.hierarchy import compute_hierarchy_violation_index
from src.semi_supervised.reliability import compute_entropy_map, evaluate_subregion_reliabilities
from src.utils.metrics import postprocess_brats_prediction, predict_probs_tta
from src.utils.training_helpers import (
    DynamicSemiSupervisedDataset,
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


def build_voxel_confidence_mask(ent_map: np.ndarray, entropy_thresh: float) -> np.ndarray:
    """Keeps confident pixels below entropy threshold, relaxing gently if slice is unusually uncertain."""
    vm = (ent_map <= entropy_thresh).astype(np.uint8)
    if float(vm.mean()) < 0.5:
        vm = (ent_map <= max(entropy_thresh, 0.65)).astype(np.uint8)
    return vm


def _score_candidate(
    base_score: float,
    tc_mask: np.ndarray,
    et_mask: np.ndarray,
    wt_voxels: int,
) -> float:
    """Prioritize rich multi-subregion tumor slices above tiny boundary slivers."""
    tc_bonus = 0.04 if int(tc_mask.sum()) > 10 else 0.0
    et_bonus = 0.04 if int(et_mask.sum()) > 10 else 0.0
    size_bonus = 0.02 * min(1.0, float(wt_voxels) / 500.0)
    return float(base_score) + tc_bonus + et_bonus + size_bonus


def _finalize_pseudo_entry(item: Dict[str, Any], mask_u8: np.ndarray) -> Dict[str, Any]:
    return {
        'image_fp16': item['image_fp16'],
        'mask_u8': mask_u8.astype(np.uint8),
        'voxel_mask_u8': item['voxel_mask_u8'],
        'weight': item['weight'],
        'score': item['score'],
        'patient_id': item['patient_id'],
        'slice_idx': item['slice_idx'],
    }


def generate_pseudo_labels_volumetric(
    model: torch.nn.Module,
    unlabeled_dataset: BraTSDataset,
    decision_engine: DecisionEngine,
    sam_refiner: Optional[HierarchicalSAMRefiner],
    device: str,
    slice_mode: str,
    use_tta: bool,
    postprocess: bool,
    entropy_thresh: float,
    max_slices_per_patient: int,
    max_pseudo_slices: int,
    max_pseudo_ratio: float = None,
    num_labeled_items: int = 0,
    blend_alpha: float = 0.65,
    sam_builder=None,
) -> tuple:
    """
    Runs volumetric TTA triage on student first, frees student VRAM before loading MedSAM
    on uncertain slices, then builds compact float16/uint8 pseudo-label pool entries.
    """
    model.eval()
    staged_candidates: List[Dict[str, Any]] = []
    counts = {'ACCEPT': 0, 'REFINE': 0, 'REJECT': 0}
    dev = torch.device(device)
    use_amp = (device == 'cuda')

    with torch.no_grad():
        for u_idx in tqdm(range(len(unlabeled_dataset)), desc="Triage Unlabeled", unit="vol"):
            entry = unlabeled_dataset.get_patient_volume_entry(u_idx)
            H, W, D = entry['orig_shape']
            brain_slices = entry['brain_slices'][::2]

            probs_by_z: Dict[int, torch.Tensor] = {}
            pred_vol = np.zeros((H, W, D), dtype=np.uint8)

            for start in range(0, len(brain_slices), 16):
                chunk_z = brain_slices[start:start + 16]
                imgs_np = unlabeled_dataset.extract_slices_batch(entry, chunk_z, slice_mode=slice_mode)
                imgs_t = torch.from_numpy(imgs_np).to(device=dev, dtype=torch.float32)
                probs = predict_probs_tta(model, imgs_t, use_tta=use_tta, use_amp=use_amp)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                preds_t = np.transpose(preds, (1, 2, 0))
                for off in (0, 1):
                    pred_vol[:, :, np.clip(chunk_z + off, 0, D - 1)] = preds_t
                for i, z in enumerate(chunk_z):
                    probs_by_z[int(z)] = probs[i].cpu()

            if postprocess:
                pred_vol = postprocess_brats_prediction(pred_vol)

            candidates = []
            for z in brain_slices:
                zi = int(z)
                pred_s = pred_vol[:, :, zi]
                wt_s = (pred_s == 1) | (pred_s == 2) | (pred_s == 3)
                wt_voxels = int(wt_s.sum())
                if wt_voxels < decision_engine.min_tumor_voxels:
                    continue

                tc_s = (pred_s == 1) | (pred_s == 3)
                et_s = (pred_s == 3)
                probs_s = probs_by_z[zi]
                logits_proxy = torch.log(probs_s.unsqueeze(0).clamp_min(1e-7))

                reliabilities = evaluate_subregion_reliabilities(
                    logits=logits_proxy,
                    wt_mask=torch.from_numpy(wt_s.astype(np.float32)).unsqueeze(0),
                    tc_mask=torch.from_numpy(tc_s.astype(np.float32)).unsqueeze(0),
                    et_mask=torch.from_numpy(et_s.astype(np.float32)).unsqueeze(0),
                )
                hierarchy_stats = compute_hierarchy_violation_index(wt_mask=wt_s, tc_mask=tc_s, et_mask=et_s)
                decision = decision_engine.evaluate(
                    reliabilities=reliabilities,
                    hierarchy_stats=hierarchy_stats,
                    wt_voxel_count=wt_voxels,
                )
                action = decision['action']
                counts[action.value] += 1
                if action in (PseudoLabelAction.ACCEPT, PseudoLabelAction.REFINE):
                    base_score = float(decision.get('score', 0.0))
                    candidates.append({
                        'z': zi,
                        'action': action,
                        'score': _score_candidate(base_score, tc_s, et_s, wt_voxels),
                        'pred_s': pred_s,
                        'wt_s': wt_s,
                        'tc_s': tc_s,
                        'et_s': et_s,
                        'probs_s': probs_s,
                    })

            if not candidates:
                continue

            candidates.sort(key=lambda c: c['score'], reverse=True)
            accepts = [c for c in candidates if c['action'] == PseudoLabelAction.ACCEPT]
            refines = [c for c in candidates if c['action'] == PseudoLabelAction.REFINE]
            half_k = max(1, max_slices_per_patient // 2)
            picked = accepts[:half_k] + refines[:max_slices_per_patient - half_k]
            if len(picked) < max_slices_per_patient:
                picked_z = {c['z'] for c in picked}
                picked.extend([c for c in candidates if c['z'] not in picked_z][:max_slices_per_patient - len(picked)])

            for cand in picked:
                zi = cand['z']
                img_np, _ = unlabeled_dataset.extract_slice(entry, zi, slice_mode=slice_mode)
                ent_map = compute_entropy_map(cand['probs_s'].unsqueeze(0))[0, 0].numpy()

                if cand['action'] == PseudoLabelAction.ACCEPT:
                    staged_candidates.append({
                        'action': PseudoLabelAction.ACCEPT,
                        'image_fp16': img_np.astype(np.float16),
                        'mask_u8': cand['pred_s'].astype(np.uint8),
                        'voxel_mask_u8': build_voxel_confidence_mask(ent_map, entropy_thresh),
                        'weight': 1.0,
                        'score': cand['score'],
                        'patient_id': entry['patient_id'],
                        'slice_idx': zi,
                    })
                else:
                    pr = cand['probs_s'].numpy()
                    raw_probs_fp16 = np.stack([
                        pr[1] + pr[2] + pr[3],
                        pr[1] + pr[3],
                        pr[3],
                    ], axis=0).astype(np.float16)
                    staged_candidates.append({
                        'action': PseudoLabelAction.REFINE,
                        'image_fp16': img_np.astype(np.float16),
                        'raw_wt': cand['wt_s'].astype(np.uint8),
                        'raw_tc': cand['tc_s'].astype(np.uint8),
                        'raw_et': cand['et_s'].astype(np.uint8),
                        'raw_probs_fp16': raw_probs_fp16,
                        'voxel_mask_u8': build_voxel_confidence_mask(ent_map, max(entropy_thresh, 0.55)),
                        'weight': 0.8,
                        'score': cand['score'],
                        'patient_id': entry['patient_id'],
                        'slice_idx': zi,
                    })

    staged_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    cap = max_pseudo_slices
    if max_pseudo_ratio is not None and num_labeled_items > 0:
        cap = min(cap, max(1, int(num_labeled_items * max_pseudo_ratio)))
    if len(staged_candidates) > cap:
        staged_candidates = staged_candidates[:cap]

    refine_items = [item for item in staged_candidates if item['action'] == PseudoLabelAction.REFINE]
    pseudo_pool: List[Dict[str, Any]] = [
        _finalize_pseudo_entry(item, item['mask_u8'])
        for item in staged_candidates
        if item['action'] == PseudoLabelAction.ACCEPT
    ]

    if refine_items:
        # Isolate MedSAM on GPU by offloading student weights before Stage 2b refinement
        model.to('cpu')
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

        owns_refiner = (sam_refiner is None and sam_builder is not None)
        active_refiner = sam_builder() if owns_refiner else sam_refiner
        if active_refiner is not None:
            print(f"  >> SAM Backend: {active_refiner.sam_version} (active={active_refiner.has_sam}) | Refining {len(refine_items)} slices (blend_alpha={blend_alpha:.2f})")
            for item in tqdm(refine_items, desc="MedSAM Refine", unit="slice"):
                rp = item['raw_probs_fp16'].astype(np.float32)
                refined = active_refiner.refine_hierarchical(
                    image=item['image_fp16'][:4].astype(np.float32),
                    raw_wt=item['raw_wt'].astype(np.float32),
                    raw_tc=item['raw_tc'].astype(np.float32),
                    raw_et=item['raw_et'].astype(np.float32),
                    raw_probs={'WT': rp[0], 'TC': rp[1], 'ET': rp[2]},
                    blend_alpha=blend_alpha,
                )
                seg_arr = refined['assembled_seg']
                mask_u8 = (seg_arr.cpu().numpy() if isinstance(seg_arr, torch.Tensor) else np.asarray(seg_arr)).astype(np.uint8)
                pseudo_pool.append(_finalize_pseudo_entry(item, mask_u8))

            if owns_refiner:
                del active_refiner
                gc.collect()
                if use_amp:
                    torch.cuda.empty_cache()

        model.to(dev)

    pseudo_pool.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    return pseudo_pool, counts


def main():
    parser = argparse.ArgumentParser(description="Hierarchy-Aware Semi-Supervised Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--ratio", type=str, default="ratio_10")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/experiments")
    parser.add_argument("--use_sam", type=str_to_bool, default=True)
    parser.add_argument("--sam_version", type=str, default="medsam", choices=["medsam", "sam2", "sam1"])
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--tau_high", type=float, default=0.78)
    parser.add_argument("--tau_low", type=float, default=0.40)
    parser.add_argument("--blend_alpha", type=float, default=0.65)
    parser.add_argument("--warmup_ckpt", type=str, default=None)
    parser.add_argument("--pseudo_cache", type=str, default=None)
    parser.add_argument("--max_pseudo_ratio", type=float, default=None)
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
    parser.add_argument("--entropy_thresh", type=float, default=0.45)
    parser.add_argument("--max_slices_per_unlabeled", type=int, default=6)
    parser.add_argument("--max_pseudo_slices", type=int, default=800)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    cohort_dirs, splits_path, modalities = resolve_dataset_config(cfg, args.splits)

    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file missing: {splits_path}")
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    partition = splits['semi_supervised_partitions'][args.ratio]
    labeled_ids = partition['labeled_patients']
    unlabeled_ids = partition['unlabeled_patients']
    val_ids = splits['val']

    exp_dir = os.path.join(args.save_dir, args.ratio)
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "best_model.pth")

    print(f"\n[Semi-Supervised] ratio={args.ratio} ({len(labeled_ids)} labeled, {len(unlabeled_ids)} unlabeled, "
          f"{len(val_ids)} val) | arch={args.arch} | mode={args.slice_mode} | sam={args.use_sam} | device={device}")

    train_transform = build_train_transform(args.augment)
    labeled_base = BraTSDataset(
        data_dir=cohort_dirs,
        modalities=modalities,
        is_labeled=True,
        patient_ids=labeled_ids,
        transform=train_transform,
        mode=args.slice_mode,
        cache_data=True,
    )
    labeled_train_ds = wrap_multi_slice_dataset(labeled_base, args.slices_per_volume)
    labeled_loader = DataLoader(labeled_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    unlabeled_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        modalities=modalities,
        is_labeled=False,
        patient_ids=unlabeled_ids,
        mode=args.slice_mode,
        cache_data=False,
    )
    val_dataset = BraTSDataset(
        data_dir=cohort_dirs,
        modalities=modalities,
        is_labeled=True,
        patient_ids=val_ids,
        mode=args.slice_mode,
        cache_data=True,
    )

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
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    warmup_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, warmup_epochs), eta_min=args.lr * 0.2
    )
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_val_dice = -1.0
    best_epoch = 0
    best_metrics = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0}
    start_time = time.time()

    if args.warmup_ckpt:
        if not os.path.exists(args.warmup_ckpt):
            raise FileNotFoundError(f"Warmup checkpoint missing: {args.warmup_ckpt}")
        warm_ckpt = torch.load(args.warmup_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(warm_ckpt['model_state_dict'])
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
            epoch=int(warm_ckpt.get('epoch', 0)),
            optimizer=optimizer,
            prefix=f"WarmupCkpt (ep={warm_ckpt.get('epoch', '?')}) | Val 3D",
            slice_stride=args.val_stride,
        )
        best_epoch = int(warm_ckpt.get('epoch', 0))

    if warmup_epochs > 0:
        print(f">>> [PHASE 1] Warm-up Training (Epochs 1..{warmup_epochs})")
        for epoch in range(1, warmup_epochs + 1):
            avg_loss = train_one_epoch(
                model=model,
                loader=labeled_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                desc=f"Warmup [{epoch:02d}/{warmup_epochs:02d}]",
            )
            warmup_sched.step()

            if (epoch % args.val_interval == 0) or (epoch == warmup_epochs):
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
                    prefix=f"Warmup Loss={avg_loss:.4f} | Val 3D",
                    slice_stride=args.val_stride,
                )
                if best_val_dice > prev_best:
                    best_epoch = epoch

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  >> Restored best warm-up weights (Epoch {ckpt.get('epoch', best_epoch)}, {best_val_dice*100:.2f}%) for triage.")

    pseudo_labeled_pool: List[Dict[str, Any]] = []
    counts = {'ACCEPT': 0, 'REFINE': 0, 'REJECT': 0}

    if args.epochs > warmup_epochs and len(unlabeled_dataset) > 0:
        pool_cache_out = os.path.join(exp_dir, "pseudo_pool_cache.pt")
        if args.pseudo_cache and os.path.exists(args.pseudo_cache):
            cached = torch.load(args.pseudo_cache, map_location='cpu', weights_only=False)
            pseudo_labeled_pool = cached['pseudo_pool']
            counts = cached['counts']
            print(f"\n>>> [PHASE 2] Loaded cached pseudo pool ({len(pseudo_labeled_pool)} slices) from {args.pseudo_cache}")
        else:
            print(f"\n>>> [PHASE 2] Volumetric Triage & Foundation Model Refinement ({len(unlabeled_dataset)} volumes)")
            decision_engine = DecisionEngine(tau_high=args.tau_high, tau_low=args.tau_low, delta_hvi=0.03)
            sam_builder = lambda: HierarchicalSAMRefiner(
                checkpoint_path=args.sam_checkpoint,
                use_sam=args.use_sam,
                sam_version=args.sam_version,
                sam2_config=args.sam2_config,
                model_type=args.sam_model_type,
                device=device,
            )

            pseudo_labeled_pool, counts = generate_pseudo_labels_volumetric(
                model=model,
                unlabeled_dataset=unlabeled_dataset,
                decision_engine=decision_engine,
                sam_refiner=None,
                device=device,
                slice_mode=args.slice_mode,
                use_tta=args.tta,
                postprocess=args.postprocess,
                entropy_thresh=args.entropy_thresh,
                max_slices_per_patient=args.max_slices_per_unlabeled,
                max_pseudo_slices=args.max_pseudo_slices,
                max_pseudo_ratio=args.max_pseudo_ratio,
                num_labeled_items=len(labeled_train_ds),
                blend_alpha=args.blend_alpha,
                sam_builder=sam_builder,
            )
            torch.save({'pseudo_pool': pseudo_labeled_pool, 'counts': counts}, pool_cache_out)
            print(f"  >> Cached pseudo pool -> {pool_cache_out}")

        print(
            f"  >> Triage slice counts: ACCEPT={counts['ACCEPT']}, REFINE={counts['REFINE']}, REJECT={counts['REJECT']} | "
            f"Selected Pool={len(pseudo_labeled_pool)} slices"
        )

        retrain_epochs = args.epochs - warmup_epochs
        print(f"\n>>> [PHASE 3] Semi-Supervised Retraining (Epochs {warmup_epochs + 1}..{args.epochs})")
        semi_dataset = DynamicSemiSupervisedDataset(
            labeled_dataset=labeled_base,
            slices_per_labeled_volume=args.slices_per_volume,
            pseudo_pool=pseudo_labeled_pool,
            transform=train_transform,
        )
        semi_loader = DataLoader(semi_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        retrain_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        retrain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            retrain_opt, T_max=max(1, retrain_epochs), eta_min=1e-6
        )

        for epoch in range(warmup_epochs + 1, args.epochs + 1):
            avg_loss = train_one_epoch(
                model=model,
                loader=semi_loader,
                criterion=criterion,
                optimizer=retrain_opt,
                scaler=scaler,
                device=device,
                desc=f"Retrain [{epoch:02d}/{args.epochs:02d}]",
            )
            retrain_sched.step()

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
                    optimizer=retrain_opt,
                    prefix=f"Retrain Loss={avg_loss:.4f} | Val 3D",
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
        'arch': args.arch,
        'slice_mode': args.slice_mode,
        'tta': args.tta,
        'postprocess': args.postprocess,
        'pseudo_slices_used': len(pseudo_labeled_pool),
        'triage_counts': counts,
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
        f"\n[Semi-Supervised Done in {elapsed:.1f}s] Best Mean Dice @ Ep {best_epoch}: {final_val_metrics['Mean']*100:.2f}% "
        f"(WT={final_val_metrics['WT']*100:.2f}%, TC={final_val_metrics['TC']*100:.2f}%, ET={final_val_metrics['ET']*100:.2f}%)"
    )


if __name__ == '__main__':
    main()
