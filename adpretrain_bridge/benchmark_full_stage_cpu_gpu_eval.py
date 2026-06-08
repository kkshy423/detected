#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-stage S0-S8 CPU vs GPU(uint8) preprocessing capstone comparison.

Capstone eval for the GPU-preprocess acceleration line. For each of two
preprocessing backends (cpu_pil baseline, gpu_tensor_uint8_aa_true) it runs the
ADP encoder ONCE per image, then derives three scores from the SAME projected
features:
  (i)  ADP-only-DINO score (training-free, stage-independent)
  (ii) AHL-DINO score per stage S0-S8 (per-stage AHL head)
  (iii) ADP-AHL bridge v2 score per stage S1-S8 (alpha*ADP_norm+(1-alpha)*AHL_norm)

It then selects thresholds per stage (stage_recall_bestf1 rule), under two
threshold modes (fixed CPU threshold, and each-backend self-calibrated), reports
full confusion matrices + AUC, equivalence diagnostics (CPU vs GPU), latency
breakdown, a CPU cross-check against the existing v2 main table, and a
pre-registered PASS/FAIL gap summary.

No retraining, no alpha tuning, no split change, no new backend. The existing
benchmark_preprocess_equivalence_paired.py is imported (its behaviour unchanged).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark_preprocess_equivalence_paired import (
    AHL_ROOT,
    CPU_BACKEND,
    MEAN,
    STD,
    BackendBank,
    Item,
    PREPROCESS_PHASE_KEYS,
    build_backend_bank,
    collect_items,
    combine_dra_outputs,
    flatten_feature_map,
    load_pil_image,
    load_stage_manifest,
    match_reference_features_cached,
    phase_stats,
    prepare_backend_from_pil,
    timed,
)
from common import (
    add_adpretrain_to_path,
    compress_four_to_two,
    encode_multiscale,
    make_encoder,
    make_projector,
    residual_features,
)

TASK_NAME = "20260607_cpu_gpu_uint8_full_stage_eval_v1"
GPU_BACKEND = "gpu_tensor_uint8_aa_true"
BACKENDS = [CPU_BACKEND, GPU_BACKEND]
STAGES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
ALIAS = "models_qiumianxiepai"

BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
STAGE_ROOT_BASE = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_plain_dino_large_norm_val49"
)
AHL_WEIGHTS_BASE = BRIDGE_ROOT / "output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages"
ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
PROJECTOR_CKPT = ADPRETRAIN_ROOT / "checkpoints/dino-large/checkpoints_img_norm.pth"
V2_MAIN_TABLE = (
    BRIDGE_ROOT
    / "summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv"
)
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "output" / TASK_NAME
DEFAULT_SUMMARY_ROOT = BRIDGE_ROOT / "summary" / TASK_NAME

FEATURE_LEVELS = 4
# Reference counts differ by usage (each aligned with its authoritative baseline):
#   ADP nearest-neighbor matching: 8 (ADP-only fixed baseline used num_ref=8)
#   AHL DRA reference bank + cfg.nRef: 5 (AHL stage weights trained with n_ref=5)
ADP_NUM_REF = 8
AHL_NUM_REF = 5


def bridge_alpha(stage: str) -> Optional[float]:
    """Authoritative bridge v2 alpha schedule (from v2 main table)."""
    idx = int(stage[1:])
    if idx == 0:
        return None  # S0 not applicable
    if idx <= 2:
        return 0.70
    if idx <= 4:
        return 0.60
    if idx <= 7:
        return 0.35
    return 0.70  # S8


def stage_rule(stage: str) -> Tuple[str, Optional[float]]:
    """stage_recall_bestf1 rule (no FPR cap), matching the v2 main table."""
    idx = int(stage[1:])
    if idx <= 2:
        return "calib_best_f1_with_recall_ge_0.90", 0.90
    if idx <= 6:
        return "calib_best_f1_with_recall_ge_0.85", 0.85
    return "calib_best_f1", None


def robust_norm_params(calib_normal_scores: np.ndarray) -> Tuple[float, float]:
    center = float(np.median(calib_normal_scores))
    mad = float(np.median(np.abs(calib_normal_scores - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q75, q25 = np.percentile(calib_normal_scores, [75, 25])
        iqr_scale = float((q75 - q25) / 1.349)
        scale = iqr_scale if iqr_scale > 1e-12 else float(np.std(calib_normal_scores))
    if scale <= 1e-12:
        scale = 1.0
    return center, scale


def robust_norm(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (scores - center) / scale


def aggregate_score_maps(score_maps: Sequence[torch.Tensor], size: int) -> np.ndarray:
    """Upsample each scale map to image size, average, then gaussian sigma=4.

    Replicates evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py:71-82.
    """
    maps = []
    for score_map in score_maps:
        upsampled = F.interpolate(score_map.unsqueeze(1), size=size, mode="bilinear", align_corners=True)
        maps.append(upsampled.squeeze(1).detach().cpu().numpy())
    scores = np.zeros_like(maps[0])
    for item in maps:
        scores += item
    scores /= len(maps)
    for idx in range(scores.shape[0]):
        scores[idx] = gaussian_filter(scores[idx], sigma=4)
    return scores


def adp_only_score(projected: Sequence[torch.Tensor], image_size: int, feature_levels: int) -> float:
    """ADP-only image score from projected residual features (single image).

    Replicates evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py:134-142.
    """
    score_maps = []
    for feature in projected:
        bsz, dim, h, w = feature.size()
        patch = feature.permute(0, 2, 3, 1).reshape(-1, dim)
        patch_score = torch.sqrt(torch.linalg.norm(patch, dim=1) + 1.0) - 1.0
        score_maps.append(patch_score.reshape(bsz, h, w).float())
    amap = aggregate_score_maps(score_maps[:feature_levels], size=image_size)
    flat = amap.reshape(amap.shape[0], -1)
    return float(flat.max(axis=1)[0])


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def add_aucs(labels: np.ndarray, scores: np.ndarray, row: Dict[str, float]) -> Dict[str, float]:
    if len(np.unique(labels)) > 1:
        row["auc_roc"] = float(roc_auc_score(labels, scores))
        row["auc_pr"] = float(average_precision_score(labels, scores))
    else:
        row["auc_roc"] = float("nan")
        row["auc_pr"] = float("nan")
    return row


def choose_threshold(labels: np.ndarray, scores: np.ndarray, recall_floor: Optional[float]) -> Dict:
    """stage_recall_bestf1 selection. Replicates evaluate_stage_recall_bestf1_policy.py:124-141."""
    candidates = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores, float(threshold))
        candidates.append(row)
    if recall_floor is not None:
        constrained = [r for r in candidates if r["recall"] >= recall_floor]
        if constrained:
            selected = max(constrained, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
            selected["constraint_satisfied"] = True
        else:
            selected = max(candidates, key=lambda r: (r["recall"], r["f1"], r["precision"], r["threshold"]))
            selected["constraint_satisfied"] = False
    else:
        selected = max(candidates, key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]))
        selected["constraint_satisfied"] = True
    return dict(selected)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_ahl_model(weights_path: Path, n_ref: int, device: torch.device):
    if str(AHL_ROOT) not in sys.path:
        sys.path.insert(0, str(AHL_ROOT))
    from modeling.DRA_AHL import DRA

    cfg = type("Cfg", (), {"total_heads": 4, "n_scales": 2, "nRef": n_ref})()
    model = DRA(cfg, backbone="resnet18").to(device)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ahl_forward(model, bank: BackendBank, feature: torch.Tensor, feature_scale: torch.Tensor) -> float:
    image = torch.cat([bank.ahl_ref_feature, feature], dim=0)
    image_scale = torch.cat([bank.ahl_ref_scale, feature_scale], dim=0)
    outputs = model(image=image, image_scale=image_scale, label=None, var=model.parameters())
    return float(combine_dra_outputs(outputs).item())


@dataclass
class ForwardResult:
    adp_score: float
    ahl_scores: Dict[str, float]  # stage -> score
    phases: Dict[str, float]


def forward_one(
    tensor: torch.Tensor,
    pre_phases: Dict[str, float],
    backend_refs: Sequence[torch.Tensor],
    backend_refs_norm: Sequence[torch.Tensor],
    bank: BackendBank,
    ahl_models: Dict[str, object],
    encoder,
    projector,
    device: torch.device,
    image_size: int,
) -> ForwardResult:
    """Single encoder forward; derive ADP-only + per-stage AHL scores."""
    phases: Dict[str, float] = {}
    with torch.no_grad():
        features, phases["encoder_ms"] = timed(device, lambda: encode_multiscale(encoder, tensor))
        matched, phases["adp_reference_match_ms"] = timed(
            device, lambda: match_reference_features_cached(features, backend_refs, backend_refs_norm)
        )
        residual, phases["residual_ms"] = timed(device, lambda: residual_features(features, matched))
        projected, phases["projector_ms"] = timed(device, lambda: projector(*residual))
        adp_score, phases["adp_only_postprocess_ms"] = timed(
            device, lambda: adp_only_score(projected, image_size, FEATURE_LEVELS)
        )
        (feature, feature_scale), phases["compress_ms"] = timed(device, lambda: compress_four_to_two(projected))
        ahl_scores: Dict[str, float] = {}
        ahl_total = 0.0
        for stage, model in ahl_models.items():
            score, ms = timed(device, lambda m=model: ahl_forward(m, bank, feature, feature_scale))
            ahl_scores[stage] = score
            ahl_total += ms
        phases["ahl_forward_all_stages_ms"] = ahl_total
        phases["ahl_forward_per_stage_ms"] = ahl_total / max(1, len(ahl_models))
    # preprocess sub-breakdown carried from pre_phases
    for k in PREPROCESS_PHASE_KEYS:
        phases[k] = pre_phases.get(k, 0.0)
    # model floor for ADP-only (no AHL head) and for AHL (one stage head)
    common_floor = (
        phases["encoder_ms"]
        + phases["adp_reference_match_ms"]
        + phases["residual_ms"]
        + phases["projector_ms"]
    )
    phases["adp_tensor_to_threshold_ms"] = common_floor + phases["adp_only_postprocess_ms"]
    phases["ahl_tensor_to_threshold_ms"] = (
        common_floor + phases["compress_ms"] + phases["ahl_forward_per_stage_ms"]
    )
    phases["adp_decoded_image_to_threshold_ms"] = pre_phases["preprocess_total_ms"] + phases["adp_tensor_to_threshold_ms"]
    phases["ahl_decoded_image_to_threshold_ms"] = pre_phases["preprocess_total_ms"] + phases["ahl_tensor_to_threshold_ms"]
    return ForwardResult(adp_score=adp_score, ahl_scores=ahl_scores, phases=phases)


def run_backend(
    backend: str,
    train_items: Sequence[Item],
    eval_items: Sequence[Item],
    ahl_models: Dict[str, object],
    encoder,
    projector,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    image_size: int,
    warmup: int,
    n_calib: int,
) -> Dict:
    """Run one backend over eval items. Reference bank built once (shared across stages)."""
    # ADP reference features (for matching) from first ADP_NUM_REF train_normal images
    selected = list(train_items)[:ADP_NUM_REF]
    refs = None
    with torch.no_grad():
        for item in selected:
            pil = load_pil_image(item.path)
            tensor, _, _ = prepare_backend_from_pil(backend, pil, device, mean_gpu, std_gpu)
            feats = encode_multiscale(encoder, tensor)
            flat = [flatten_feature_map(x).detach() for x in feats]
            if refs is None:
                refs = [[] for _ in flat]
            for idx, fmap in enumerate(flat):
                refs[idx].append(fmap)
    backend_refs = [torch.cat(items, dim=0).to(device) for items in refs]
    backend_refs_norm = [F.normalize(ref, p=2, dim=1) for ref in backend_refs]

    # AHL reference bank (compressed feature/scale) - AHL_NUM_REF, matching DRA cfg.nRef
    bank = build_backend_bank(backend, train_items, encoder, projector, device, mean_gpu, std_gpu, AHL_NUM_REF)

    adp_scores: Dict[str, List[float]] = {"calib": [], "test": []}
    ahl_scores: Dict[str, Dict[str, List[float]]] = {s: {"calib": [], "test": []} for s in ahl_models}
    rel_paths: Dict[str, List[str]] = {"calib": [], "test": []}
    labels: Dict[str, List[int]] = {"calib": [], "test": []}
    latency_rows: List[Dict] = []

    for item_index, item in enumerate(eval_items):
        pil = load_pil_image(item.path)
        (tensor, pre_phases, _), _ = timed(
            device, lambda p=pil: prepare_backend_from_pil(backend, p, device, mean_gpu, std_gpu)
        )
        result = forward_one(
            tensor, pre_phases, backend_refs, backend_refs_norm, bank, ahl_models, encoder, projector, device, image_size
        )
        adp_scores[item.split].append(result.adp_score)
        for stage in ahl_models:
            ahl_scores[stage][item.split].append(result.ahl_scores[stage])
        rel_paths[item.split].append(item.rel_path)
        labels[item.split].append(item.label)
        timed_flag = int(item.split == "test" and item_index - n_calib >= warmup)
        latency_rows.append({"backend": backend, "split": item.split, "rel_path": item.rel_path, "timed": timed_flag, **result.phases})

    return {
        "adp_scores": adp_scores,
        "ahl_scores": ahl_scores,
        "rel_paths": rel_paths,
        "labels": labels,
        "latency_rows": latency_rows,
        "bank_setup_ms": bank.setup_ms,
    }


def derive_bridge_scores(
    stage: str,
    adp_calib: np.ndarray,
    adp_test: np.ndarray,
    ahl_calib: np.ndarray,
    ahl_test: np.ndarray,
    calib_labels: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Per-backend bridge v2 score. Returns None for S0 (N/A)."""
    alpha = bridge_alpha(stage)
    if alpha is None:
        return None
    adp_c, adp_s = robust_norm_params(adp_calib[calib_labels == 0])
    ahl_c, ahl_s = robust_norm_params(ahl_calib[calib_labels == 0])
    return {
        "calib": alpha * robust_norm(adp_calib, adp_c, adp_s) + (1.0 - alpha) * robust_norm(ahl_calib, ahl_c, ahl_s),
        "test": alpha * robust_norm(adp_test, adp_c, adp_s) + (1.0 - alpha) * robust_norm(ahl_test, ahl_c, ahl_s),
        "alpha": alpha,
    }


def method_stage_scores(results: Dict[str, Dict], stage: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Return {method: {backend: {'calib':arr,'test':arr,'alpha':..}}} for one stage."""
    out: Dict[str, Dict] = {"ADP-only-DINO": {}, "AHL-DINO": {}}
    calib_labels = {b: np.array(results[b]["labels"]["calib"], dtype=int) for b in BACKENDS}
    for backend in BACKENDS:
        r = results[backend]
        out["ADP-only-DINO"][backend] = {
            "calib": np.array(r["adp_scores"]["calib"], dtype=float),
            "test": np.array(r["adp_scores"]["test"], dtype=float),
        }
        out["AHL-DINO"][backend] = {
            "calib": np.array(r["ahl_scores"][stage]["calib"], dtype=float),
            "test": np.array(r["ahl_scores"][stage]["test"], dtype=float),
        }
    bridge = bridge_alpha(stage)
    if bridge is not None:
        out["ADP-AHL-bridge-v2"] = {}
        for backend in BACKENDS:
            b = derive_bridge_scores(
                stage,
                out["ADP-only-DINO"][backend]["calib"],
                out["ADP-only-DINO"][backend]["test"],
                out["AHL-DINO"][backend]["calib"],
                out["AHL-DINO"][backend]["test"],
                calib_labels[backend],
            )
            out["ADP-AHL-bridge-v2"][backend] = b
    return out


def build_metric_rows(results: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (performance_rows, equivalence_rows, gap_rows)."""
    perf_rows: List[Dict] = []
    equiv_rows: List[Dict] = []
    gap_rows: List[Dict] = []
    test_labels = {b: np.array(results[b]["labels"]["test"], dtype=int) for b in BACKENDS}
    calib_labels = {b: np.array(results[b]["labels"]["calib"], dtype=int) for b in BACKENDS}
    test_rels = {b: results[b]["rel_paths"]["test"] for b in BACKENDS}

    for stage in STAGES:
        _, recall_floor = stage_rule(stage)
        ms = method_stage_scores(results, stage)
        for method, per_backend in ms.items():
            if per_backend.get(CPU_BACKEND) is None:
                continue
            # CPU self-calibrated threshold (also the fixed threshold source)
            cpu_calib = per_backend[CPU_BACKEND]["calib"]
            cpu_test = per_backend[CPU_BACKEND]["test"]
            cpu_thr_row = choose_threshold(calib_labels[CPU_BACKEND], cpu_calib, recall_floor)
            fixed_threshold = float(cpu_thr_row["threshold"])
            alpha = per_backend[CPU_BACKEND].get("alpha", "")

            for backend in BACKENDS:
                pb = per_backend[backend]
                calib_s = pb["calib"]
                test_s = pb["test"]
                # mode (i): fixed CPU threshold
                fixed_m = add_aucs(test_labels[backend], test_s, confusion(test_labels[backend], test_s, fixed_threshold))
                perf_rows.append(_perf_row(method, stage, backend, "fixed_cpu_threshold", alpha, fixed_threshold, fixed_m))
                # mode (ii): self-calibrated
                self_thr = float(choose_threshold(calib_labels[backend], calib_s, recall_floor)["threshold"])
                self_m = add_aucs(test_labels[backend], test_s, confusion(test_labels[backend], test_s, self_thr))
                perf_rows.append(_perf_row(method, stage, backend, "self_calibrated", alpha, self_thr, self_m))

            # equivalence diagnostics (CPU vs GPU, test scores aligned by rel_path)
            equiv_rows.append(_equiv_row(method, stage, per_backend, test_rels, fixed_threshold))
            # gap summary (both threshold modes)
            gap_rows.extend(_gap_rows(method, stage, perf_rows))
    return perf_rows, equiv_rows, gap_rows


def _perf_row(method, stage, backend, mode, alpha, threshold, m) -> Dict:
    return {
        "method": method, "stage": stage, "backend": backend, "threshold_mode": mode,
        "alpha": alpha, "threshold": threshold,
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "accuracy": m["accuracy"],
        "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"],
        "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
    }


def _equiv_row(method, stage, per_backend, test_rels, fixed_threshold) -> Dict:
    cpu = per_backend[CPU_BACKEND]["test"]
    gpu = per_backend[GPU_BACKEND]["test"]
    cpu_rels = test_rels[CPU_BACKEND]
    gpu_map = {rel: gpu[i] for i, rel in enumerate(test_rels[GPU_BACKEND])}
    diffs = []
    changed = 0
    changed_dist = []  # distance-to-threshold of changed samples (fixed threshold)
    for i, rel in enumerate(cpu_rels):
        c = float(cpu[i]); g = float(gpu_map[rel])
        diffs.append(abs(c - g))
        cp = int(c >= fixed_threshold); gp = int(g >= fixed_threshold)
        if cp != gp:
            changed += 1
            changed_dist.append(min(abs(c - fixed_threshold), abs(g - fixed_threshold)))
    diffs = np.array(diffs, dtype=float)
    return {
        "method": method, "stage": stage,
        "score_abs_diff_mean": float(diffs.mean()),
        "score_abs_diff_median": float(np.median(diffs)),
        "score_abs_diff_p95": float(np.percentile(diffs, 95)),
        "score_abs_diff_max": float(diffs.max()),
        "pred_changed_test": int(changed),
        "changed_max_dist_to_threshold": float(max(changed_dist)) if changed_dist else 0.0,
        "changed_all_near_threshold": int(all(d <= 0.05 for d in changed_dist)) if changed_dist else 1,
        "fixed_threshold": float(fixed_threshold),
    }


def _gap_rows(method, stage, perf_rows) -> List[Dict]:
    out = []
    for mode in ["fixed_cpu_threshold", "self_calibrated"]:
        cpu = _find_perf(perf_rows, method, stage, CPU_BACKEND, mode)
        gpu = _find_perf(perf_rows, method, stage, GPU_BACKEND, mode)
        if cpu is None or gpu is None:
            continue
        f1_gap = abs(float(gpu["f1"]) - float(cpu["f1"]))
        recall_gap = abs(float(gpu["recall"]) - float(cpu["recall"]))
        aucpr_gap = abs(float(gpu["auc_pr"]) - float(cpu["auc_pr"])) if not (np.isnan(gpu["auc_pr"]) or np.isnan(cpu["auc_pr"])) else float("nan")
        passed = (f1_gap <= 0.01 and recall_gap <= 0.02 and (np.isnan(aucpr_gap) or aucpr_gap <= 0.005))
        out.append({
            "method": method, "stage": stage, "threshold_mode": mode,
            "cpu_f1": cpu["f1"], "gpu_f1": gpu["f1"], "f1_gap": f1_gap,
            "cpu_recall": cpu["recall"], "gpu_recall": gpu["recall"], "recall_gap": recall_gap,
            "cpu_auc_pr": cpu["auc_pr"], "gpu_auc_pr": gpu["auc_pr"], "auc_pr_gap": aucpr_gap,
            "gap_pass": int(passed),
        })
    return out


def _find_perf(perf_rows, method, stage, backend, mode):
    for r in perf_rows:
        if r["method"] == method and r["stage"] == stage and r["backend"] == backend and r["threshold_mode"] == mode:
            return r
    return None


def build_time_rows(results: Dict[str, Dict]) -> List[Dict]:
    """Time breakdown per backend (timed test rows only)."""
    phase_keys = [
        "adp_decoded_image_to_threshold_ms", "ahl_decoded_image_to_threshold_ms",
        "preprocess_total_ms", "adp_tensor_to_threshold_ms", "ahl_tensor_to_threshold_ms",
        "encoder_ms", "adp_reference_match_ms", "residual_ms", "projector_ms",
        "adp_only_postprocess_ms", "compress_ms", "ahl_forward_per_stage_ms",
        "image_load_decode_ms",
        # preprocess sub-breakdown
        "cpu_resize_ms", "cpu_crop_ms", "cpu_to_tensor_ms", "cpu_normalize_ms", "h2d_copy_ms",
        "pil_to_tensor_cpu_ms", "raw_h2d_copy_ms", "gpu_cast_scale_ms", "gpu_resize_ms",
        "gpu_center_crop_ms", "gpu_normalize_ms",
    ]
    rows = []
    for backend in BACKENDS:
        timed_rows = [r for r in results[backend]["latency_rows"] if int(r["timed"]) == 1]
        for phase in phase_keys:
            if not timed_rows or phase not in timed_rows[0]:
                continue
            stats = phase_stats(timed_rows, phase)
            rows.append({
                "backend": backend, "phase": phase,
                "mean_ms": stats["mean"], "median_ms": stats["median"],
                "p90_ms": stats["p90"], "p95_ms": stats["p95"], "max_ms": stats["max"],
            })
    return rows


def load_v2_main_table(path: Path) -> Dict[Tuple[str, str], Dict]:
    """Load v2 main table CSV keyed by (method, stage)."""
    if not path.exists():
        return {}
    method_map = {
        "ADP-only-DINO": "ADP-only-DINO",
        "AHL-DINO": "AHL-DINO",
        "ADP-AHL bridge v2": "ADP-AHL-bridge-v2",
    }
    out: Dict[Tuple[str, str], Dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = method_map.get(row.get("Method", "").strip())
            if m is None:
                continue
            stage = row.get("Stage", "").strip()
            out[(m, stage)] = row
    return out


def cpu_cross_check(perf_rows: List[Dict], v2: Dict[Tuple[str, str], Dict], tol: float = 0.02) -> List[Dict]:
    """Compare cpu_pil self_calibrated metrics against v2 main table; flag drift."""
    rows = []
    for r in perf_rows:
        if r["backend"] != CPU_BACKEND or r["threshold_mode"] != "self_calibrated":
            continue
        ref = v2.get((r["method"], r["stage"]))
        if not ref:
            continue

        def fv(key):
            try:
                return float(ref.get(key, "") or "nan")
            except ValueError:
                return float("nan")

        f1_ref, r_ref, aucpr_ref = fv("F1"), fv("R"), fv("AUC-PR")
        if np.isnan(f1_ref):
            continue
        f1_d = abs(r["f1"] - f1_ref)
        r_d = abs(r["recall"] - r_ref) if not np.isnan(r_ref) else float("nan")
        aucpr_d = abs(r["auc_pr"] - aucpr_ref) if not np.isnan(aucpr_ref) else float("nan")
        flag = (f1_d > tol) or (not np.isnan(r_d) and r_d > tol) or (not np.isnan(aucpr_d) and aucpr_d > tol)
        rows.append({
            "method": r["method"], "stage": r["stage"],
            "online_f1": r["f1"], "v2_f1": f1_ref, "f1_diff": f1_d,
            "online_recall": r["recall"], "v2_recall": r_ref, "recall_diff": r_d,
            "online_auc_pr": r["auc_pr"], "v2_auc_pr": aucpr_ref, "auc_pr_diff": aucpr_d,
            "drift_flag": int(flag),
        })
    return rows


def fnum(v, nd=4):
    try:
        if v == "" or v is None:
            return ""
        f = float(v)
        return "nan" if np.isnan(f) else f"{f:.{nd}f}"
    except (ValueError, TypeError):
        return str(v)


def write_master_performance(path: Path, perf_rows: List[Dict]) -> None:
    lines = [
        f"# {TASK_NAME} — Master Performance Table", "",
        "Rows = method × stage × backend × threshold_mode. Thresholds from calibration; test only reported.", "",
        "| method | stage | backend | mode | alpha | thr | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in perf_rows:
        lines.append(
            f"| {r['method']} | {r['stage']} | {r['backend']} | {r['threshold_mode']} | {fnum(r['alpha'],2)} | "
            f"{fnum(r['threshold'],4)} | {fnum(r['precision'])} | {fnum(r['recall'])} | {fnum(r['f1'])} | "
            f"{fnum(r['accuracy'])} | {fnum(r['auc_roc'])} | {fnum(r['auc_pr'])} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gap_summary(path: Path, gap_rows: List[Dict], equiv_rows: List[Dict]) -> Tuple[int, int]:
    equiv_map = {(e["method"], e["stage"]): e for e in equiv_rows}
    n_pass = sum(1 for g in gap_rows if g["gap_pass"] == 1)
    n_total = len(gap_rows)
    lines = [
        f"# {TASK_NAME} — Gap Summary (PASS/FAIL)", "",
        "PASS = F1 gap ≤ 0.01 AND Recall gap ≤ 0.02 AND AUC-PR gap ≤ 0.005 (ref_and_query).",
        "Final equivalence also requires pred_changed all near-threshold (see changed_near_threshold).", "",
        f"**Overall: {n_pass}/{n_total} (method×stage×mode) cells PASS.**", "",
        "| method | stage | mode | F1 gap | Recall gap | AUC-PR gap | pred_changed | changed_near_thr | PASS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for g in gap_rows:
        e = equiv_map.get((g["method"], g["stage"]), {})
        lines.append(
            f"| {g['method']} | {g['stage']} | {g['threshold_mode']} | {fnum(g['f1_gap'])} | {fnum(g['recall_gap'])} | "
            f"{fnum(g['auc_pr_gap'])} | {e.get('pred_changed_test','')} | {e.get('changed_all_near_threshold','')} | "
            f"{'PASS' if g['gap_pass'] else 'FAIL'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_pass, n_total


def write_time_table(path: Path, time_rows: List[Dict]) -> None:
    lines = [f"# {TASK_NAME} — Time Table", "", "Timed over steady test images (batch=1, GPU-synced). ms.", ""]
    lines += ["| backend | phase | mean | median | p90 | p95 | max |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in time_rows:
        lines.append(
            f"| {r['backend']} | {r['phase']} | {fnum(r['mean_ms'],2)} | {fnum(r['median_ms'],2)} | "
            f"{fnum(r['p90_ms'],2)} | {fnum(r['p95_ms'],2)} | {fnum(r['max_ms'],2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_check(path: Path, cc_rows: List[Dict]) -> int:
    n_drift = sum(r["drift_flag"] for r in cc_rows)
    lines = [
        f"# {TASK_NAME} — CPU Cross-Check vs v2 main table", "",
        "cpu_pil online (self_calibrated) vs summary/20260603 v2 main table. Drift if |diff|>0.02.", "",
        f"**Drift cells: {n_drift}/{len(cc_rows)}** (0 expected if online ADP/AHL/bridge replicate the offline pipeline).", "",
        "| method | stage | online F1 | v2 F1 | F1 diff | online R | v2 R | R diff | online AUC-PR | v2 AUC-PR | drift |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in cc_rows:
        lines.append(
            f"| {r['method']} | {r['stage']} | {fnum(r['online_f1'])} | {fnum(r['v2_f1'])} | {fnum(r['f1_diff'])} | "
            f"{fnum(r['online_recall'])} | {fnum(r['v2_recall'])} | {fnum(r['recall_diff'])} | "
            f"{fnum(r['online_auc_pr'])} | {fnum(r['v2_auc_pr'])} | {'DRIFT' if r['drift_flag'] else 'ok'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_drift


def write_decision_handoff(path: Path, n_pass: int, n_total: int, n_drift: int, time_rows: List[Dict], gap_rows: List[Dict]) -> None:
    def tval(backend, phase):
        for r in time_rows:
            if r["backend"] == backend and r["phase"] == phase:
                return r["median_ms"]
        return float("nan")

    fails = [g for g in gap_rows if g["gap_pass"] == 0]
    cpu_pre = tval(CPU_BACKEND, "preprocess_total_ms")
    gpu_pre = tval(GPU_BACKEND, "preprocess_total_ms")
    cpu_ahl_dec = tval(CPU_BACKEND, "ahl_decoded_image_to_threshold_ms")
    gpu_ahl_dec = tval(GPU_BACKEND, "ahl_decoded_image_to_threshold_ms")
    cpu_floor = tval(CPU_BACKEND, "ahl_tensor_to_threshold_ms")
    gpu_floor = tval(GPU_BACKEND, "ahl_tensor_to_threshold_ms")
    verdict = "全阶段检测等价确证" if not fails else f"{len(fails)} 个 cell FAIL，需诊断"
    lines = [
        f"# {TASK_NAME} — Decision Handoff", "",
        "## 一页结论", "",
        f"- 检测能力门槛: **{n_pass}/{n_total}** cell PASS（F1 gap≤0.01 ∧ Recall gap≤0.02 ∧ AUC-PR gap≤0.005）。",
        f"- CPU 交叉校验漂移 cell: **{n_drift}**（0 = 在线 ADP/AHL/bridge 复刻无偏）。",
        f"- 判定: **{verdict}**。", "",
        "## 时间（median, ms, ref_and_query, AHL 主口径）", "",
        f"- preprocess: CPU {fnum(cpu_pre,2)} vs GPU {fnum(gpu_pre,2)}",
        f"- AHL decoded-image-to-threshold: CPU {fnum(cpu_ahl_dec,2)} vs GPU {fnum(gpu_ahl_dec,2)}",
        f"- 模型段地板 ahl_tensor_to_threshold: CPU {fnum(cpu_floor,2)} vs GPU {fnum(gpu_floor,2)}（应近似相等）", "",
        "## FAIL 阶段" if fails else "## 无 FAIL 阶段", "",
    ]
    for g in fails:
        lines.append(f"- {g['method']} {g['stage']} [{g['threshold_mode']}]: F1 gap {fnum(g['f1_gap'])}, Recall gap {fnum(g['recall_gap'])}, AUC-PR gap {fnum(g['auc_pr_gap'])}")
    lines += [
        "", "## 推荐", "",
        "见 master_performance_table / gap_summary / equivalence_diagnostics / time_table / cpu_cross_check。",
        "F1 偶尔略超 CPU 一律按持平+噪声解读，不叙述 GPU 更优。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    p.add_argument("--stage-root-base", type=Path, default=STAGE_ROOT_BASE)
    p.add_argument("--ahl-weights-base", type=Path, default=AHL_WEIGHTS_BASE)
    p.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    p.add_argument("--projector-checkpoint", type=Path, default=PROJECTOR_CKPT)
    p.add_argument("--v2-main-table", type=Path, default=V2_MAIN_TABLE)
    p.add_argument("--alias", default=ALIAS)
    p.add_argument("--stages", default=",".join(STAGES), help="comma list, e.g. S7,S8 for smoke")
    p.add_argument("--manifest-stage", default="S8", help="stage whose manifest supplies the shared calib/test/ref lists")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-test", type=int, default=None, help="cap test items per split half for smoke")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global STAGES
    STAGES = [s.strip() for s in args.stages.split(",") if s.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full-stage eval requires CUDA.")
    # tf32 consistent for both backends
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    add_adpretrain_to_path(str(args.adpretrain_root))
    encoder = make_encoder("dinov2-large", str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    encoder.eval(); projector.eval()
    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # shared calib/test/ref item lists from one manifest (calib/test identical across stages)
    manifest = load_stage_manifest(args.stage_root_base / args.manifest_stage, args.alias)
    train_items = collect_items(manifest, "train_normal", "train", 0)
    calib_items = collect_items(manifest, "calib_normal", "calib", 0) + collect_items(manifest, "calib_anomaly", "calib", 1)
    test_items = collect_items(manifest, "test_normal", "test", 0) + collect_items(manifest, "test_anomaly", "test", 1)
    if args.max_test is not None:
        calib_items = collect_items(manifest, "calib_normal", "calib", 0)[: args.max_test] + collect_items(manifest, "calib_anomaly", "calib", 1)[: args.max_test]
        test_items = collect_items(manifest, "test_normal", "test", 0)[: args.max_test] + collect_items(manifest, "test_anomaly", "test", 1)[: args.max_test]
    eval_items = calib_items + test_items
    n_calib = len(calib_items)

    ahl_models = {s: load_ahl_model(args.ahl_weights_base / s / "ahl" / f"{args.alias}_ctest.pkl", AHL_NUM_REF, device) for s in STAGES}
    image_size = 224

    results = {}
    for backend in BACKENDS:
        results[backend] = run_backend(
            backend, train_items, eval_items, ahl_models, encoder, projector, device,
            mean_gpu, std_gpu, image_size, args.warmup, n_calib,
        )

    perf_rows, equiv_rows, gap_rows = build_metric_rows(results)
    time_rows = build_time_rows(results)
    v2 = load_v2_main_table(args.v2_main_table)
    cc_rows = cpu_cross_check(perf_rows, v2)

    out = args.output_root; out.mkdir(parents=True, exist_ok=True)
    summ = args.summary_root; summ.mkdir(parents=True, exist_ok=True)
    write_csv(out / "master_performance_table.csv", perf_rows)
    write_csv(out / "equivalence_diagnostics.csv", equiv_rows)
    write_csv(out / "gap_rows.csv", gap_rows)
    write_csv(out / "time_table.csv", time_rows)
    write_csv(out / "cpu_cross_check.csv", cc_rows)
    write_master_performance(out / "master_performance_table.md", perf_rows)
    n_pass, n_total = write_gap_summary(out / "gap_summary.md", gap_rows, equiv_rows)
    write_time_table(out / "time_table.md", time_rows)
    n_drift = write_cross_check(out / "cpu_cross_check.md", cc_rows)
    write_decision_handoff(out / "decision_handoff.md", n_pass, n_total, n_drift, time_rows, gap_rows)

    config = {
        "task": TASK_NAME, "stages": STAGES, "backends": BACKENDS,
        "adp_num_ref": ADP_NUM_REF, "ahl_num_ref": AHL_NUM_REF,
        "feature_levels": FEATURE_LEVELS, "bridge_alpha": {s: bridge_alpha(s) for s in STAGES},
        "manifest_stage": args.manifest_stage, "warmup": args.warmup, "max_test": args.max_test,
        "seed": args.seed, "device": str(device), "tf32": False,
        "counts": {"train_ref_pool": len(train_items), "calib": len(calib_items), "test": len(test_items)},
        "torch_version": torch.__version__,
    }
    write_json(out / "config_snapshot.json", config)
    for name in ["master_performance_table.csv", "master_performance_table.md", "equivalence_diagnostics.csv",
                 "gap_rows.csv", "gap_summary.md", "time_table.csv", "time_table.md", "cpu_cross_check.csv",
                 "cpu_cross_check.md", "decision_handoff.md", "config_snapshot.json"]:
        (summ / name).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "status": "ok", "task": TASK_NAME, "stages": STAGES,
        "gap_pass": n_pass, "gap_total": n_total, "cross_check_drift": n_drift,
        "output_root": str(out), "summary_root": str(summ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()








