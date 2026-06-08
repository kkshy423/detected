#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8 GPU-preprocess AHL retrain smoke test.

This script keeps ADP/AHL model definitions unchanged. It only exports S8
features with the selected GPU preprocessing backend, retrains AHL on that
feature cache, then compares CPU baseline, GPU no-retrain, and GPU retrain
with the same calibration-derived threshold policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_preprocess_equivalence_paired import (
    AHL_ROOT,
    CPU_BACKEND,
    MEAN,
    STD,
    BackendBank,
    Item,
    PREPROCESS_PHASE_KEYS,
    build_backend_bank,
    choose_backend_threshold,
    collect_items,
    flatten_feature_map,
    load_pil_image,
    load_stage_manifest,
    metrics_at,
    numeric_stats,
    phase_stats,
    prepare_backend_from_pil,
    score_tensor,
    timed,
)
from common import (
    add_adpretrain_to_path,
    compress_four_to_two,
    ensure_dataset_links,
    list_image_files,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
)
from fewshot_qm_xiepai_common import ALIAS_CLASS, BRIDGE_ROOT, stage_output_dir


TASK_NAME = "20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1"
GPU_BACKEND = "gpu_tensor_torchvision_aa_true"
STAGE = "S8"
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "output" / TASK_NAME
DEFAULT_SUMMARY_ROOT = BRIDGE_ROOT / "summary" / TASK_NAME
DEFAULT_SPLIT_ROOT = BRIDGE_ROOT / "splits" / "20260529_qm_xiepai_6_1_fixed_180_70_val49"
DEFAULT_CPU_CACHE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm"
)
DEFAULT_GPU_CACHE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_tensor_aa_true_s8_retrain_smoke_v1"
)
DEFAULT_CPU_STAGE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_plain_dino_large_norm_val49/S8"
)
DEFAULT_GPU_STAGE_ROOT_BASE = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_plain_dino_large_norm_gpu_tensor_aa_true_s8_retrain_smoke_v1"
)
DEFAULT_ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
DEFAULT_PROJECTOR = Path("/ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth")
DEFAULT_CPU_AHL_WEIGHTS = (
    BRIDGE_ROOT
    / "output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages/S8/ahl/models_qiumianxiepai_ctest.pkl"
)
DEFAULT_POLICY = "strategy_mild_stage_v2_1_safe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cpu-cache-root", type=Path, default=DEFAULT_CPU_CACHE_ROOT)
    parser.add_argument("--gpu-cache-root", type=Path, default=DEFAULT_GPU_CACHE_ROOT)
    parser.add_argument("--cpu-stage-root", type=Path, default=DEFAULT_CPU_STAGE_ROOT)
    parser.add_argument("--gpu-stage-root-base", type=Path, default=DEFAULT_GPU_STAGE_ROOT_BASE)
    parser.add_argument("--adpretrain-root", type=Path, default=DEFAULT_ADPRETRAIN_ROOT)
    parser.add_argument("--projector-checkpoint", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--cpu-ahl-weights", type=Path, default=DEFAULT_CPU_AHL_WEIGHTS)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--threshold-policy", default=DEFAULT_POLICY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adp-export-num-ref", type=int, default=8)
    parser.add_argument("--online-n-ref", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_all_images(class_root: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for split in ["train", "val", "test"]:
        split_root = class_root / split
        if not split_root.exists():
            continue
        for class_dir in sorted([p for p in split_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for image_path in list_image_files(class_dir):
                rows.append((image_path.relative_to(class_root).as_posix(), image_path))
    return rows


def save_feature_pair(cache_class_root: Path, rel_path: str, feature: torch.Tensor, feature_scale: torch.Tensor) -> None:
    rel = Path(rel_path)
    feature_path = cache_class_root / "feature" / rel.with_suffix(".npy")
    scale_path = cache_class_root / "feature_scale" / rel.with_suffix(".npy")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature.squeeze(0).detach().cpu().numpy().astype(np.float32))
    np.save(scale_path, feature_scale.squeeze(0).detach().cpu().numpy().astype(np.float32))


def cache_complete(cache_class_root: Path, expected_count: int) -> bool:
    runtime = cache_class_root / "feature_runtime.json"
    if not runtime.exists():
        return False
    feature_count = len(list((cache_class_root / "feature").glob("**/*.npy"))) if (cache_class_root / "feature").exists() else 0
    scale_count = (
        len(list((cache_class_root / "feature_scale").glob("**/*.npy")))
        if (cache_class_root / "feature_scale").exists()
        else 0
    )
    return feature_count >= expected_count and scale_count >= expected_count


def build_adp_reference_features(
    ref_paths: Sequence[Path],
    encoder,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    refs = None
    with torch.no_grad():
        for path in ref_paths:
            pil = load_pil_image(path)
            tensor, _, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
            features = list(encoder.encode_image_from_tensors(tensor, return_global=False, shape="img"))
            flat = [flatten_feature_map(x).detach() for x in features]
            if refs is None:
                refs = [[] for _ in flat]
            for idx, fmap in enumerate(flat):
                refs[idx].append(fmap)
    if refs is None:
        raise ValueError("No ADP reference features were built")
    ref_tensors = [torch.cat(items, dim=0).to(device) for items in refs]
    ref_norm = [F.normalize(ref, p=2, dim=1) for ref in ref_tensors]
    return ref_tensors, ref_norm


def export_gpu_preprocess_cache(args: argparse.Namespace, encoder, projector, device: torch.device) -> Dict:
    source_class_root = args.cpu_cache_root / args.alias
    cache_class_root = args.gpu_cache_root / args.alias
    all_items = list_all_images(source_class_root)
    if not all_items:
        raise ValueError(f"No source images found under {source_class_root}")
    if cache_complete(cache_class_root, len(all_items)):
        return {
            "status": "reused",
            "cache_class_root": str(cache_class_root),
            "sample_count": len(all_items),
            "runtime_file": str(cache_class_root / "feature_runtime.json"),
        }
    if cache_class_root.exists() and any(cache_class_root.iterdir()):
        raise FileExistsError(f"GPU cache exists but is incomplete; refusing to overwrite: {cache_class_root}")

    ensure_dataset_links(source_class_root, cache_class_root)
    (cache_class_root / "feature").mkdir(parents=True, exist_ok=True)
    (cache_class_root / "feature_scale").mkdir(parents=True, exist_ok=True)

    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    train_good = sorted(list_image_files(source_class_root / "train" / "good"), key=lambda p: p.name)
    ref_paths = train_good[: args.adp_export_num_ref]
    if not ref_paths:
        raise ValueError(f"No ADP references under {source_class_root / 'train/good'}")
    refs, ref_norm = build_adp_reference_features(ref_paths, encoder, device, mean_gpu, std_gpu)

    phase_preprocess: List[float] = []
    phase_encoder: List[float] = []
    phase_match_project: List[float] = []
    total_times: List[float] = []
    counts: Dict[str, int] = {}

    with torch.no_grad():
        for rel_path, image_path in all_items:
            counts[rel_path.split("/", 1)[0]] = counts.get(rel_path.split("/", 1)[0], 0) + 1
            pil = load_pil_image(image_path)
            tensor, pre_phases, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
            _, enc_ms = timed(device, lambda: None)
            del enc_ms
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            features = list(encoder.encode_image_from_tensors(tensor, return_global=False, shape="img"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            matched = match_reference_features(features, refs)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            feature, feature_scale = compress_four_to_two(projected)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            save_feature_pair(cache_class_root, rel_path, feature, feature_scale)
            preprocess_ms = float(pre_phases["preprocess_total_ms"])
            phase_preprocess.append(preprocess_ms)
            phase_encoder.append((t2 - t1) * 1000.0)
            phase_match_project.append((t3 - t2) * 1000.0)
            total_times.append(preprocess_ms + (t3 - t1) * 1000.0)

    runtime = {
        "status": "ok",
        "task": TASK_NAME,
        "preprocess_backend": GPU_BACKEND,
        "time_kind": "single_image_adpretrain_feature_mean_ms",
        "time_adpretrain_feature_ms": float(np.mean(total_times)),
        "time_adpretrain_projected_feature_ms": float(np.mean(total_times)),
        "time_load_transform_ms": float(np.mean(phase_preprocess)),
        "time_adpretrain_encoder_ms": float(np.mean(phase_encoder)),
        "time_adpretrain_projector_ms": float(np.mean(phase_match_project)),
        "preprocess_stats": numeric_stats(phase_preprocess),
        "encoder_stats": numeric_stats(phase_encoder),
        "match_project_stats": numeric_stats(phase_match_project),
        "total_stats": numeric_stats(total_times),
        "sample_count": int(len(total_times)),
        "reference_count": int(len(ref_paths)),
        "reference_paths": [str(p) for p in ref_paths],
    }
    write_json(cache_class_root / "feature_runtime.json", runtime)
    meta = {
        "status": "ok",
        "task": TASK_NAME,
        "source_class_root": str(source_class_root),
        "cache_class_root": str(cache_class_root),
        "alias": args.alias,
        "backbone": "dinov2-large",
        "projector_checkpoint": str(args.projector_checkpoint),
        "preprocess_backend": GPU_BACKEND,
        "adp_export_num_ref": args.adp_export_num_ref,
        "counts": counts,
    }
    write_json(args.gpu_cache_root / "feature_cache_manifest.json", meta)
    write_json(cache_class_root / "adpretrain_bridge_meta.json", meta)
    return {"status": "exported", "cache_class_root": str(cache_class_root), "sample_count": len(total_times)}


def run_ahl_retrain(args: argparse.Namespace) -> Path:
    retrain_weights = stage_output_dir(args.output_root, STAGE) / "ahl" / f"{args.alias}_ctest.pkl"
    if retrain_weights.exists():
        return retrain_weights
    cmd = [
        sys.executable,
        str(BRIDGE_ROOT / "run_fewshot_ahl_stage_val_threshold.py"),
        "--stage",
        STAGE,
        "--split-root",
        str(args.split_root),
        "--cache-root",
        str(args.gpu_cache_root),
        "--stage-root-base",
        str(args.gpu_stage_root_base),
        "--output-root",
        str(args.output_root),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--n-ref",
        str(args.online_n_ref),
        "--reuse-stage-root",
    ]
    subprocess.run(cmd, cwd=str(BRIDGE_ROOT), check=True)
    eval_cmd = [
        sys.executable,
        str(BRIDGE_ROOT / "evaluate_fewshot_stage_metrics_val_threshold.py"),
        "--stage",
        STAGE,
        "--output-root",
        str(args.output_root),
        "--primary-policy",
        args.threshold_policy,
    ]
    subprocess.run(eval_cmd, cwd=str(BRIDGE_ROOT), check=True)
    if not retrain_weights.exists():
        raise FileNotFoundError(retrain_weights)
    return retrain_weights


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


def group_metric_row(
    group: str,
    threshold: float,
    calib_labels: Sequence[int],
    calib_scores: Sequence[float],
    test_labels: Sequence[int],
    test_scores: Sequence[float],
) -> Dict:
    calib = metrics_at(calib_labels, calib_scores, threshold)
    test = metrics_at(test_labels, test_scores, threshold)
    return {
        "group": group,
        "threshold": threshold,
        "calib_acc": calib["accuracy"],
        "calib_p": calib["precision"],
        "calib_r": calib["recall"],
        "calib_f1": calib["f1"],
        "test_acc": test["accuracy"],
        "test_p": test["precision"],
        "test_r": test["recall"],
        "test_f1": test["f1"],
        "auc_roc": test["auc_roc"],
        "auc_pr": test["auc_pr"],
        "tp": test["tp"],
        "fp": test["fp"],
        "tn": test["tn"],
        "fn": test["fn"],
    }


def quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def score_distribution_rows(score_rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for group in sorted({r["group"] for r in score_rows}):
        for split in ["calib", "test"]:
            for label in [0, 1]:
                vals = [float(r["score"]) for r in score_rows if r["group"] == group and r["split"] == split and int(r["label"]) == label]
                if not vals:
                    continue
                out.append(
                    {
                        "group": group,
                        "split": split,
                        "label": label,
                        "n": len(vals),
                        "p05": quantile(vals, 5),
                        "p10": quantile(vals, 10),
                        "p25": quantile(vals, 25),
                        "p50": quantile(vals, 50),
                        "p75": quantile(vals, 75),
                        "p90": quantile(vals, 90),
                        "p95": quantile(vals, 95),
                        "max": max(vals),
                    }
                )
    return out


def evaluate_online_groups(args: argparse.Namespace, encoder, projector, retrain_weights: Path, device: torch.device) -> Dict:
    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    manifest = load_stage_manifest(args.cpu_stage_root, args.alias)
    train_items = collect_items(manifest, "train_normal", "train", 0)
    calib_items = collect_items(manifest, "calib_normal", "calib", 0) + collect_items(manifest, "calib_anomaly", "calib", 1)
    test_items = collect_items(manifest, "test_normal", "test", 0) + collect_items(manifest, "test_anomaly", "test", 1)
    all_items = calib_items + test_items
    labels = {"calib": [x.label for x in calib_items], "test": [x.label for x in test_items]}

    banks: Dict[str, BackendBank] = {
        CPU_BACKEND: build_backend_bank(CPU_BACKEND, train_items, encoder, projector, device, mean_gpu, std_gpu, args.online_n_ref),
        GPU_BACKEND: build_backend_bank(GPU_BACKEND, train_items, encoder, projector, device, mean_gpu, std_gpu, args.online_n_ref),
    }
    cpu_model = load_ahl_model(args.cpu_ahl_weights, args.online_n_ref, device)
    retrain_model = load_ahl_model(retrain_weights, args.online_n_ref, device)
    groups = [
        {"name": "cpu_baseline", "backend": CPU_BACKEND, "bank": CPU_BACKEND, "model": cpu_model},
        {"name": "gpu_no_retrain", "backend": GPU_BACKEND, "bank": GPU_BACKEND, "model": cpu_model},
        {"name": "gpu_retrain", "backend": GPU_BACKEND, "bank": GPU_BACKEND, "model": retrain_model},
    ]

    score_rows: List[Dict] = []
    latency_rows: List[Dict] = []
    group_scores = {g["name"]: {"calib": [], "test": []} for g in groups}

    for item_index, item in enumerate(all_items):
        pil = load_pil_image(item.path)
        prepared = {
            CPU_BACKEND: prepare_backend_from_pil(CPU_BACKEND, pil, device, mean_gpu, std_gpu),
            GPU_BACKEND: prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu),
        }
        group_order = groups[item_index % len(groups) :] + groups[: item_index % len(groups)]
        for score_position, group in enumerate(group_order, start=1):
            backend = group["backend"]
            tensor, pre_phases, _ = prepared[backend]
            score, _, score_phases = score_tensor(
                tensor,
                banks[group["bank"]],
                encoder,
                projector,
                group["model"],
                device,
                threshold=0.0,
            )
            group_scores[group["name"]][item.split].append(score)
            score_rows.append(
                {
                    "group": group["name"],
                    "split": item.split,
                    "role": item.role,
                    "label": item.label,
                    "rel_path": item.rel_path,
                    "score": score,
                }
            )
            phases = dict(score_phases)
            for key in PREPROCESS_PHASE_KEYS:
                phases[key] = pre_phases.get(key, 0.0)
            phases["decoded_image_to_threshold_ms"] = pre_phases["preprocess_total_ms"] + score_phases["tensor_to_threshold_ms"]
            latency_rows.append(
                {
                    "group": group["name"],
                    "backend": backend,
                    "split": item.split,
                    "role": item.role,
                    "label": item.label,
                    "rel_path": item.rel_path,
                    "score_position": score_position,
                    "timed": int(item.split == "test" and item_index - len(calib_items) >= args.warmup),
                    **phases,
                }
            )

    metric_rows: List[Dict] = []
    thresholds = {}
    for group in [g["name"] for g in groups]:
        threshold_row = choose_backend_threshold(args.threshold_policy, STAGE, labels["calib"], group_scores[group]["calib"])
        threshold = float(threshold_row["threshold"])
        thresholds[group] = threshold
        metric_rows.append(group_metric_row(group, threshold, labels["calib"], group_scores[group]["calib"], labels["test"], group_scores[group]["test"]))

    baseline_test = [r for r in score_rows if r["group"] == "cpu_baseline" and r["split"] == "test"]
    baseline_by_rel = {r["rel_path"]: r for r in baseline_test}
    changed_rows: List[Dict] = []
    for row in [r for r in score_rows if r["group"] != "cpu_baseline" and r["split"] == "test"]:
        base = baseline_by_rel[row["rel_path"]]
        base_pred = int(float(base["score"]) >= thresholds["cpu_baseline"])
        pred = int(float(row["score"]) >= thresholds[row["group"]])
        if base_pred == pred:
            continue
        changed_rows.append(
            {
                "group": row["group"],
                "rel_path": row["rel_path"],
                "role": row["role"],
                "label": row["label"],
                "cpu_score": base["score"],
                "cpu_threshold": thresholds["cpu_baseline"],
                "cpu_pred": base_pred,
                "group_score": row["score"],
                "group_threshold": thresholds[row["group"]],
                "group_pred": pred,
                "score_diff": float(row["score"]) - float(base["score"]),
            }
        )

    phase_keys = [
        "decoded_image_to_threshold_ms",
        "preprocess_total_ms",
        "tensor_to_threshold_ms",
        "cpu_resize_ms",
        "cpu_crop_ms",
        "cpu_to_tensor_ms",
        "cpu_normalize_ms",
        "h2d_copy_ms",
        "pil_to_tensor_cpu_ms",
        "raw_h2d_copy_ms",
        "gpu_cast_scale_ms",
        "gpu_resize_ms",
        "gpu_center_crop_ms",
        "gpu_normalize_ms",
        "encoder_ms",
        "adp_reference_match_ms",
        "residual_ms",
        "projector_ms",
        "compress_ms",
        "ahl_reference_attach_ms",
        "ahl_forward_ms",
        "score_postprocess_ms",
        "threshold_ms",
    ]
    latency_summary_rows: List[Dict] = []
    for group in [g["name"] for g in groups]:
        timed_rows = [r for r in latency_rows if r["group"] == group and int(r["timed"]) == 1]
        for phase in phase_keys:
            stats = phase_stats(timed_rows, phase)
            latency_summary_rows.append(
                {
                    "group": group,
                    "phase": phase,
                    "mean_ms": stats["mean"],
                    "median_ms": stats["median"],
                    "p90_ms": stats["p90"],
                    "p95_ms": stats["p95"],
                    "max_ms": stats["max"],
                }
            )

    return {
        "metric_rows": metric_rows,
        "score_rows": score_rows,
        "score_distribution_rows": score_distribution_rows(score_rows),
        "changed_rows": changed_rows,
        "latency_summary_rows": latency_summary_rows,
        "thresholds": thresholds,
        "counts": {
            "train_ref_pool": len(train_items),
            "calib": len(calib_items),
            "test": len(test_items),
            "steady_test": max(0, len(test_items) - args.warmup),
        },
    }


def metric_by_group(rows: Sequence[Dict], group: str) -> Dict:
    for row in rows:
        if row["group"] == group:
            return row
    raise KeyError(group)


def latency_value(rows: Sequence[Dict], group: str, phase: str, field: str = "median_ms") -> float:
    for row in rows:
        if row["group"] == group and row["phase"] == phase:
            return float(row[field])
    raise KeyError((group, phase, field))


def write_markdown(args: argparse.Namespace, results: Dict, export_info: Dict, retrain_weights: Path) -> None:
    metrics = results["metric_rows"]
    latency = results["latency_summary_rows"]
    cpu = metric_by_group(metrics, "cpu_baseline")
    gpu_no = metric_by_group(metrics, "gpu_no_retrain")
    gpu_re = metric_by_group(metrics, "gpu_retrain")
    f1_gap = abs(float(gpu_re["test_f1"]) - float(cpu["test_f1"]))
    auc_pr_gap = abs(float(gpu_re["auc_pr"]) - float(cpu["auc_pr"]))
    recall_drop = float(cpu["test_r"]) - float(gpu_re["test_r"])
    gpu_re_median = latency_value(latency, "gpu_retrain", "decoded_image_to_threshold_ms")
    continue_gpu = f1_gap <= 0.02 and auc_pr_gap <= 0.02 and recall_drop <= 0.02 and gpu_re_median <= 40.0

    lines = [
        f"# {TASK_NAME}",
        "",
        "## Setup",
        "",
        f"- Stage: `{STAGE}`",
        f"- GPU preprocessing backend: `{GPU_BACKEND}`",
        f"- Threshold policy: `{args.threshold_policy}`",
        f"- CPU AHL weights: `{args.cpu_ahl_weights}`",
        f"- GPU retrain weights: `{retrain_weights}`",
        f"- GPU cache root: `{args.gpu_cache_root}`",
        f"- Export status: `{export_info['status']}`",
        f"- Online n_ref: `{args.online_n_ref}`; ADP export num_ref: `{args.adp_export_num_ref}`",
        "",
        "## Metrics",
        "",
        "| Group | threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['group']} | {float(row['threshold']):.6f} | {float(row['test_p']):.4f} | "
            f"{float(row['test_r']):.4f} | {float(row['test_f1']):.4f} | {float(row['test_acc']):.4f} | "
            f"{float(row['auc_roc']):.4f} | {float(row['auc_pr']):.4f} | {int(row['tp'])} | {int(row['fp'])} | "
            f"{int(row['tn'])} | {int(row['fn'])} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Group | decoded median | decoded P95 | preprocess median | tensor-to-threshold median |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in ["cpu_baseline", "gpu_no_retrain", "gpu_retrain"]:
        lines.append(
            f"| {group} | {latency_value(latency, group, 'decoded_image_to_threshold_ms'):.2f} | "
            f"{latency_value(latency, group, 'decoded_image_to_threshold_ms', 'p95_ms'):.2f} | "
            f"{latency_value(latency, group, 'preprocess_total_ms'):.2f} | "
            f"{latency_value(latency, group, 'tensor_to_threshold_ms'):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Changed Samples Vs CPU Baseline",
            "",
            f"- Changed rows: `{len(results['changed_rows'])}`",
            "",
            "## Decision",
            "",
            f"- GPU retrain F1 gap vs CPU baseline: `{f1_gap:.4f}`",
            f"- GPU retrain AUC-PR gap vs CPU baseline: `{auc_pr_gap:.4f}`",
            f"- GPU retrain Recall drop vs CPU baseline: `{recall_drop:.4f}`",
            f"- GPU retrain decoded-image-to-threshold median: `{gpu_re_median:.2f} ms`",
            f"- Decision: `{'continue_gpu_preprocess_full_stage' if continue_gpu else 'stop_gpu_preprocess_replacement_route'}`",
        ]
    )
    if continue_gpu:
        lines.append("- Interpretation: GPU retrain is close enough to CPU baseline for this smoke criterion; it can be considered for a full-stage follow-up.")
    else:
        lines.append("- Interpretation: GPU retrain still does not recover CPU-PIL baseline closely enough; keep CPU-PIL as production metric/backend.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.summary_root / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This smoke test requires CUDA.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)

    add_adpretrain_to_path(str(args.adpretrain_root))
    encoder = make_encoder("dinov2-large", str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    encoder.eval()
    projector.eval()

    export_info = export_gpu_preprocess_cache(args, encoder, projector, device)
    retrain_weights = run_ahl_retrain(args)
    results = evaluate_online_groups(args, encoder, projector, retrain_weights, device)

    write_csv(args.output_root / "metrics_summary.csv", results["metric_rows"])
    write_csv(args.output_root / "score_distribution_summary.csv", results["score_distribution_rows"])
    write_csv(args.output_root / "changed_samples_vs_cpu.csv", results["changed_rows"])
    write_csv(args.output_root / "latency_summary.csv", results["latency_summary_rows"])
    write_csv(args.output_root / "online_scores.csv", results["score_rows"])
    for name in ["metrics_summary.csv", "score_distribution_summary.csv", "changed_samples_vs_cpu.csv", "latency_summary.csv", "online_scores.csv"]:
        target = args.summary_root / name
        target.write_text((args.output_root / name).read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "task": TASK_NAME,
        "stage": STAGE,
        "preprocess_backend": GPU_BACKEND,
        "output_root": str(args.output_root),
        "summary_root": str(args.summary_root),
        "split_root": str(args.split_root),
        "cpu_cache_root": str(args.cpu_cache_root),
        "gpu_cache_root": str(args.gpu_cache_root),
        "cpu_stage_root": str(args.cpu_stage_root),
        "gpu_stage_root_base": str(args.gpu_stage_root_base),
        "adpretrain_root": str(args.adpretrain_root),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cpu_ahl_weights": str(args.cpu_ahl_weights),
        "gpu_retrain_weights": str(retrain_weights),
        "threshold_policy": args.threshold_policy,
        "adp_export_num_ref": args.adp_export_num_ref,
        "online_n_ref": args.online_n_ref,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "warmup": args.warmup,
        "device": str(device),
        "torch_version": torch.__version__,
        "export_info": export_info,
        "counts": results["counts"],
        "thresholds": results["thresholds"],
    }
    write_json(args.output_root / "config_snapshot.json", config)
    write_json(args.summary_root / "config_snapshot.json", config)
    write_markdown(args, results, export_info, retrain_weights)
    print(json.dumps({"status": "ok", "output_root": str(args.output_root), "summary_root": str(args.summary_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
