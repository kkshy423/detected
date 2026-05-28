#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ADPretrain-only on the 20260519 fixed 180/79 split.

This is a no-AHL, no-finetune baseline. It uses stage train_normal images only
as ADPretrain reference images, selects thresholds from the fixed calibration
set, and evaluates the fixed 180 normal / 79 anomaly test set.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (
    build_reference_features,
    build_transform,
    encode_multiscale,
    load_image_tensor,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
)
from summarize_fixed_180_79_strategy_mild_v2_1_safe import production_p95, safe_v2_1
from threshold_policies import metric_at
from fewshot_qm_xiepai_common import ADPRETRAIN_ROOT, OFFICIAL_CLIP_PROJECTOR, read_lines, write_json

DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/20260519_qm_xiepai_6_1_fixed_180_79")
DEFAULT_OUTPUT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_adpretrain_only_fixed_180_79_all_stage_p95_safe")
DEFAULT_SOURCE_PARENT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1")
DEFAULT_STAGES = [f"S{i}" for i in range(9)]
PRIMARY_POLICY = "strategy_mild_stage_v2_1_safe"
REPORT_POLICIES = ["production_normal_p95", "strategy_mild_stage_v2_1_safe"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=None)
    parser.add_argument("--source-parent", type=Path, default=DEFAULT_SOURCE_PARENT)
    parser.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CLIP_PROJECTOR)
    parser.add_argument("--backbone", default="clip-base")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-levels", type=int, default=4)
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    return parser.parse_args()


def resolve_source_class_root(args) -> Path:
    if args.source_class_root is not None:
        return args.source_class_root
    candidates = sorted(args.source_parent.glob("models__*"))
    usable = [p for p in candidates if (p / "train" / "good").is_dir() and (p / "test" / "defect").is_dir()]
    if len(usable) != 1:
        raise RuntimeError(f"Expected one usable models__* class root under {args.source_parent}, got: {usable}")
    return usable[0]


def aggregate_score_maps(score_maps, size):
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


def image_score_from_map(score_map):
    flat = score_map.reshape(score_map.shape[0], -1)
    return flat.max(axis=1)


def metric_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else None,
        "auc_pr": float(average_precision_score(labels, scores)) if len(set(labels.tolist())) > 1 else None,
    }


def split_items(split_root: Path, stage: str, source_class_root: Path) -> List[Dict]:
    rows = []
    specs = [
        (split_root / stage / "train_normal.txt", "reference_normal", "reference", 0),
        (split_root / "calib_normal.txt", "calib_normal", "val", 0),
        (split_root / "calib_anomaly.txt", "calib_anomaly", "val", 1),
        (split_root / "test_normal.txt", "test_normal", "test", 0),
        (split_root / "test_anomaly.txt", "test_anomaly", "test", 1),
    ]
    for path, role, split, label in specs:
        rels = read_lines(path)
        for rel in rels:
            rows.append({"role": role, "split": split, "label": label, "source_rel": rel, "path": source_class_root / rel})
    return rows


def eval_items(rows, encoder, projector, refs, transform, device, feature_levels: int):
    score_rows = []
    feature_times = []
    post_times = []
    total_times = []
    with torch.no_grad():
        for row in rows:
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_start = time.perf_counter()
            image = load_image_tensor(row["path"], transform, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            features = encode_multiscale(encoder, image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            feature_times.append((time.perf_counter() - total_start) * 1000.0)
            post_start = time.perf_counter()
            matched = match_reference_features(features, refs)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            score_maps = []
            for feature in projected:
                bsz, dim, h, w = feature.size()
                patch = feature.permute(0, 2, 3, 1).reshape(-1, dim)
                patch_score = torch.sqrt(torch.linalg.norm(patch, dim=1) + 1.0) - 1.0
                score_maps.append(patch_score.reshape(bsz, h, w).float())
            size = image.shape[-1]
            amap = aggregate_score_maps(score_maps[:feature_levels], size=size)
            image_score = image_score_from_map(amap)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            post_times.append((time.perf_counter() - post_start) * 1000.0)
            total_times.append((time.perf_counter() - total_start) * 1000.0)
            score_rows.append({
                "role": row["role"],
                "split": row["split"],
                "label": int(row["label"]),
                "score": float(image_score),
                "source_rel": row["source_rel"],
            })
    return score_rows, feature_times, post_times, total_times


def labels_scores(score_rows, split_name: str):
    part = [r for r in score_rows if r["split"] == split_name]
    labels = np.asarray([r["label"] for r in part], dtype=np.int64)
    scores = np.asarray([r["score"] for r in part], dtype=np.float64)
    return labels, scores


def write_scores(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "split", "label", "score", "source_rel"])
        writer.writeheader()
        writer.writerows(rows)


def format_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def main():
    args = parse_args()
    source_class_root = resolve_source_class_root(args)
    if (args.output_root / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_root / 'summary.json'}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.checkpoint), device)
    encoder.eval()
    projector.eval()

    summary_rows = []
    summary_payload = {
        "status": "ok",
        "method": "ADPretrain-only official CLIP-B16 projected residual feature norm",
        "primary_policy": PRIMARY_POLICY,
        "split_root": str(args.split_root),
        "source_class_root": str(source_class_root),
        "checkpoint": str(args.checkpoint),
        "backbone": args.backbone,
        "num_ref": args.num_ref,
        "device": str(device),
        "stages": {},
        "notes": [
            "No ADPretrain/projector training or fine-tuning is performed.",
            "Stage train_normal images are used only as normal reference images.",
            "All deployable thresholds are selected on the fixed calibration set; test is used only for final evaluation.",
        ],
    }

    for stage in args.stages:
        stage_root = args.output_root / "stages" / stage
        metrics_dir = stage_root / "metrics"
        if (metrics_dir / "metrics.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing stage metrics: {metrics_dir / 'metrics.json'}")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        ref_rels = read_lines(args.split_root / stage / "train_normal.txt")[: args.num_ref]
        ref_paths = [source_class_root / rel for rel in ref_rels]
        missing = [str(p) for p in ref_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing reference images for {stage}: {missing[:5]}")
        refs = build_reference_features(encoder, ref_paths, transform, device, args.num_ref)

        rows = [r for r in split_items(args.split_root, stage, source_class_root) if r["split"] != "reference"]
        missing_eval = [str(r["path"]) for r in rows if not r["path"].exists()]
        if missing_eval:
            raise FileNotFoundError(f"Missing eval images for {stage}: {missing_eval[:5]}")
        score_rows, feature_times, post_times, total_times = eval_items(rows, encoder, projector, refs, transform, device, args.feature_levels)
        write_scores(metrics_dir / "scores.csv", score_rows)

        calib_labels, calib_scores = labels_scores(score_rows, "val")
        test_labels, test_scores = labels_scores(score_rows, "test")
        production = production_p95(calib_labels, calib_scores, stage)
        safe = safe_v2_1(calib_labels, calib_scores, stage)
        thresholds = {
            "production_normal_p95": {
                "calibration": production,
                "test": metric_at(test_labels, test_scores, float(production["threshold"])),
            },
            "strategy_mild_stage_v2_1_safe": {
                "calibration": safe,
                "test": metric_at(test_labels, test_scores, float(safe["threshold"])),
            },
        }
        primary = thresholds[PRIMARY_POLICY]["test"]
        stage_metrics = {
            "status": "ok",
            "stage": stage,
            "method": summary_payload["method"],
            "primary_policy": PRIMARY_POLICY,
            "primary": primary,
            "threshold_results": thresholds,
            "calibration_auc": metric_auc(calib_labels, calib_scores),
            "test_auc": metric_auc(test_labels, test_scores),
            "time_adpretrain_feature_ms": float(np.mean(feature_times)) if feature_times else None,
            "time_ahl_process_ms": float(np.mean(post_times)) if post_times else None,
            "time_total_ms": float(np.mean(total_times)) if total_times else None,
            "time_kind": "single_image_mean_ms",
            "time_note": "ADPretrain-only path: feature time includes image load/transform plus encoder feature extraction; AHL process field is reused for residual matching, projection, score map aggregation, and image score extraction.",
            "counts": {
                "reference_normal": len(ref_paths),
                "calib_normal": int((calib_labels == 0).sum()),
                "calib_anomaly": int((calib_labels == 1).sum()),
                "test_normal": int((test_labels == 0).sum()),
                "test_anomaly": int((test_labels == 1).sum()),
            },
            "reference_paths": [str(p) for p in ref_paths],
            "score_file": str(metrics_dir / "scores.csv"),
            "inference_time_ms": float(np.mean(total_times)) if total_times else None,
        }
        write_json(metrics_dir / "metrics.json", stage_metrics)

        lines = [
            f"# ADPretrain-only Fixed 180/79 {stage}",
            "",
            "| Policy | Threshold | P | R | Acc | F1 | ADP feature ms | Postprocess ms | Total ms | TP | FP | TN | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in REPORT_POLICIES:
            if policy not in thresholds:
                continue
            row = thresholds[policy]["test"]
            lines.append(
                f"| {policy} | {row.get('threshold', 0.0):.4f} | {row.get('precision', 0.0):.4f} | "
                f"{row.get('recall', 0.0):.4f} | {row.get('accuracy', 0.0):.4f} | {row.get('f1', 0.0):.4f} | "
                f"{format_float(stage_metrics['time_adpretrain_feature_ms'])} | {format_float(stage_metrics['time_ahl_process_ms'])} | "
                f"{format_float(stage_metrics['time_total_ms'])} | {int(row.get('tp', 0))} | {int(row.get('fp', 0))} | "
                f"{int(row.get('tn', 0))} | {int(row.get('fn', 0))} |"
            )
        lines.extend([
            "",
            f"Test AUC-ROC: {format_float(stage_metrics['test_auc']['auc_roc'])}; AUC-PR: {format_float(stage_metrics['test_auc']['auc_pr'])}.",
            "Both reported policies are selected on calibration data only.",
            "",
        ])
        (stage_root / "result.md").write_text("\n".join(lines), encoding="utf-8")
        stage_metrics["result_file"] = str(stage_root / "result.md")
        write_json(metrics_dir / "metrics.json", stage_metrics)

        for policy in REPORT_POLICIES:
            if policy not in thresholds:
                continue
            test_row = thresholds[policy]["test"]
            calib_row = thresholds[policy].get("calibration", {})
            summary_rows.append({
                "stage": stage,
                "policy": policy,
                "status": "ok",
                "threshold": test_row.get("threshold"),
                "calib_precision": calib_row.get("precision"),
                "calib_recall": calib_row.get("recall"),
                "calib_f1": calib_row.get("f1"),
                "calib_fpr": calib_row.get("calibration_fpr"),
                "calib_fp": calib_row.get("fp"),
                "calib_fn": calib_row.get("fn"),
                "test_precision": test_row.get("precision"),
                "test_recall": test_row.get("recall"),
                "test_f1": test_row.get("f1"),
                "test_accuracy": test_row.get("accuracy"),
                "test_tp": test_row.get("tp"),
                "test_fp": test_row.get("fp"),
                "test_tn": test_row.get("tn"),
                "test_fn": test_row.get("fn"),
                "test_auc_roc": stage_metrics["test_auc"]["auc_roc"],
                "test_auc_pr": stage_metrics["test_auc"]["auc_pr"],
                "inference_time_ms": stage_metrics["inference_time_ms"],
            })
        summary_payload["stages"][stage] = stage_metrics

    write_json(args.output_root / "summary.json", summary_payload)
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md = [
        "# ADPretrain-only Fixed 180/79 S0-S8",
        "",
        "Two deployable thresholds are selected on calibration data only: `production_normal_p95` and `strategy_mild_stage_v2_1_safe`.",
        "",
        "| Stage | Policy | P | R | F1 | Acc | FP | FN | AUC-ROC | AUC-PR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["policy"] not in ["production_normal_p95", PRIMARY_POLICY]:
            continue
        md.append(
            f"| {row['stage']} | {row['policy']} | {format_float(row['test_precision'])} | {format_float(row['test_recall'])} | "
            f"{format_float(row['test_f1'])} | {format_float(row['test_accuracy'])} | {int(row['test_fp'])} | {int(row['test_fn'])} | "
            f"{format_float(row['test_auc_roc'])} | {format_float(row['test_auc_pr'])} |"
        )
    md.extend([
        "",
        "Initial conclusion should be made after reading metrics.json for every stage; this file is a compact index only.",
        "",
    ])
    (args.output_root / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(args.output_root), "stages": args.stages}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
