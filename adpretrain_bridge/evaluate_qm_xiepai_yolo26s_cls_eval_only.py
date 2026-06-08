#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate pre-trained YOLO26s-cls stage weights on a new split only."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fewshot_qm_xiepai_common import read_json, write_json
from run_qm_xiepai_yolo26s_cls_stage import (
    DEFAULT_MODEL,
    defect_class_index,
    ensure_ultralytics,
    list_images,
)
from threshold_policies import apply_thresholds


DEFAULT_DATA_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_6_1_cls_val_threshold_20260529")
DEFAULT_OUTPUT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only")
DEFAULT_WEIGHTS_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[f"S{i}" for i in range(9)])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--primary-policy", default="strategy_stage_adaptive")
    return parser.parse_args()


def sync_if_cuda() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def score_split(model, defect_idx: int, stage_root: Path, split: str, imgsz: int, device: str) -> Tuple[List[Dict], np.ndarray, np.ndarray, Dict]:
    rows: List[Dict] = []
    elapsed: List[float] = []
    for class_name, label in [("good", 0), ("defect", 1)]:
        for path in list_images(stage_root / split / class_name):
            sync_if_cuda()
            start = time.perf_counter()
            pred = model.predict(source=str(path), imgsz=imgsz, device=device, verbose=False)[0]
            sync_if_cuda()
            elapsed.append((time.perf_counter() - start) * 1000.0)
            probs = pred.probs.data.detach().cpu().numpy()
            rows.append({
                "split": split,
                "path": str(path),
                "class": class_name,
                "label": int(label),
                "score": float(probs[defect_idx]),
            })
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    timing = {
        "time_kind": "single_image_yolo_predict_mean_ms",
        "mean_ms": float(np.mean(elapsed)) if elapsed else None,
        "total_ms": float(np.sum(elapsed)) if elapsed else None,
        "sample_count": len(elapsed),
    }
    return rows, labels, scores, timing


def stage_weights(weights_root: Path, stage: str) -> Path:
    best = weights_root / "stages" / stage / "ultralytics" / "train" / "weights" / "best.pt"
    last = weights_root / "stages" / stage / "ultralytics" / "train" / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No YOLO weights found for {stage} under {weights_root}")


def main() -> None:
    args = parse_args()
    stage_root = args.data_root / args.stage
    output_stage = args.output_root / "stages" / args.stage
    metrics_dir = output_stage / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(stage_root / "manifest.json")
    counts = manifest["counts"]
    if counts["train_anomaly"] == 0:
        skipped = {
            "stage": args.stage,
            "status": "skipped",
            "reason": "YOLO supervised training requires at least one anomaly sample; S0 has none.",
            "train_counts": {"normal": counts["train_normal"], "anomaly": counts["train_anomaly"]},
            "test_counts": {"normal": counts["test_normal"], "anomaly": counts["test_anomaly"]},
        }
        write_json(metrics_dir / "metrics.json", skipped)
        print(json.dumps(skipped, ensure_ascii=False), flush=True)
        return

    ensure_ultralytics()
    from ultralytics import YOLO

    weights = stage_weights(args.weights_root, args.stage)
    model = YOLO(str(weights))
    defect_idx = defect_class_index(model.names)

    val_rows, val_labels, val_scores, val_timing = score_split(model, defect_idx, stage_root, "val", args.imgsz, args.device)
    test_rows, test_labels, test_scores, test_timing = score_split(model, defect_idx, stage_root, "test", args.imgsz, args.device)
    threshold_results = apply_thresholds(val_labels, val_scores, test_labels, test_scores, stage=args.stage)
    primary = threshold_results[args.primary_policy]["test"]

    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path", "class", "label", "score"])
        writer.writeheader()
        writer.writerows(val_rows + test_rows)

    metrics = {
        "stage": args.stage,
        "status": "ok",
        "method": "YOLO26s-cls supervised baseline eval-only",
        "weights": str(weights),
        "primary_policy": args.primary_policy,
        "primary": primary,
        "threshold_results": threshold_results,
        "auc_roc": float(roc_auc_score(test_labels, test_scores)),
        "auc_pr": float(average_precision_score(test_labels, test_scores)),
        "calibration_auc_roc": float(roc_auc_score(val_labels, val_scores)),
        "calibration_auc_pr": float(average_precision_score(val_labels, val_scores)),
        "train_counts": {"normal": counts["train_normal"], "anomaly": counts["train_anomaly"]},
        "calibration_counts": {"normal": counts["val_normal"], "anomaly": counts["val_anomaly"]},
        "test_counts": {"normal": counts["test_normal"], "anomaly": counts["test_anomaly"]},
        "defect_class_index": int(defect_idx),
        "time_kind": "single_image_prediction_mean_ms",
        "time_ahl_process_ms": float(test_timing["mean_ms"]) if test_timing["mean_ms"] is not None else None,
        "time_yolo_predict_ms": float(test_timing["mean_ms"]) if test_timing["mean_ms"] is not None else None,
        "time_total_ms": float(test_timing["mean_ms"]) if test_timing["mean_ms"] is not None else None,
        "calibration_timing": val_timing,
        "test_timing": test_timing,
        "score_file": str(metrics_dir / "scores.csv"),
    }
    write_json(metrics_dir / "metrics.json", metrics)

    lines = [
        f"# YOLO26s-cls {args.stage} eval-only",
        "",
        f"Primary policy: `{args.primary_policy}`",
        "",
        "| P | R | Acc | F1 | Val pred ms/img | Test pred ms/img | AUC-ROC | AUC-PR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {primary['precision']:.4f} | {primary['recall']:.4f} | {primary['accuracy']:.4f} | {primary['f1']:.4f} | "
        f"{(val_timing['mean_ms'] or 0.0):.4f} | {(test_timing['mean_ms'] or 0.0):.4f} | "
        f"{metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} |",
        "",
    ]
    (output_stage / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "stage": args.stage, "metrics": str(metrics_dir / 'metrics.json')}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
