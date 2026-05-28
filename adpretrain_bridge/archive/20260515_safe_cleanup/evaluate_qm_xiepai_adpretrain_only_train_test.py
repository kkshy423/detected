#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ADPretrain-only feature-norm scores on the qm_xiepai train/test split.

This script does not train or fine-tune ADPretrain. It reuses the exported
official CLIP-B16 projected residual feature cache and follows the same
image-level best-F1 metric policy used for the AHL comparison.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fewshot_qm_xiepai_common import (
    ALIAS_CLASS,
    CACHE_BASE,
    STAGE_SPECS,
    SOURCE_CLASS_ROOT,
    read_lines,
    write_json,
)


DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test")
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_adpretrain_only_clip_b16_plain_train_test_20260512"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_BASE / "plain_official_img_angle")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--stages", nargs="+", default=list(STAGE_SPECS.keys()))
    parser.add_argument("--topk", type=int, default=1)
    return parser.parse_args()


def metric_at(labels, scores, threshold):
    preds = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def best_f1(labels, scores):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2.0 * precisions * recalls) / (precisions + recalls + 1e-12)
    if len(thresholds) == 0:
        row = metric_at(labels, scores, float(scores.max()) if len(scores) else 0.0)
        row["source_precision_recall_curve_f1"] = float(f1_scores[0]) if len(f1_scores) else row["f1"]
        return row
    valid = f1_scores[:-1]
    idx = int(np.nanargmax(valid))
    row = metric_at(labels, scores, float(thresholds[idx]))
    row["source_precision_recall_curve_f1"] = float(valid[idx])
    return row


def read_split(split_root, stage):
    stage_root = split_root / stage
    return {
        "train_normal": read_lines(stage_root / "train_normal.txt"),
        "train_anomaly": read_lines(stage_root / "train_anomaly.txt"),
        "test_normal": read_lines(split_root / "test_normal.txt"),
        "test_anomaly": read_lines(split_root / "test_anomaly.txt"),
    }


def feature_path(cache_class_root, scale, source_rel):
    return cache_class_root / scale / Path(source_rel).with_suffix(".npy")


def resize_nearest(arr, target_hw=(14, 14)):
    if arr.shape == target_hw:
        return arr
    h, w = arr.shape
    th, tw = target_hw
    yi = np.minimum((np.arange(th) * h / th).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(tw) * w / tw).astype(np.int64), w - 1)
    return arr[yi][:, xi]


def score_one(cache_class_root, source_rel, topk):
    maps = []
    for scale in ("feature", "feature_scale"):
        path = feature_path(cache_class_root, scale, source_rel)
        if not path.exists():
            raise FileNotFoundError(path)
        feat = np.load(str(path)).astype(np.float32)
        patch_norm = np.linalg.norm(feat, axis=0)
        patch_score = np.sqrt(patch_norm + 1.0) - 1.0
        maps.append(resize_nearest(patch_score, (14, 14)))
    score_map = np.mean(np.stack(maps, axis=0), axis=0)
    flat = np.sort(score_map.reshape(-1))
    k = max(1, min(int(topk), flat.size))
    return float(np.mean(flat[-k:]))


def evaluate_stage(args, stage):
    split = read_split(args.split_root, stage)
    cache_class_root = args.cache_root / args.alias
    rows = []
    for role, label in (("test_normal", 0), ("test_anomaly", 1)):
        for source_rel in split[role]:
            rows.append({
                "stage": stage,
                "role": role,
                "source_rel": source_rel,
                "label": label,
                "score": score_one(cache_class_root, source_rel, args.topk),
            })
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    primary = best_f1(labels, scores)
    spec = STAGE_SPECS[stage]
    out = {
        "stage": stage,
        "status": "ok",
        "method": "adpretrain_only_cache_feature_norm",
        "feature_cache": str(args.cache_root),
        "source_class_root": str(SOURCE_CLASS_ROOT),
        "split_root": str(args.split_root),
        "score_rule": "mean of feature and feature_scale patch L2-norm maps; image score is top-k mean",
        "topk": args.topk,
        "primary_policy": "adpretrain_eval_best_f1",
        "primary": primary,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "test_counts": {
            "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()),
        },
        "train_counts": {
            "normal": spec.available_normal,
            "anomaly": spec.available_anomaly,
        },
        "adpretrain_training": "not_run; official CLIP-B16 projected residual feature cache is reused",
        "notes": [
            "This is a no-AHL baseline from the same feature cache used by plain AHL.",
            "Scores do not use stage labels; stages are repeated for table alignment with AHL.",
            "Precision/Recall/F1 use the same fixed-test best-F1 policy as the AHL train/test-only reports.",
        ],
    }
    stage_dir = args.output_root / "stages" / stage
    metrics_dir = stage_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "metrics.json", out)
    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "role", "label", "score", "source_rel"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# ADPretrain-only {stage}",
        "",
        "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Threshold |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {primary['precision']:.4f} | {primary['recall']:.4f} | {primary['f1']:.4f} | "
            f"{primary['accuracy']:.4f} | {out['auc_roc']:.4f} | {out['auc_pr']:.4f} | "
            f"{primary['threshold']:.6f} |"
        ),
        "",
    ]
    (stage_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [evaluate_stage(args, stage) for stage in args.stages]
    summary_dir = args.output_root / "metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "summary.json", {"status": "ok", "results": results})
    lines = [
        "# ADPretrain-only CLIP-B16 Plain Train/Test Summary",
        "",
        "| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        p = item["primary"]
        tc = item["train_counts"]
        lines.append(
            f"| {item['stage']} | {tc['normal']}/{tc['anomaly']} | {p['precision']:.4f} | "
            f"{p['recall']:.4f} | {p['f1']:.4f} | {p['accuracy']:.4f} | "
            f"{item['auc_roc']:.4f} | {item['auc_pr']:.4f} |"
        )
    lines += [
        "",
        "Note: ADPretrain is not trained or fine-tuned here; the same fixed-test cache scores are repeated across stages for alignment with AHL.",
        "",
    ]
    (args.output_root / "result.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
