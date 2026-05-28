#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate official ADPretrain CLIP-B16 scoring on the fixed qm_xiepai test set.

This is the no-AHL baseline. It does not train ADPretrain or the projector.
It loads the official CLIP-B16 projector checkpoint, uses the same first normal
references as the exported cache path, computes ADPretrain residual feature
norm maps, then evaluates image-level scores on the fixed 140 normal + 60
anomaly test split.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    build_reference_features,
    build_transform,
    encode_multiscale,
    list_image_files,
    load_image_tensor,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
)
from fewshot_qm_xiepai_common import (
    ADPRETRAIN_ROOT,
    OFFICIAL_CLIP_PROJECTOR,
    SOURCE_CLASS_ROOT,
    STAGE_SPECS,
    read_lines,
    write_json,
)


DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test")
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    parser.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CLIP_PROJECTOR)
    parser.add_argument("--backbone", default="clip-base")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-levels", type=int, default=4)
    parser.add_argument("--stage-table", nargs="+", default=list(STAGE_SPECS.keys()))
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


def source_items(split_root, source_class_root):
    rows = []
    for role, label in (("test_normal", 0), ("test_anomaly", 1)):
        for rel in read_lines(split_root / f"{role}.txt"):
            rows.append({"role": role, "label": label, "source_rel": rel, "path": source_class_root / rel})
    return rows


def main():
    args = parse_args()
    metrics_dir = args.output_root / "metrics"
    if (metrics_dir / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing metrics: {metrics_dir / 'metrics.json'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    requested = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), requested)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.checkpoint), requested)
    encoder.eval()
    projector.eval()

    ref_paths = list_image_files(args.source_class_root / "train" / "good")[: args.num_ref]
    refs = build_reference_features(encoder, ref_paths, transform, requested, args.num_ref)
    rows = source_items(args.split_root, args.source_class_root)

    score_rows = []
    inference_times = []
    with torch.no_grad():
        for row in rows:
            image = load_image_tensor(row["path"], transform, requested)
            size = image.shape[-1]
            if requested.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            features = encode_multiscale(encoder, image)
            matched = match_reference_features(features, refs)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            score_maps = []
            for feature in projected:
                bsz, dim, h, w = feature.size()
                patch = feature.permute(0, 2, 3, 1).reshape(-1, dim)
                patch_score = torch.sqrt(torch.linalg.norm(patch, dim=1) + 1.0) - 1.0
                score_maps.append(patch_score.reshape(bsz, h, w).float())
            amap = aggregate_score_maps(score_maps[: args.feature_levels], size=size)
            image_score = image_score_from_map(amap)[0]
            if requested.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - start) * 1000.0)
            score_rows.append({
                "role": row["role"],
                "label": int(row["label"]),
                "score": float(image_score),
                "source_rel": row["source_rel"],
            })

    labels = np.asarray([r["label"] for r in score_rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in score_rows], dtype=np.float64)
    primary = best_f1(labels, scores)
    result = {
        "status": "ok",
        "method": "ADPretrain-only official CLIP-B16 projected residual feature norm",
        "primary_policy": "adpretrain_eval_best_f1",
        "primary": primary,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "test_counts": {
            "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()),
        },
        "score_rule": "official ADPretrain residual -> projector -> per-layer L2 norm -> bilinear upsample -> average -> gaussian sigma=4 -> image max/top1",
        "backbone": args.backbone,
        "checkpoint": str(args.checkpoint),
        "num_ref": args.num_ref,
        "reference_paths": [str(p) for p in ref_paths],
        "source_class_root": str(args.source_class_root),
        "split_root": str(args.split_root),
        "device": str(requested),
        "inference_time_ms": float(np.mean(inference_times)) if inference_times else None,
        "stage_table_note": "No ADPretrain training or fine-tuning is performed; the same fixed-test result is repeated by stage only for comparison with AHL and YOLO curves.",
    }
    write_json(metrics_dir / "metrics.json", result)
    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "label", "score", "source_rel"])
        writer.writeheader()
        writer.writerows(score_rows)

    lines = [
        "# ADPretrain-only Official CLIP-B16 Fixed-Test Summary",
        "",
        "| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in args.stage_table:
        spec = STAGE_SPECS[stage]
        lines.append(
            f"| {stage} | {spec.available_normal}/{spec.available_anomaly} | "
            f"{primary['precision']:.4f} | {primary['recall']:.4f} | {primary['f1']:.4f} | "
            f"{primary['accuracy']:.4f} | {result['auc_roc']:.4f} | {result['auc_pr']:.4f} | "
            f"{result['inference_time_ms']:.4f} |"
        )
    lines.extend([
        "",
        "Note: this is a single no-training ADPretrain-only evaluation on the fixed 140 normal + 60 anomaly test set.",
        "",
    ])
    (args.output_root / "result.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
