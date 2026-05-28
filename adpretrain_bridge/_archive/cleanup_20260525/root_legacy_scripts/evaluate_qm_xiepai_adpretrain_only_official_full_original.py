#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate official ADPretrain CLIP-B16 scoring on the full original test set.

This is a no-AHL, no-finetune baseline. The normal train/good images are used
only as ADPretrain references. Evaluation uses the original test split:
test/good as normal and test/defect as anomaly.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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
from evaluate_qm_xiepai_adpretrain_only_official_train_test import (
    aggregate_score_maps,
    best_f1,
    image_score_from_map,
)
from fewshot_qm_xiepai_common import ADPRETRAIN_ROOT, OFFICIAL_CLIP_PROJECTOR, SOURCE_CLASS_ROOT, write_json


DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_adpretrain_only_clip_b16_official_full_original_20260513"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    parser.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CLIP_PROJECTOR)
    parser.add_argument("--backbone", default="clip-base")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-levels", type=int, default=4)
    return parser.parse_args()


def source_items(source_class_root):
    rows = []
    for path in list_image_files(source_class_root / "test" / "good"):
        rows.append({"role": "test_normal", "label": 0, "source_rel": str(path.relative_to(source_class_root)), "path": path})
    for path in list_image_files(source_class_root / "test" / "defect"):
        rows.append({"role": "test_anomaly", "label": 1, "source_rel": str(path.relative_to(source_class_root)), "path": path})
    return rows


def main():
    args = parse_args()
    metrics_dir = args.output_root / "metrics"
    if (metrics_dir / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing metrics: {metrics_dir / 'metrics.json'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.checkpoint), device)
    encoder.eval()
    projector.eval()

    train_good = list_image_files(args.source_class_root / "train" / "good")
    ref_paths = train_good[: args.num_ref]
    refs = build_reference_features(encoder, ref_paths, transform, device, args.num_ref)
    rows = source_items(args.source_class_root)

    score_rows = []
    inference_times = []
    with torch.no_grad():
        for row in rows:
            image = load_image_tensor(row["path"], transform, device)
            size = image.shape[-1]
            if device.type == "cuda":
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
            if device.type == "cuda":
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
        "eval_name": "full_original_test_good_plus_all_test_defect",
        "primary_policy": "adpretrain_eval_best_f1",
        "primary": primary,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "test_counts": {
            "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()),
            "total": int(labels.size),
        },
        "data_inventory": {
            "reference_train_good_total": len(train_good),
            "eval_test_good_total": int((labels == 0).sum()),
            "eval_test_defect_total": int((labels == 1).sum()),
        },
        "score_rule": "official ADPretrain residual -> projector -> per-layer L2 norm -> bilinear upsample -> average -> gaussian sigma=4 -> image max/top1",
        "backbone": args.backbone,
        "checkpoint": str(args.checkpoint),
        "num_ref": args.num_ref,
        "reference_paths": [str(p) for p in ref_paths],
        "source_class_root": str(args.source_class_root),
        "device": str(device),
        "inference_time_ms": float(np.mean(inference_times)) if inference_times else None,
        "notes": [
            "No ADPretrain/projector training or fine-tuning is performed.",
            "train/good is used only as normal reference data and is not included in evaluation metrics.",
            "Evaluation uses all original test/good normal images and all original test/defect anomaly images.",
        ],
    }
    write_json(metrics_dir / "metrics.json", result)
    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "label", "score", "source_rel"])
        writer.writeheader()
        writer.writerows(score_rows)

    lines = [
        "# ADPretrain-only Official CLIP-B16 Full Original Test",
        "",
        "| Eval normal | Eval anomaly | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {result['test_counts']['normal']} | {result['test_counts']['anomaly']} | "
            f"{primary['precision']:.4f} | {primary['recall']:.4f} | {primary['f1']:.4f} | "
            f"{primary['accuracy']:.4f} | {result['auc_roc']:.4f} | {result['auc_pr']:.4f} | "
            f"{result['inference_time_ms']:.4f} |"
        ),
        "",
        "This run uses the same official ADPretrain-only scoring as the fixed-test run, but evaluates all original test defects.",
        "",
    ]
    (args.output_root / "result.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
