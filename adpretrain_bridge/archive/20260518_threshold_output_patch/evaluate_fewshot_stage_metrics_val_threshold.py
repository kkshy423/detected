#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one AHL stage using validation-side thresholds and held-out test metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fewshot_qm_xiepai_common import ALIAS_CLASS, config_dir, metrics_dir, read_json, read_result_rows, stage_output_dir, write_json
from threshold_policies import apply_thresholds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=[f"S{i}" for i in range(9)])
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--alias", default=ALIAS_CLASS)
    p.add_argument("--ahl-subdir", default="ahl")
    p.add_argument("--primary-policy", default="val_best_f1")
    return p.parse_args()


def fmt(v):
    return "" if v is None else f"{float(v):.4f}"


def main():
    args = parse_args()
    manifest_path = config_dir(args.output_root) / f"{args.stage}_stage_manifest.json"
    manifest = read_json(manifest_path)
    result_file = stage_output_dir(args.output_root, args.stage) / args.ahl_subdir / "result.txt"
    result_rows = read_result_rows(result_file)
    eval_order = manifest["eval_order"]
    mappings = {row["stage_rel"]: row for row in manifest["mappings"]}
    if len(result_rows) < len(eval_order):
        raise ValueError(f"Not enough result rows: {len(result_rows)} < {len(eval_order)}")
    last_rows = result_rows[-len(eval_order):]
    score_rows = []
    for stage_rel, (label, score) in zip(eval_order, last_rows):
        meta = mappings[stage_rel]
        score_rows.append({
            "role": meta["role"],
            "label": int(label),
            "score": float(score),
            "stage_rel": stage_rel,
            "source_rel": meta["source_rel"],
        })
    calib_rows = [r for r in score_rows if r["role"].startswith("calib_")]
    test_rows = [r for r in score_rows if r["role"].startswith("test_")]
    calib_labels = np.asarray([r["label"] for r in calib_rows], dtype=np.int64)
    calib_scores = np.asarray([r["score"] for r in calib_rows], dtype=np.float64)
    test_labels = np.asarray([r["label"] for r in test_rows], dtype=np.int64)
    test_scores = np.asarray([r["score"] for r in test_rows], dtype=np.float64)
    if len(set(calib_labels.tolist())) != 2 or len(set(test_labels.tolist())) != 2:
        raise ValueError("Calibration and test sets must both contain normal and anomaly labels")
    threshold_results = apply_thresholds(calib_labels, calib_scores, test_labels, test_scores)
    primary = threshold_results[args.primary_policy]["test"]
    out = {
        "stage": args.stage,
        "status": "ok",
        "method": "AHL plain no-CHMM + ADPretrain official CLIP-B16 features",
        "primary_policy": args.primary_policy,
        "primary": primary,
        "threshold_results": threshold_results,
        "auc_roc": float(roc_auc_score(test_labels, test_scores)),
        "auc_pr": float(average_precision_score(test_labels, test_scores)),
        "calibration_auc_roc": float(roc_auc_score(calib_labels, calib_scores)),
        "calibration_auc_pr": float(average_precision_score(calib_labels, calib_scores)),
        "train_counts": {
            "normal": int(manifest["counts"]["train_normal"]),
            "anomaly": int(manifest["counts"]["train_anomaly"]),
        },
        "calibration_counts": {
            "normal": int((calib_labels == 0).sum()),
            "anomaly": int((calib_labels == 1).sum()),
        },
        "test_counts": {
            "normal": int((test_labels == 0).sum()),
            "anomaly": int((test_labels == 1).sum()),
        },
        "score_file": str(metrics_dir(args.output_root, args.stage) / "scores.csv"),
        "result_file": str(result_file),
        "manifest_file": str(manifest_path),
        "notes": [
            "Primary threshold is selected on validation/calibration scores and applied unchanged to held-out test scores.",
            "test_best_f1_oracle is reported only for comparison with previous offline tables and is not the primary deployable policy.",
        ],
    }
    mdir = metrics_dir(args.output_root, args.stage)
    mdir.mkdir(parents=True, exist_ok=True)
    write_json(mdir / "metrics.json", out)
    with (mdir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "label", "score", "stage_rel", "source_rel"])
        writer.writeheader()
        writer.writerows(score_rows)
    lines = [
        f"# AHL Val-Threshold {args.stage}",
        "",
        f"Primary policy: `{args.primary_policy}`",
        "",
        "| P | R | F1 | Acc | AUC-ROC | AUC-PR |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(primary['precision'])} | {fmt(primary['recall'])} | {fmt(primary['f1'])} | {fmt(primary['accuracy'])} | {fmt(out['auc_roc'])} | {fmt(out['auc_pr'])} |",
        "",
        "| Policy | Val threshold | Val F1 | Test P | Test R | Test F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in threshold_results.items():
        cal = value.get("calibration") or {}
        test = value["test"]
        lines.append(f"| {key} | {fmt(test['threshold'])} | {fmt(cal.get('f1'))} | {fmt(test['precision'])} | {fmt(test['recall'])} | {fmt(test['f1'])} |")
    lines.append("")
    (mdir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (stage_output_dir(args.output_root, args.stage) / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
