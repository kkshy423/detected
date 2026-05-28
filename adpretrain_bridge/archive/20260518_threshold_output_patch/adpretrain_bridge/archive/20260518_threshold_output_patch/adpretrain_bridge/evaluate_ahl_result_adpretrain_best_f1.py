#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a generic AHL result.txt block with ADPretrain-style best-F1."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, f1_score

from fewshot_qm_xiepai_common import read_result_rows, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generic AHL result.txt using the last eval block.")
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--label", default="full_original_plain")
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
        return metric_at(labels, scores, float(scores.max()) if len(scores) else 0.0)
    valid = f1_scores[:-1]
    idx = int(np.nanargmax(valid))
    row = metric_at(labels, scores, float(thresholds[idx]))
    row["source_precision_recall_curve_f1"] = float(valid[idx])
    return row


def main():
    args = parse_args()
    rows = read_result_rows(args.result_file)
    if len(rows) < args.eval_size:
        raise ValueError(f"Not enough rows: {len(rows)} < {args.eval_size}")
    block = rows[-args.eval_size:]
    labels = np.asarray([x[0] for x in block], dtype=np.int64)
    scores = np.asarray([x[1] for x in block], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Evaluation block must contain both classes")
    primary = best_f1(labels, scores)
    out = {
        "label": args.label,
        "status": "ok",
        "eval_size": args.eval_size,
        "counts": {"normal": int((labels == 0).sum()), "anomaly": int((labels == 1).sum())},
        "primary_policy": "adpretrain_eval_best_f1",
        "primary": primary,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "result_file": str(args.result_file),
        "note": "AHL original code has no fixed classification threshold; this follows ADPretrain-style best-F1 on the evaluated score block.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", out)
    lines = [
        f"# {args.label} Metrics",
        "",
        "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Threshold |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {primary['precision']:.4f} | {primary['recall']:.4f} | {primary['f1']:.4f} | {primary['accuracy']:.4f} | {out['auc_roc']:.4f} | {out['auc_pr']:.4f} | {primary['threshold']:.6f} |",
        "",
    ]
    (args.output_dir / "metrics.md").write_text("\\n".join(lines), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
