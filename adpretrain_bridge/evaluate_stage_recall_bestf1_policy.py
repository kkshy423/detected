#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a staged recall-constrained best-F1 threshold policy from existing scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_OUTPUT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/"
    "summary/20260530_stage_recall_bestf1_policy"
)

METHODS = {
    "AHL-DINO": Path("output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages"),
    "ADP-only-DINO": Path(
        "output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/stages"
    ),
    "YOLO26s-eval": Path("output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only/stages"),
}

STAGES = [f"S{i}" for i in range(9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="overwrite output files")
    return parser.parse_args()


def normalize_split(row: Dict[str, str]) -> str:
    split = str(row.get("split", "")).strip().lower()
    if split == "val":
        return "calib"
    if split == "test":
        return "test"
    role = str(row.get("role", "")).strip().lower()
    if role.startswith("calib_"):
        return "calib"
    if role.startswith("test_"):
        return "test"
    raise ValueError(f"Could not identify calib/test split for row: {row}")


def image_id(row: Dict[str, str]) -> str:
    for key in ["source_rel", "path", "stage_rel"]:
        if row.get(key):
            return str(row[key])
    return ""


def load_scores(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        rows.append({
            "split": normalize_split(row),
            "label": int(float(row["label"])),
            "score": float(row["score"]),
            "image_id": image_id(row),
        })
    return rows


def arrays(rows: Sequence[Dict], split: str) -> Tuple[np.ndarray, np.ndarray]:
    part = [row for row in rows if row["split"] == split]
    return (
        np.asarray([row["label"] for row in part], dtype=np.int64),
        np.asarray([row["score"] for row in part], dtype=np.float64),
    )


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(preds & positives))
    fp = int(np.sum(preds & negatives))
    tn = int(np.sum((~preds) & negatives))
    fn = int(np.sum((~preds) & positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def aucs(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def stage_rule(stage: str) -> Tuple[str, float | None]:
    idx = int(stage[1:])
    if idx <= 2:
        return "calib_best_f1_with_recall_ge_0.90", 0.90
    if idx <= 6:
        return "calib_best_f1_with_recall_ge_0.85", 0.85
    return "calib_best_f1", None


def choose_threshold(labels: np.ndarray, scores: np.ndarray, recall_floor: float | None) -> Dict:
    candidates = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores, float(threshold))
        row["threshold"] = float(threshold)
        candidates.append(row)
    if recall_floor is not None:
        constrained = [row for row in candidates if row["recall"] >= recall_floor]
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


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def build_markdown(rows: Sequence[Dict], missing_rows: Sequence[Dict]) -> str:
    lines = [
        "# 20260530 Stage Recall-Constrained Best-F1 Policy",
        "",
        "Eval-only summary from existing score files. Thresholds are selected from calibration/val scores only; test labels are used only for reporting.",
        "",
        "Policy:",
        "",
        "- S0-S2: choose calibration best-F1 among thresholds with calibration recall >= 0.90.",
        "- S3-S6: choose calibration best-F1 among thresholds with calibration recall >= 0.85.",
        "- S7-S8: choose unconstrained calibration best-F1.",
        "",
    ]
    for method in METHODS:
        lines.extend([
            f"## {method}",
            "",
            "| Stage | Policy | Threshold | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN | AUC-ROC | AUC-PR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in [r for r in rows if r["method"] == method]:
            lines.append(
                f"| {row['stage']} | {row['policy']} | {fmt(row['threshold'])} | "
                f"{fmt(row['calib_recall'])} | {fmt(row['calib_f1'])} | {fmt(row['calib_fpr'])} | "
                f"{fmt(row['test_precision'])} | {fmt(row['test_recall'])} | {fmt(row['test_f1'])} | "
                f"{row['test_fp']} | {row['test_fn']} | {fmt(row['auc_roc'])} | {fmt(row['auc_pr'])} |"
            )
        lines.append("")
    lines.extend([
        "## Short Diagnosis",
        "",
        "- This table tests the requested threshold policy only; it does not compare against test-best thresholds.",
        "- If `constraint_satisfied` is false in the CSV, the requested calibration recall floor was infeasible for that score set.",
        "- S7-S8 intentionally allow lower calibration recall if that is where calibration F1 is maximized.",
        "",
    ])
    if missing_rows:
        lines.extend([
            "## Unavailable Inputs",
            "",
            "| Method | Stage | Missing file |",
            "|---|---|---|",
        ])
        for row in missing_rows:
            lines.append(f"| {row['method']} | {row['stage']} | `{row['missing_file']}` |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_root / "stage_recall_bestf1_policy_metrics.csv"
    out_md = args.output_root / "stage_recall_bestf1_policy_metrics.md"
    out_missing = args.output_root / "unavailable_inputs.csv"
    if not args.force:
        existing = [str(path) for path in [out_csv, out_md, out_missing] if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {existing}")

    rows = []
    missing_rows = []
    for method, stages_root in METHODS.items():
        for stage in STAGES:
            scores_path = stages_root / stage / "metrics" / "scores.csv"
            if not scores_path.exists():
                missing_rows.append({
                    "method": method,
                    "stage": stage,
                    "missing_file": str(scores_path),
                })
                continue
            score_rows = load_scores(scores_path)
            calib_labels, calib_scores = arrays(score_rows, "calib")
            test_labels, test_scores = arrays(score_rows, "test")
            policy, recall_floor = stage_rule(stage)
            selected = choose_threshold(calib_labels, calib_scores, recall_floor)
            test_metric = confusion(test_labels, test_scores, selected["threshold"])
            auc_roc, auc_pr = aucs(test_labels, test_scores)
            rows.append({
                "method": method,
                "stage": stage,
                "policy": policy,
                "recall_floor": recall_floor,
                "constraint_satisfied": selected["constraint_satisfied"],
                "threshold": selected["threshold"],
                "calib_precision": selected["precision"],
                "calib_recall": selected["recall"],
                "calib_f1": selected["f1"],
                "calib_accuracy": selected["accuracy"],
                "calib_fpr": selected["fpr"],
                "calib_fp": selected["fp"],
                "calib_fn": selected["fn"],
                "test_precision": test_metric["precision"],
                "test_recall": test_metric["recall"],
                "test_f1": test_metric["f1"],
                "test_accuracy": test_metric["accuracy"],
                "test_fpr": test_metric["fpr"],
                "test_fp": test_metric["fp"],
                "test_fn": test_metric["fn"],
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
            })

    fieldnames = [
        "method", "stage", "policy", "recall_floor", "constraint_satisfied", "threshold",
        "calib_precision", "calib_recall", "calib_f1", "calib_accuracy", "calib_fpr", "calib_fp", "calib_fn",
        "test_precision", "test_recall", "test_f1", "test_accuracy", "test_fpr", "test_fp", "test_fn",
        "auc_roc", "auc_pr",
    ]
    write_csv(out_csv, fieldnames, rows)
    write_csv(out_missing, ["method", "stage", "missing_file"], missing_rows)
    out_md.write_text(build_markdown(rows, missing_rows), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output_root": str(args.output_root),
        "rows": len(rows),
        "missing_rows": len(missing_rows),
        "csv": str(out_csv),
        "markdown": str(out_md),
        "missing_csv": str(out_missing),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
