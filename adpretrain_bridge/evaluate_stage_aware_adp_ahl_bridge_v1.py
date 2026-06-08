#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate fixed stage-aware ADP/AHL score-level bridge v1 from existing scores."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


STAGES = ["S5", "S6", "S7", "S8"]
ADP_ROOT = Path("output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/stages")
AHL_ROOT = Path("output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages")
BASELINE_SUMMARY = Path("summary/20260530_stage_recall_bestf1_policy/metrics_summary.csv")
DEFAULT_OUTPUT = Path("summary/20260530_stage_aware_adp_ahl_bridge_v1")


@dataclass(frozen=True)
class SplitData:
    labels: np.ndarray
    scores: np.ndarray
    ids: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


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
    raise ValueError(f"Cannot infer split from row: {row}")


def source_id(row: Dict[str, str]) -> str:
    for key in ["source_rel", "path", "stage_rel"]:
        if row.get(key):
            return str(row[key])
    raise ValueError(f"Cannot infer sample id from row: {row}")


def load_scores(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    out: Dict[str, Dict[str, Dict[str, object]]] = {"calib": {}, "test": {}}
    for row in read_csv(path):
        split = normalize_split(row)
        sid = source_id(row)
        if sid in out[split]:
            raise ValueError(f"Duplicate sample id in {path}: {split}/{sid}")
        out[split][sid] = {
            "label": int(float(row["label"])),
            "score": float(row["score"]),
        }
    return out


def pack_split(data: Dict[str, Dict[str, object]]) -> SplitData:
    ids = sorted(data)
    return SplitData(
        labels=np.asarray([int(data[sid]["label"]) for sid in ids], dtype=np.int64),
        scores=np.asarray([float(data[sid]["score"]) for sid in ids], dtype=np.float64),
        ids=ids,
    )


def aligned(adp: Dict[str, Dict[str, object]], ahl: Dict[str, Dict[str, object]]) -> Tuple[SplitData, SplitData]:
    adp_data = pack_split(adp)
    ahl_data = pack_split(ahl)
    if adp_data.ids != ahl_data.ids:
        raise ValueError("ADP and AHL sample ids are not aligned")
    if not np.array_equal(adp_data.labels, ahl_data.labels):
        raise ValueError("ADP and AHL labels are not aligned")
    return adp_data, ahl_data


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


def recall_floor(stage: str) -> float | None:
    idx = int(stage[1:])
    if idx <= 2:
        return 0.90
    if idx <= 6:
        return 0.85
    return None


def select_threshold(labels: np.ndarray, scores: np.ndarray, floor: float | None) -> Dict[str, float]:
    rows = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)
    if floor is None:
        row = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]))
        row["constraint_satisfied"] = True
        return dict(row)
    eligible = [row for row in rows if row["recall"] >= floor]
    if eligible:
        row = max(eligible, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        row["constraint_satisfied"] = True
    else:
        row = max(rows, key=lambda r: (r["recall"], r["f1"], r["precision"], r["threshold"]))
        row["constraint_satisfied"] = False
    return dict(row)


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def load_baselines() -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}
    for row in read_csv(BASELINE_SUMMARY):
        if row["stage"] in STAGES and row["method"] in ["ADP-only-DINO", "AHL-DINO"]:
            out[(row["method"], row["stage"])] = row
    return out


def baseline_to_row(row: Dict[str, str], method_name: str) -> Dict[str, object]:
    return {
        "method": method_name,
        "stage": row["stage"],
        "status": row["status"],
        "policy": row["policy"],
        "alpha": "",
        "threshold": row["threshold"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "accuracy": row["accuracy"],
        "auc_roc": row["auc_roc"],
        "auc_pr": row["auc_pr"],
        "tp": row["tp"],
        "fp": row["fp"],
        "tn": row["tn"],
        "fn": row["fn"],
        "time_total_ms": row.get("time_total_ms", ""),
        "reason": row.get("reason", ""),
    }


def build_md(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# 20260530 Stage-Aware ADP/AHL Bridge V1",
        "",
        "- split: `20260529_qm_xiepai_6_1_fixed_180_70_val49`",
        "- normalization: ADP/AHL scores are robust-normalized by calibration-normal median/MAD per stage",
        "- fixed bridge: `score = alpha * ADP_norm + (1-alpha) * AHL_norm`",
        "- alpha: S5-S7 use `0.35`; S8 reports `0.70` and ADP-only fallback",
        "- threshold policy: S5-S6 use calibration best-F1 under `recall>=0.85`; S7-S8 use calibration best-F1",
        "- source: existing ADP-only-DINO and AHL-DINO score files; no retraining; no ADP/AHL model changes",
        "",
        "| Method | Stage | Status | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN | Total ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['stage']} | {row['status']} | {fmt(row['precision'])} | "
            f"{fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['auc_roc'])} | {fmt(row['auc_pr'])} | {row['tp']} | {row['fp']} | "
            f"{row['tn']} | {row['fn']} | {fmt(row.get('time_total_ms', ''))} |"
        )
    lines.append("")
    lines.append("- CSV: `summary/20260530_stage_aware_adp_ahl_bridge_v1/metrics_summary.csv`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_root / "metrics_summary.csv"
    metrics_md = args.output_root / "metrics_summary.md"
    detail_csv = args.output_root / "bridge_detail.csv"
    if not args.force:
        existing = [str(path) for path in [metrics_csv, metrics_md, detail_csv] if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {existing}")

    baselines = load_baselines()
    rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for stage in STAGES:
        adp = load_scores(ADP_ROOT / stage / "metrics" / "scores.csv")
        ahl = load_scores(AHL_ROOT / stage / "metrics" / "scores.csv")
        adp_calib, ahl_calib = aligned(adp["calib"], ahl["calib"])
        adp_test, ahl_test = aligned(adp["test"], ahl["test"])

        adp_center, adp_scale = robust_norm_params(adp_calib.scores[adp_calib.labels == 0])
        ahl_center, ahl_scale = robust_norm_params(ahl_calib.scores[ahl_calib.labels == 0])
        adp_calib_norm = robust_norm(adp_calib.scores, adp_center, adp_scale)
        ahl_calib_norm = robust_norm(ahl_calib.scores, ahl_center, ahl_scale)
        adp_test_norm = robust_norm(adp_test.scores, adp_center, adp_scale)
        ahl_test_norm = robust_norm(ahl_test.scores, ahl_center, ahl_scale)

        if stage in ["S5", "S6", "S7"]:
            candidates = [("bridge_alpha_0.35", 0.35)]
        else:
            candidates = [("bridge_alpha_0.70", 0.70)]

        for method_name, alpha in candidates:
            calib_score = alpha * adp_calib_norm + (1.0 - alpha) * ahl_calib_norm
            test_score = alpha * adp_test_norm + (1.0 - alpha) * ahl_test_norm
            selected = select_threshold(adp_calib.labels, calib_score, recall_floor(stage))
            metric = confusion(adp_test.labels, test_score, float(selected["threshold"]))
            auc_roc, auc_pr = aucs(adp_test.labels, test_score)
            ahl_time = baselines[("AHL-DINO", stage)].get("time_total_ms", "")
            row = {
                "method": method_name,
                "stage": stage,
                "status": "ok",
                "policy": "stage_fixed_recall_best_f1",
                "alpha": alpha,
                "threshold": selected["threshold"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "accuracy": metric["accuracy"],
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "tp": metric["tp"],
                "fp": metric["fp"],
                "tn": metric["tn"],
                "fn": metric["fn"],
                # Bridge needs the AHL path plus ADP score reuse, so keep AHL-DINO path time as the closest existing estimate.
                "time_total_ms": ahl_time,
                "reason": "",
            }
            rows.append(row)
            detail_rows.append({
                **row,
                "calib_precision": selected["precision"],
                "calib_recall": selected["recall"],
                "calib_f1": selected["f1"],
                "calib_accuracy": selected["accuracy"],
                "calib_fpr": selected["fpr"],
                "constraint_satisfied": selected["constraint_satisfied"],
                "adp_center": adp_center,
                "adp_scale": adp_scale,
                "ahl_center": ahl_center,
                "ahl_scale": ahl_scale,
            })

        rows.append(baseline_to_row(baselines[("ADP-only-DINO", stage)], "ADP-only-DINO"))
        rows.append(baseline_to_row(baselines[("AHL-DINO", stage)], "AHL-DINO"))
        if stage == "S8":
            rows.append(baseline_to_row(baselines[("ADP-only-DINO", stage)], "ADP-only fallback"))

    fieldnames = [
        "method", "stage", "status", "policy", "alpha", "threshold", "precision", "recall", "f1",
        "accuracy", "auc_roc", "auc_pr", "tp", "fp", "tn", "fn", "time_total_ms", "reason",
    ]
    detail_fields = fieldnames + [
        "calib_precision", "calib_recall", "calib_f1", "calib_accuracy", "calib_fpr",
        "constraint_satisfied", "adp_center", "adp_scale", "ahl_center", "ahl_scale",
    ]
    write_csv(metrics_csv, fieldnames, rows)
    write_csv(detail_csv, detail_fields, detail_rows)
    metrics_md.write_text(build_md(rows), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output_root": str(args.output_root),
        "metrics_rows": len(rows),
        "detail_rows": len(detail_rows),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
