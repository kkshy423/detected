#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval-only diagnostics for late-stage AHL-DINO calibration thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from threshold_policies import (
    best_f1_threshold,
    best_f1_with_fpr_cap,
    fpr_from_metric,
    metric_at,
    normal_percentile_threshold,
)


DEFAULT_OUTPUT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/"
    "summary/20260530_dino_late_threshold_diagnostics"
)

INPUTS = {
    ("AHL-DINO", "S7"): Path(
        "output/20260529_ahl_dino_large_180_70_val49_stage_v1/"
        "stages/S7/metrics"
    ),
    ("AHL-DINO", "S8"): Path(
        "output/20260529_ahl_dino_large_180_70_val49_stage_v1/"
        "stages/S8/metrics"
    ),
    ("ADP-only-DINO", "S7"): Path(
        "output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/"
        "stages/S7/metrics"
    ),
    ("ADP-only-DINO", "S8"): Path(
        "output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/"
        "stages/S8/metrics"
    ),
    ("YOLO26s-eval", "S7"): Path(
        "output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only/"
        "stages/S7/metrics"
    ),
    ("YOLO26s-eval", "S8"): Path(
        "output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only/"
        "stages/S8/metrics"
    ),
}

PROBE_POLICIES = [
    "current",
    "val_best_f1",
    "val_best_f1_fpr10",
    "recall90_fpr20",
    "recall85_fpr10",
    "recall80_fpr10",
    "normal_p95",
    "normal_p90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="overwrite the four diagnostic output files")
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


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
    raise ValueError(f"Could not identify image id for row: {row}")


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


def fpr(row: Dict) -> float:
    return float(fpr_from_metric(row))


def with_calib_fpr(row: Dict) -> Dict:
    out = dict(row)
    out["calibration_fpr"] = fpr(out)
    return out


def policy_thresholds(calib_labels: np.ndarray, calib_scores: np.ndarray, current_threshold: float) -> Dict[str, Dict]:
    return {
        "current": with_calib_fpr(metric_at(calib_labels, calib_scores, current_threshold)),
        "val_best_f1": with_calib_fpr(best_f1_threshold(calib_labels, calib_scores)),
        "val_best_f1_fpr10": best_f1_with_fpr_cap(calib_labels, calib_scores, 0.10),
        "recall90_fpr20": capped_recall_threshold(calib_labels, calib_scores, 0.90, 0.20),
        "recall85_fpr10": capped_recall_threshold(calib_labels, calib_scores, 0.85, 0.10),
        "recall80_fpr10": capped_recall_threshold(calib_labels, calib_scores, 0.80, 0.10),
        "normal_p95": with_calib_fpr(normal_percentile_threshold(calib_labels, calib_scores, 95.0)),
        "normal_p90": with_calib_fpr(normal_percentile_threshold(calib_labels, calib_scores, 90.0)),
    }


def capped_recall_threshold(labels: np.ndarray, scores: np.ndarray, recall_floor: float, fpr_cap: float) -> Dict:
    rows = []
    for threshold in np.unique(scores):
        row = with_calib_fpr(metric_at(labels, scores, float(threshold)))
        rows.append(row)
    capped = [row for row in rows if row["calibration_fpr"] <= fpr_cap]
    if not capped:
        row = with_calib_fpr(best_f1_threshold(labels, scores))
        row["selection_rule"] = "fallback_validation_best_f1_no_fpr_cap_candidate"
    else:
        strict = [row for row in capped if row["recall"] >= recall_floor]
        if strict:
            row = max(strict, key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]))
            row["selection_rule"] = "max_f1_with_recall_floor_and_fpr_cap"
        else:
            row = max(capped, key=lambda r: (r["recall"], r["f1"], r["precision"], r["threshold"]))
            row["selection_rule"] = "max_recall_under_fpr_cap_recall_floor_unmet"
    row = dict(row)
    row["recall_floor"] = float(recall_floor)
    row["fpr_cap"] = float(fpr_cap)
    return row


def quantile_summary(scores: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(scores)),
        "p05": float(np.percentile(scores, 5)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
    }


def percentile_rank(scores: np.ndarray, score: float) -> float:
    less = float(np.sum(scores < score))
    equal = float(np.sum(scores == score))
    return 100.0 * (less + 0.5 * equal) / float(len(scores))


def fmt(value) -> str:
    return "" if value is None else f"{float(value):.4f}"


def probe_lookup(rows: Sequence[Dict], method: str, stage: str, policy: str) -> Dict:
    return next(row for row in rows if row["method"] == method and row["stage"] == stage and row["policy"] == policy)


def dist_lookup(rows: Sequence[Dict], method: str, stage: str, split: str, label: int) -> Dict:
    return next(
        row for row in rows
        if row["method"] == method and row["stage"] == stage and row["split"] == split and row["label"] == label
    )


def build_diagnosis(distribution_rows: Sequence[Dict], probe_rows: Sequence[Dict]) -> str:
    lines = [
        "# 20260530 DINO Late Threshold Diagnostics",
        "",
        "This report is eval-only. Every probed threshold is selected from calibration scores only. "
        "Held-out test labels are used only to report resulting metrics.",
        "",
        "## A. Is AHL-DINO late recall mainly a threshold issue or a ranking issue?",
        "",
        "| Stage | Test AUC-ROC | Test AUC-PR | Current threshold | Current R | Current FP | Current FN | "
        "recall90_fpr20 threshold | recall90_fpr20 R | recall90_fpr20 FP | recall90_fpr20 FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ["S7", "S8"]:
        current = probe_lookup(probe_rows, "AHL-DINO", stage, "current")
        recall90 = probe_lookup(probe_rows, "AHL-DINO", stage, "recall90_fpr20")
        lines.append(
            f"| {stage} | {fmt(current['auc_roc'])} | {fmt(current['auc_pr'])} | {fmt(current['threshold'])} | "
            f"{fmt(current['test_r'])} | {current['test_fp']} | {current['test_fn']} | {fmt(recall90['threshold'])} | "
            f"{fmt(recall90['test_r'])} | {recall90['test_fp']} | {recall90['test_fn']} |"
        )
    lines.extend([
        "",
        "AHL-DINO retains strong test ranking metrics in both stages. Under strict FPR caps, however, recall cannot be "
        "fully restored without accepting more false positives. The low late-stage recall is therefore a threshold "
        "operating-point issue under the current low-FPR target, not evidence that the ranking has collapsed.",
        "",
        "## B. Is there an obvious calib-defect versus test-defect score shift?",
        "",
        "| Stage | Calib defect p25 | Calib defect p50 | Calib defect p75 | Test defect p25 | Test defect p50 | "
        "Test defect p75 | Test median - calib median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    median_deltas = {}
    for stage in ["S7", "S8"]:
        calib = dist_lookup(distribution_rows, "AHL-DINO", stage, "calib", 1)
        test = dist_lookup(distribution_rows, "AHL-DINO", stage, "test", 1)
        delta = float(test["p50"]) - float(calib["p50"])
        median_deltas[stage] = delta
        lines.append(
            f"| {stage} | {fmt(calib['p25'])} | {fmt(calib['p50'])} | {fmt(calib['p75'])} | "
            f"{fmt(test['p25'])} | {fmt(test['p50'])} | {fmt(test['p75'])} | {fmt(delta)} |"
        )
    lines.extend([
        "",
        "The defect-score distributions should be read together with the table above. A negative median delta means test "
        "defects score lower than calibration defects; a positive value means the opposite. This quantifies distribution "
        "shift without using test labels to select a threshold.",
        "",
        "## C. Which calibration-derived threshold is more suitable for late AHL-DINO?",
        "",
        "| Stage | Policy | Threshold | Calib P | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stage in ["S7", "S8"]:
        for policy in PROBE_POLICIES:
            row = probe_lookup(probe_rows, "AHL-DINO", stage, policy)
            lines.append(
                f"| {stage} | {policy} | {fmt(row['threshold'])} | {fmt(row['calib_p'])} | {fmt(row['calib_r'])} | "
                f"{fmt(row['calib_f1'])} | {fmt(row['calib_fpr'])} | {fmt(row['test_p'])} | {fmt(row['test_r'])} | "
                f"{fmt(row['test_f1'])} | {row['test_fp']} | {row['test_fn']} |"
            )
    lines.extend([
        "",
        "No single threshold is universally preferable without a production cost target. For a late-stage balanced "
        "operating point under a strict calibration false-positive cap, keep `val_best_f1_fpr10` / `current`. For a "
        "higher-recall production mode that accepts more false alarms, compare `recall90_fpr20` directly against the "
        "current policy. `normal_p90` and `normal_p95` are included as normal-only calibration references.",
        "",
        "## Notes",
        "",
        "- `current` is read from each method's existing `metrics.json` primary threshold.",
        "- `val_best_f1_fpr10` maximizes calibration F1 among thresholds with calibration FPR <= 0.10.",
        "- `recallXX_fprYY` uses calibration labels only and keeps the stated FPR cap. If the recall floor is infeasible, "
        "it reports the highest calibration recall reachable under the cap.",
        "- `fn_score_margin.csv` uses the current threshold. `rank_percentile` is the FN score percentile within all test scores "
        "for the same method and stage.",
        "- No test-label-selected threshold is exported as a production candidate.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    distribution_rows: List[Dict] = []
    probe_rows: List[Dict] = []
    fn_rows: List[Dict] = []

    for (method, stage), metrics_dir in INPUTS.items():
        metrics = read_json(metrics_dir / "metrics.json")
        scores = load_scores(metrics_dir / "scores.csv")
        calib_labels, calib_scores = arrays(scores, "calib")
        test_labels, test_scores = arrays(scores, "test")
        current_threshold = float((metrics.get("primary") or {})["threshold"])
        thresholds = policy_thresholds(calib_labels, calib_scores, current_threshold)

        for split, labels, values in [
            ("calib", calib_labels, calib_scores),
            ("test", test_labels, test_scores),
        ]:
            for label in [0, 1]:
                part = values[labels == label]
                row = {
                    "method": method,
                    "stage": stage,
                    "split": split,
                    "label": label,
                    "threshold": current_threshold,
                }
                row.update(quantile_summary(part))
                distribution_rows.append(row)

        auc_roc = float(roc_auc_score(test_labels, test_scores))
        auc_pr = float(average_precision_score(test_labels, test_scores))
        for policy in PROBE_POLICIES:
            threshold = float(thresholds[policy]["threshold"])
            calib = metric_at(calib_labels, calib_scores, threshold)
            test = metric_at(test_labels, test_scores, threshold)
            probe_rows.append({
                "method": method,
                "stage": stage,
                "policy": policy,
                "threshold": threshold,
                "calib_p": calib["precision"],
                "calib_r": calib["recall"],
                "calib_f1": calib["f1"],
                "calib_fpr": fpr(calib),
                "test_p": test["precision"],
                "test_r": test["recall"],
                "test_f1": test["f1"],
                "test_fp": test["fp"],
                "test_fn": test["fn"],
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
            })

        test_part = [row for row in scores if row["split"] == "test"]
        for row in test_part:
            if row["label"] == 1 and row["score"] < current_threshold:
                fn_rows.append({
                    "method": method,
                    "stage": stage,
                    "image_id": row["image_id"],
                    "label": row["label"],
                    "score": row["score"],
                    "threshold": current_threshold,
                    "margin": current_threshold - row["score"],
                    "rank_percentile": percentile_rank(test_scores, row["score"]),
                })

    fn_rows.sort(key=lambda row: (row["margin"], row["method"], row["stage"], row["image_id"]))
    write_csv(
        args.output_root / "score_distribution_summary.csv",
        ["method", "stage", "split", "label", "n", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max", "threshold"],
        distribution_rows,
    )
    write_csv(
        args.output_root / "threshold_policy_probe.csv",
        [
            "method", "stage", "policy", "threshold",
            "calib_p", "calib_r", "calib_f1", "calib_fpr",
            "test_p", "test_r", "test_f1", "test_fp", "test_fn",
            "auc_roc", "auc_pr",
        ],
        probe_rows,
    )
    write_csv(
        args.output_root / "fn_score_margin.csv",
        ["method", "stage", "image_id", "label", "score", "threshold", "margin", "rank_percentile"],
        fn_rows,
    )
    (args.output_root / "diagnosis.md").write_text(build_diagnosis(distribution_rows, probe_rows), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output_root": str(args.output_root),
        "distribution_rows": len(distribution_rows),
        "probe_rows": len(probe_rows),
        "fn_rows": len(fn_rows),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
