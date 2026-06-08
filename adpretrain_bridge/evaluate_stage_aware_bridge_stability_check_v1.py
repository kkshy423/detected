#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration bootstrap stability check for fixed stage-aware ADP/AHL bridge v1."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


STAGES = ["S5", "S6", "S7", "S8"]
N_BOOTSTRAP = 100
SEED = 20260530

ADP_ROOT = Path("output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/stages")
AHL_ROOT = Path("output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages")
DEFAULT_OUTPUT = Path("summary/20260530_stage_aware_bridge_stability_check_v1")


@dataclass(frozen=True)
class SplitData:
    labels: np.ndarray
    scores: np.ndarray
    ids: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
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
        scale = float((q75 - q25) / 1.349) if (q75 - q25) > 1e-12 else float(np.std(calib_normal_scores))
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
        "P": precision,
        "R": recall,
        "F1": f1,
        "Acc": accuracy,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def recall_floor(stage: str) -> Optional[float]:
    return 0.85 if stage in ["S5", "S6"] else None


def select_threshold(labels: np.ndarray, scores: np.ndarray, floor: Optional[float]) -> Dict[str, float]:
    candidates = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores, float(threshold))
        row["Threshold"] = float(threshold)
        candidates.append(row)
    if floor is None:
        row = max(candidates, key=lambda r: (r["F1"], r["P"], r["R"], r["Threshold"]))
        row["constraint_satisfied"] = True
        return dict(row)
    eligible = [row for row in candidates if row["R"] > floor]
    if eligible:
        row = max(eligible, key=lambda r: (r["F1"], r["P"], r["Threshold"]))
        row["constraint_satisfied"] = True
    else:
        row = max(candidates, key=lambda r: (r["R"], r["F1"], r["P"], r["Threshold"]))
        row["constraint_satisfied"] = False
    return dict(row)


def bridge_alpha(stage: str) -> float:
    return 0.70 if stage == "S8" else 0.35


def method_scores(stage: str) -> Dict[str, Tuple[SplitData, SplitData]]:
    adp = load_scores(ADP_ROOT / stage / "metrics" / "scores.csv")
    ahl = load_scores(AHL_ROOT / stage / "metrics" / "scores.csv")
    adp_calib, ahl_calib = aligned(adp["calib"], ahl["calib"])
    adp_test, ahl_test = aligned(adp["test"], ahl["test"])

    adp_center, adp_scale = robust_norm_params(adp_calib.scores[adp_calib.labels == 0])
    ahl_center, ahl_scale = robust_norm_params(ahl_calib.scores[ahl_calib.labels == 0])
    alpha = bridge_alpha(stage)
    bridge_calib_scores = (
        alpha * robust_norm(adp_calib.scores, adp_center, adp_scale)
        + (1.0 - alpha) * robust_norm(ahl_calib.scores, ahl_center, ahl_scale)
    )
    bridge_test_scores = (
        alpha * robust_norm(adp_test.scores, adp_center, adp_scale)
        + (1.0 - alpha) * robust_norm(ahl_test.scores, ahl_center, ahl_scale)
    )
    bridge_name = f"bridge_alpha_{alpha:.2f}"
    return {
        "ADP-only-DINO": (adp_calib, adp_test),
        "AHL-DINO": (ahl_calib, ahl_test),
        bridge_name: (
            SplitData(labels=adp_calib.labels, scores=bridge_calib_scores, ids=adp_calib.ids),
            SplitData(labels=adp_test.labels, scores=bridge_test_scores, ids=adp_test.ids),
        ),
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_md(summary_rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# 20260530 Stage-Aware Bridge Stability Check V1",
        "",
        "- bootstrap: 100 stratified calibration bootstraps per stage/method",
        "- threshold policy: S5-S6 calibration best-F1 under `Recall>0.85`; S7-S8 calibration best-F1",
        "- bridge alpha: S5-S7 `0.35`, S8 `0.70`",
        "- normalization: bridge scores use full calibration-normal robust median/MAD, then bootstrap only reselects thresholds",
        "- test set is fixed and never used for threshold selection",
        "",
        "| Stage | Method | F1 mean | F1 std | F1 p05 | F1 p50 | F1 p95 | P mean | R mean | Acc mean | FP mean | FN mean | win_rate_vs_ADP | win_rate_vs_AHL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['Stage']} | {row['Method']} | {fmt(row['F1 mean'])} | {fmt(row['F1 std'])} | "
            f"{fmt(row['F1 p05'])} | {fmt(row['F1 p50'])} | {fmt(row['F1 p95'])} | "
            f"{fmt(row['P mean'])} | {fmt(row['R mean'])} | {fmt(row['Acc mean'])} | "
            f"{fmt(row['FP mean'])} | {fmt(row['FN mean'])} | {fmt(row['win_rate_vs_ADP'])} | {fmt(row['win_rate_vs_AHL'])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
    ])
    bridge_by_stage = {row["Stage"]: row for row in summary_rows if str(row["Method"]).startswith("bridge_alpha")}
    for stage in STAGES:
        row = bridge_by_stage[stage]
        win_adp = float(row["win_rate_vs_ADP"])
        r_mean = float(row["R mean"])
        r_std_note = ""
        if stage in ["S5", "S6", "S7"]:
            decision = "bridge mainline confirmed" if win_adp >= 0.80 else "bridge mainline not confirmed"
        else:
            decision = (
                "use bridge-alpha70 in main table"
                if win_adp >= 0.60 and r_mean >= 0.80
                else "fallback to ADP-only in main table; keep bridge-alpha70 as low-FP/high-precision mode"
            )
        lines.append(
            f"- {stage}: {decision}; bridge win_rate_vs_ADP={fmt(win_adp)}, R mean={fmt(r_mean)}{r_std_note}."
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_root / "stability_summary.csv"
    summary_md = args.output_root / "stability_summary.md"
    detail_csv = args.output_root / "bootstrap_detail.csv"
    if not args.force:
        existing = [str(path) for path in [summary_csv, summary_md, detail_csv] if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {existing}")

    rng = np.random.default_rng(args.seed)
    detail_rows: List[Dict[str, object]] = []

    for stage in STAGES:
        score_sets = method_scores(stage)
        ref_calib = next(iter(score_sets.values()))[0]
        normal_idx = np.flatnonzero(ref_calib.labels == 0)
        defect_idx = np.flatnonzero(ref_calib.labels == 1)
        for iteration in range(args.n_bootstrap):
            boot_idx = np.concatenate([
                rng.choice(normal_idx, size=len(normal_idx), replace=True),
                rng.choice(defect_idx, size=len(defect_idx), replace=True),
            ])
            for method, (calib, test) in score_sets.items():
                selected = select_threshold(calib.labels[boot_idx], calib.scores[boot_idx], recall_floor(stage))
                metric = confusion(test.labels, test.scores, float(selected["Threshold"]))
                detail_rows.append({
                    "Iteration": iteration,
                    "Stage": stage,
                    "Method": method,
                    "Threshold": selected["Threshold"],
                    "Calib P": selected["P"],
                    "Calib R": selected["R"],
                    "Calib F1": selected["F1"],
                    "Calib FPR": selected["FPR"],
                    "Constraint satisfied": selected["constraint_satisfied"],
                    "P": metric["P"],
                    "R": metric["R"],
                    "F1": metric["F1"],
                    "Acc": metric["Acc"],
                    "TP": metric["TP"],
                    "FP": metric["FP"],
                    "TN": metric["TN"],
                    "FN": metric["FN"],
                })

    summary_rows: List[Dict[str, object]] = []
    detail_by_stage_method: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in detail_rows:
        detail_by_stage_method.setdefault((row["Stage"], row["Method"]), []).append(row)

    for stage in STAGES:
        methods = [method for (s, method) in detail_by_stage_method if s == stage]
        adp_f1 = np.asarray([float(row["F1"]) for row in detail_by_stage_method[(stage, "ADP-only-DINO")]])
        ahl_f1 = np.asarray([float(row["F1"]) for row in detail_by_stage_method[(stage, "AHL-DINO")]])
        for method in methods:
            rows = detail_by_stage_method[(stage, method)]
            f1_values = np.asarray([float(row["F1"]) for row in rows])
            f1_stats = summarize(f1_values)
            p_values = np.asarray([float(row["P"]) for row in rows])
            r_values = np.asarray([float(row["R"]) for row in rows])
            acc_values = np.asarray([float(row["Acc"]) for row in rows])
            fp_values = np.asarray([float(row["FP"]) for row in rows])
            fn_values = np.asarray([float(row["FN"]) for row in rows])
            summary_rows.append({
                "Stage": stage,
                "Method": method,
                "F1 mean": f1_stats["mean"],
                "F1 std": f1_stats["std"],
                "F1 p05": f1_stats["p05"],
                "F1 p50": f1_stats["p50"],
                "F1 p95": f1_stats["p95"],
                "P mean": float(np.mean(p_values)),
                "R mean": float(np.mean(r_values)),
                "Acc mean": float(np.mean(acc_values)),
                "FP mean": float(np.mean(fp_values)),
                "FN mean": float(np.mean(fn_values)),
                "win_rate_vs_ADP": "" if method == "ADP-only-DINO" else float(np.mean(f1_values > adp_f1)),
                "win_rate_vs_AHL": "" if method == "AHL-DINO" else float(np.mean(f1_values > ahl_f1)),
            })

    method_order = {"ADP-only-DINO": 0, "AHL-DINO": 1}
    summary_rows.sort(key=lambda row: (row["Stage"], method_order.get(str(row["Method"]), 2), str(row["Method"])))

    summary_fields = [
        "Stage", "Method", "F1 mean", "F1 std", "F1 p05", "F1 p50", "F1 p95",
        "P mean", "R mean", "Acc mean", "FP mean", "FN mean", "win_rate_vs_ADP", "win_rate_vs_AHL",
    ]
    detail_fields = [
        "Iteration", "Stage", "Method", "Threshold", "Calib P", "Calib R", "Calib F1", "Calib FPR",
        "Constraint satisfied", "P", "R", "F1", "Acc", "TP", "FP", "TN", "FN",
    ]
    write_csv(summary_csv, summary_fields, summary_rows)
    write_csv(detail_csv, detail_fields, detail_rows)
    summary_md.write_text(build_md(summary_rows), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output_root": str(args.output_root),
        "summary_rows": len(summary_rows),
        "detail_rows": len(detail_rows),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
