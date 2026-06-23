#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize CPU/PIL vs GPU-uint8 full-flow ADP, AHL, and bridge results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


STAGES = [f"S{i}" for i in range(9)]
METHODS = ["ADP-only-DINO", "AHL-DINO", "ADP-AHL-bridge-v2"]


def bridge_alpha(stage: str) -> Optional[float]:
    if stage in {"S1", "S2"}:
        return 0.70
    if stage in {"S3", "S4"}:
        return 0.60
    if stage in {"S5", "S6", "S7"}:
        return 0.35
    if stage == "S8":
        return 0.70
    return None


def stage_recall_floor(stage: str) -> Optional[float]:
    idx = int(stage[1:])
    if idx <= 2:
        return 0.90
    if idx <= 6:
        return 0.85
    return None


def robust_norm_params(calib_normal: np.ndarray) -> Tuple[float, float]:
    center = float(np.median(calib_normal))
    mad = float(np.median(np.abs(calib_normal - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q75, q25 = np.percentile(calib_normal, [75, 25])
        iqr = float((q75 - q25) / 1.349)
        scale = iqr if iqr > 1e-12 else float(np.std(calib_normal))
    if scale <= 1e-12:
        scale = 1.0
    return center, scale


def robust_norm(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (x - center) / scale


def source_key(row: Dict[str, str]) -> str:
    role = row.get("role", "").strip()
    rel = row.get("source_rel") or row.get("stage_rel") or row.get("path") or ""
    return f"{role}/{Path(rel).name}"


def split_of(row: Dict[str, str]) -> Optional[str]:
    explicit = row.get("split", "").strip()
    if explicit in {"calib", "test"}:
        return explicit
    role = row.get("role", "").strip()
    if role.startswith("calib"):
        return "calib"
    if role.startswith("test"):
        return "test"
    return None


def load_scores(path: Path) -> Dict[str, Dict[str, Tuple[int, float]]]:
    out: Dict[str, Dict[str, Tuple[int, float]]] = {"calib": {}, "test": {}}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = split_of(row)
            if split is None:
                continue
            key = source_key(row)
            out[split][key] = (int(float(row["label"])), float(row["score"]))
    return out


def aligned(adp: Dict, ahl: Dict, split: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(adp[split]) & set(ahl[split]))
    if not keys:
        raise RuntimeError(f"No aligned rows for split={split}")
    labels = np.array([adp[split][k][0] for k in keys], dtype=int)
    ahl_labels = np.array([ahl[split][k][0] for k in keys], dtype=int)
    if not np.array_equal(labels, ahl_labels):
        bad = int(np.sum(labels != ahl_labels))
        raise RuntimeError(f"ADP/AHL label mismatch for split={split}: {bad}")
    adp_scores = np.array([adp[split][k][1] for k in keys], dtype=float)
    ahl_scores = np.array([ahl[split][k][1] for k in keys], dtype=float)
    return keys, labels, adp_scores, ahl_scores


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray, floor: Optional[float]) -> Dict[str, float]:
    candidates = [confusion(labels, scores, float(t)) for t in np.unique(scores)]
    if floor is not None:
        ok = [c for c in candidates if c["recall"] >= floor]
        if ok:
            return max(ok, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        return max(candidates, key=lambda r: (r["recall"], r["f1"], r["precision"], r["threshold"]))
    return max(candidates, key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]))


def add_auc(labels: np.ndarray, scores: np.ndarray, row: Dict[str, float]) -> Dict[str, float]:
    row = dict(row)
    if len(np.unique(labels)) > 1:
        row["auc_roc"] = float(roc_auc_score(labels, scores))
        row["auc_pr"] = float(average_precision_score(labels, scores))
    else:
        row["auc_roc"] = math.nan
        row["auc_pr"] = math.nan
    return row


def eval_scores(
    stage: str,
    calib_labels: np.ndarray,
    calib_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    floor = stage_recall_floor(stage)
    if fixed_threshold is None:
        threshold = float(choose_threshold(calib_labels, calib_scores, floor)["threshold"])
    else:
        threshold = float(fixed_threshold)
    return add_auc(test_labels, test_scores, confusion(test_labels, test_scores, threshold))


def build_variant(stage: str, adp_path: Path, ahl_path: Path) -> Dict[str, Dict]:
    adp = load_scores(adp_path)
    ahl = load_scores(ahl_path)
    calib_keys, calib_labels, calib_adp, calib_ahl = aligned(adp, ahl, "calib")
    test_keys, test_labels, test_adp, test_ahl = aligned(adp, ahl, "test")
    out = {
        "ADP-only-DINO": {
            "calib_keys": calib_keys,
            "test_keys": test_keys,
            "calib_labels": calib_labels,
            "test_labels": test_labels,
            "calib_scores": calib_adp,
            "test_scores": test_adp,
            "alpha": "",
        },
        "AHL-DINO": {
            "calib_keys": calib_keys,
            "test_keys": test_keys,
            "calib_labels": calib_labels,
            "test_labels": test_labels,
            "calib_scores": calib_ahl,
            "test_scores": test_ahl,
            "alpha": "",
        },
    }
    alpha = bridge_alpha(stage)
    if alpha is not None:
        adp_c, adp_s = robust_norm_params(calib_adp[calib_labels == 0])
        ahl_c, ahl_s = robust_norm_params(calib_ahl[calib_labels == 0])
        out["ADP-AHL-bridge-v2"] = {
            "calib_keys": calib_keys,
            "test_keys": test_keys,
            "calib_labels": calib_labels,
            "test_labels": test_labels,
            "calib_scores": alpha * robust_norm(calib_adp, adp_c, adp_s)
            + (1.0 - alpha) * robust_norm(calib_ahl, ahl_c, ahl_s),
            "test_scores": alpha * robust_norm(test_adp, adp_c, adp_s)
            + (1.0 - alpha) * robust_norm(test_ahl, ahl_c, ahl_s),
            "alpha": alpha,
        }
    return out


def load_v2(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("Method", "").strip()
            if method == "ADP-AHL bridge v2":
                method = "ADP-AHL-bridge-v2"
            if method not in METHODS:
                continue
            stage = row.get("Stage", "").strip()
            out[(method, stage)] = row
    return out


def fnum(v, digits: int = 4) -> str:
    try:
        f = float(v)
    except Exception:
        return ""
    if math.isnan(f):
        return ""
    return f"{f:.{digits}f}"


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def pred_change(cpu: Dict, gpu: Dict, cpu_thr: float, gpu_thr: float) -> Dict[str, float]:
    gpu_by_key = {k: gpu["test_scores"][i] for i, k in enumerate(gpu["test_keys"])}
    cpu_pairs = [(k, cpu["test_scores"][i]) for i, k in enumerate(cpu["test_keys"]) if k in gpu_by_key]
    if not cpu_pairs:
        return {"aligned_test": 0, "score_abs_median": math.nan, "score_abs_p95": math.nan, "pred_changed_self": math.nan, "pred_changed_fixed_cpu_thr": math.nan}
    cpu_scores = np.array([s for _, s in cpu_pairs], dtype=float)
    gpu_scores = np.array([gpu_by_key[k] for k, _ in cpu_pairs], dtype=float)
    diffs = np.abs(gpu_scores - cpu_scores)
    cpu_pred_self = (cpu_scores >= cpu_thr).astype(int)
    gpu_pred_self = (gpu_scores >= gpu_thr).astype(int)
    gpu_pred_fixed = (gpu_scores >= cpu_thr).astype(int)
    return {
        "aligned_test": int(len(cpu_pairs)),
        "score_abs_median": float(np.median(diffs)),
        "score_abs_p95": float(np.percentile(diffs, 95)),
        "pred_changed_self": int(np.sum(cpu_pred_self != gpu_pred_self)),
        "pred_changed_fixed_cpu_thr": int(np.sum(cpu_pred_self != gpu_pred_fixed)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-adp-output", type=Path, required=True)
    p.add_argument("--cpu-ahl-output", type=Path, required=True)
    p.add_argument("--gpu-adp-output", type=Path, required=True)
    p.add_argument("--gpu-ahl-output", type=Path, required=True)
    p.add_argument("--v2-table", type=Path, default=Path("summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    variants = {
        "cpu_pil_rerun": (args.cpu_adp_output, args.cpu_ahl_output),
        "gpu_uint8_rerun": (args.gpu_adp_output, args.gpu_ahl_output),
    }
    v2 = load_v2(args.v2_table)

    score_bank: Dict[str, Dict[str, Dict[str, Dict]]] = {stage: {} for stage in STAGES}
    rows: List[Dict] = []
    fixed_rows: List[Dict] = []

    for stage in STAGES:
        for backend, (adp_root, ahl_root) in variants.items():
            adp_path = adp_root / "stages" / stage / "metrics" / "scores.csv"
            ahl_path = ahl_root / "stages" / stage / "metrics" / "scores.csv"
            score_bank[stage][backend] = build_variant(stage, adp_path, ahl_path)

        for method in METHODS:
            if method not in score_bank[stage]["cpu_pil_rerun"]:
                continue
            cpu = score_bank[stage]["cpu_pil_rerun"][method]
            cpu_m = eval_scores(stage, cpu["calib_labels"], cpu["calib_scores"], cpu["test_labels"], cpu["test_scores"])
            cpu_thr = float(cpu_m["threshold"])
            for backend in variants:
                if method not in score_bank[stage][backend]:
                    continue
                item = score_bank[stage][backend][method]
                self_m = eval_scores(stage, item["calib_labels"], item["calib_scores"], item["test_labels"], item["test_scores"])
                row = {
                    "backend": backend,
                    "threshold_mode": "self_calibrated",
                    "method": method,
                    "stage": stage,
                    "alpha": item.get("alpha", ""),
                    **self_m,
                }
                hist = v2.get((method, stage))
                if hist:
                    row["v2_f1"] = hist.get("F1", "")
                    row["v2_recall"] = hist.get("R", "")
                    row["v2_auc_pr"] = hist.get("AUC-PR", "")
                    row["delta_f1_vs_v2"] = float(row["f1"]) - float(hist["F1"]) if hist.get("F1") else ""
                    row["delta_recall_vs_v2"] = float(row["recall"]) - float(hist["R"]) if hist.get("R") else ""
                    row["delta_auc_pr_vs_v2"] = float(row["auc_pr"]) - float(hist["AUC-PR"]) if hist.get("AUC-PR") else ""
                rows.append(row)

                fixed_m = eval_scores(stage, item["calib_labels"], item["calib_scores"], item["test_labels"], item["test_scores"], fixed_threshold=cpu_thr)
                fixed_rows.append({
                    "backend": backend,
                    "threshold_mode": "fixed_cpu_threshold",
                    "method": method,
                    "stage": stage,
                    "alpha": item.get("alpha", ""),
                    **fixed_m,
                })

    gap_rows: List[Dict] = []
    by_key = {(r["method"], r["stage"], r["backend"]): r for r in rows}
    for stage in STAGES:
        for method in METHODS:
            key_cpu = (method, stage, "cpu_pil_rerun")
            key_gpu = (method, stage, "gpu_uint8_rerun")
            if key_cpu not in by_key or key_gpu not in by_key:
                continue
            cpu_row = by_key[key_cpu]
            gpu_row = by_key[key_gpu]
            pc = pred_change(
                score_bank[stage]["cpu_pil_rerun"][method],
                score_bank[stage]["gpu_uint8_rerun"][method],
                float(cpu_row["threshold"]),
                float(gpu_row["threshold"]),
            )
            gap_rows.append({
                "method": method,
                "stage": stage,
                "cpu_f1": cpu_row["f1"],
                "gpu_f1": gpu_row["f1"],
                "delta_f1_gpu_minus_cpu": float(gpu_row["f1"]) - float(cpu_row["f1"]),
                "cpu_recall": cpu_row["recall"],
                "gpu_recall": gpu_row["recall"],
                "delta_recall_gpu_minus_cpu": float(gpu_row["recall"]) - float(cpu_row["recall"]),
                "cpu_auc_pr": cpu_row["auc_pr"],
                "gpu_auc_pr": gpu_row["auc_pr"],
                "delta_auc_pr_gpu_minus_cpu": float(gpu_row["auc_pr"]) - float(cpu_row["auc_pr"]),
                "cpu_threshold": cpu_row["threshold"],
                "gpu_threshold": gpu_row["threshold"],
                **pc,
                "cpu_delta_f1_vs_v2": cpu_row.get("delta_f1_vs_v2", ""),
                "gpu_delta_f1_vs_v2": gpu_row.get("delta_f1_vs_v2", ""),
            })

    metric_fields = [
        "backend", "threshold_mode", "method", "stage", "alpha", "threshold",
        "precision", "recall", "f1", "accuracy", "auc_roc", "auc_pr",
        "tp", "fp", "tn", "fn", "v2_f1", "v2_recall", "v2_auc_pr",
        "delta_f1_vs_v2", "delta_recall_vs_v2", "delta_auc_pr_vs_v2",
    ]
    gap_fields = [
        "method", "stage", "cpu_f1", "gpu_f1", "delta_f1_gpu_minus_cpu",
        "cpu_recall", "gpu_recall", "delta_recall_gpu_minus_cpu",
        "cpu_auc_pr", "gpu_auc_pr", "delta_auc_pr_gpu_minus_cpu",
        "cpu_threshold", "gpu_threshold", "aligned_test", "score_abs_median",
        "score_abs_p95", "pred_changed_self", "pred_changed_fixed_cpu_thr",
        "cpu_delta_f1_vs_v2", "gpu_delta_f1_vs_v2",
    ]
    write_csv(args.out_dir / "cpu_gpu_full_flow_master_table.csv", rows, metric_fields)
    write_csv(args.out_dir / "cpu_gpu_full_flow_fixed_cpu_threshold_table.csv", fixed_rows, metric_fields[:16])
    write_csv(args.out_dir / "cpu_gpu_full_flow_gap_summary.csv", gap_rows, gap_fields)

    lines = [
        "# CPU/PIL vs GPU uint8 full-flow rerun v1",
        "",
        "Thresholds are selected on calib only; test is only reported. The primary table below uses self-calibrated thresholds per backend.",
        "",
        "## Relationship",
        "",
        "- ADP-only-DINO: training-free DINO projected residual matching, ADP `num_ref=8`.",
        "- AHL-DINO: trainable few-shot detector on ADP/DINO feature cache, AHL `n_ref=5`, epochs=30, steps=20, batch=48, seed=20260517.",
        "- ADP-AHL-bridge-v2: late score fusion after robust normalization on calib normal scores; alpha schedule is fixed and no test oracle is used.",
        "",
        "## CPU/GPU Gap Summary",
        "",
        "| method | stage | CPU F1 | GPU F1 | dF1 | CPU R | GPU R | dR | CPU AUPR | GPU AUPR | dAUPR | pred changed self | p95 score diff | CPU dF1 vs v2 | GPU dF1 vs v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in gap_rows:
        lines.append(
            f"| {r['method']} | {r['stage']} | {fnum(r['cpu_f1'])} | {fnum(r['gpu_f1'])} | {fnum(r['delta_f1_gpu_minus_cpu'])} | "
            f"{fnum(r['cpu_recall'])} | {fnum(r['gpu_recall'])} | {fnum(r['delta_recall_gpu_minus_cpu'])} | "
            f"{fnum(r['cpu_auc_pr'])} | {fnum(r['gpu_auc_pr'])} | {fnum(r['delta_auc_pr_gpu_minus_cpu'])} | "
            f"{r['pred_changed_self']} | {fnum(r['score_abs_p95'])} | {fnum(r['cpu_delta_f1_vs_v2'])} | {fnum(r['gpu_delta_f1_vs_v2'])} |"
        )

    bridge_gaps = [r for r in gap_rows if r["method"] == "ADP-AHL-bridge-v2"]
    ahl_gaps = [r for r in gap_rows if r["method"] == "AHL-DINO"]
    adp_gaps = [r for r in gap_rows if r["method"] == "ADP-only-DINO"]

    def max_abs(rows_: List[Dict], key: str) -> float:
        vals = [abs(float(r[key])) for r in rows_ if r.get(key) != ""]
        return max(vals) if vals else math.nan

    lines.extend([
        "",
        "## Compact Diagnostics",
        "",
        f"- ADP-only max |dF1 GPU-CPU|: {fnum(max_abs(adp_gaps, 'delta_f1_gpu_minus_cpu'))}; max |dAUPR|: {fnum(max_abs(adp_gaps, 'delta_auc_pr_gpu_minus_cpu'))}.",
        f"- AHL-only max |dF1 GPU-CPU|: {fnum(max_abs(ahl_gaps, 'delta_f1_gpu_minus_cpu'))}; compare against CPU rerun vs v2 columns before blaming preprocessing.",
        f"- Bridge max |dF1 GPU-CPU|: {fnum(max_abs(bridge_gaps, 'delta_f1_gpu_minus_cpu'))}; this is the primary production-facing replacement signal.",
        "",
        "Files:",
        "",
        "- `cpu_gpu_full_flow_master_table.csv`",
        "- `cpu_gpu_full_flow_gap_summary.csv`",
        "- `cpu_gpu_full_flow_fixed_cpu_threshold_table.csv`",
    ])
    (args.out_dir / "cpu_gpu_full_flow_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "ok", "out_dir": str(args.out_dir), "rows": len(rows), "gap_rows": len(gap_rows)})


if __name__ == "__main__":
    main()
