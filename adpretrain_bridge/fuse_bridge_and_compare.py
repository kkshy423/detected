#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse GPU offline ADP-only + AHL scores into bridge v2, pick stage_recall_bestf1
thresholds on calib, evaluate on test, and emit a GPU-vs-v2(CPU) comparison table.

Reads:
  - AHL GPU scores:  <ahl_output>/stages/S*/metrics/scores.csv
  - ADP GPU scores:  <adp_output>/stages/S*/metrics/scores.csv
  - v2 main table:   summary/20260603.../final_bridge_v2_main_table.csv (CPU reference)
Writes gpu_offline_master_table.{md,csv} + gpu_vs_v2_gap.md under summary dir.

bridge v2 alpha: S1-2=0.70, S3-4=0.60, S5-7=0.35, S8=0.70 (S0 N/A).
threshold: S0-2 best-F1 R>=0.90, S3-6 R>=0.85, S7-8 best-F1. calib picks, test reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

STAGES = [f"S{i}" for i in range(9)]


def bridge_alpha(stage: str) -> Optional[float]:
    idx = int(stage[1:])
    if idx == 0:
        return None
    if idx <= 2:
        return 0.70
    if idx <= 4:
        return 0.60
    if idx <= 7:
        return 0.35
    return 0.70


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


def robust_norm(x: np.ndarray, c: float, s: float) -> np.ndarray:
    return (x - c) / s


def load_scores(path: Path) -> Dict[str, Dict]:
    """Return {split: {key: (label, score)}} where key = role + '/' + basename.

    Handles both schemas: ADP-only GPU (role,split,label,score,source_rel) and
    AHL offline (role,label,score,stage_rel,source_rel). split is derived from
    role prefix; source_rel differs between the two so we align on role+basename.
    """
    out: Dict[str, Dict] = {"calib": {}, "test": {}}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = row["role"]
            split = "calib" if role.startswith("calib") else ("test" if role.startswith("test") else None)
            if split is None:
                continue
            base = Path(row["source_rel"]).name
            key = f"{role}/{base}"
            out[split][key] = (int(float(row["label"])), float(row["score"]))
    return out


def confusion(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict:
    preds = (scores >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum()); fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum()); fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def choose_threshold(labels: np.ndarray, scores: np.ndarray, floor: Optional[float]) -> Dict:
    cands = [confusion(labels, scores, float(t)) for t in np.unique(scores)]
    if floor is not None:
        ok = [c for c in cands if c["recall"] >= floor]
        if ok:
            return max(ok, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        return max(cands, key=lambda r: (r["recall"], r["f1"], r["precision"], r["threshold"]))
    return max(cands, key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]))


def add_auc(labels: np.ndarray, scores: np.ndarray, row: Dict) -> Dict:
    if len(np.unique(labels)) > 1:
        row["auc_roc"] = float(roc_auc_score(labels, scores))
        row["auc_pr"] = float(average_precision_score(labels, scores))
    else:
        row["auc_roc"] = float("nan"); row["auc_pr"] = float("nan")
    return row


def aligned(adp: Dict, ahl: Dict, split: str):
    """Return label, adp_score, ahl_score arrays aligned by common source_rel."""
    keys = sorted(set(adp[split]) & set(ahl[split]))
    labels = np.array([adp[split][k][0] for k in keys], dtype=int)
    adp_s = np.array([adp[split][k][1] for k in keys], dtype=float)
    ahl_s = np.array([ahl[split][k][1] for k in keys], dtype=float)
    return keys, labels, adp_s, ahl_s


def eval_method(stage: str, calib_labels, calib_scores, test_labels, test_scores) -> Dict:
    floor = stage_recall_floor(stage)
    thr = float(choose_threshold(calib_labels, calib_scores, floor)["threshold"])
    m = add_auc(test_labels, test_scores, confusion(test_labels, test_scores, thr))
    return m


def load_v2(path: Path) -> Dict[Tuple[str, str], Dict]:
    mp = {"ADP-only-DINO": "ADP-only-DINO", "AHL-DINO": "AHL-DINO", "ADP-AHL bridge v2": "ADP-AHL-bridge-v2"}
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = mp.get(row.get("Method", "").strip())
            if m:
                out[(m, row.get("Stage", "").strip())] = row
    return out


def fnum(v, n=4):
    try:
        f = float(v)
        return "nan" if np.isnan(f) else f"{f:.{n}f}"
    except (ValueError, TypeError):
        return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ahl-output", type=Path, required=True)
    p.add_argument("--adp-output", type=Path, required=True)
    p.add_argument("--v2-table", type=Path, default=Path("/ghome/huangjd/code/detected/adpretrain_bridge/summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv"))
    p.add_argument("--summary-root", type=Path, required=True)
    args = p.parse_args()
    v2 = load_v2(args.v2_table)
    perf_rows: List[Dict] = []

    for stage in STAGES:
        adp_path = args.adp_output / "stages" / stage / "metrics" / "scores.csv"
        ahl_path = args.ahl_output / "stages" / stage / "metrics" / "scores.csv"
        if not adp_path.exists() or not ahl_path.exists():
            print(f"skip {stage}: missing scores"); continue
        adp = load_scores(adp_path); ahl = load_scores(ahl_path)
        _, c_lab, c_adp, c_ahl = aligned(adp, ahl, "calib")
        _, t_lab, t_adp, t_ahl = aligned(adp, ahl, "test")

        # ADP-only, AHL
        perf_rows.append({"method": "ADP-only-DINO", "stage": stage, "alpha": "", **eval_method(stage, c_lab, c_adp, t_lab, t_adp)})
        perf_rows.append({"method": "AHL-DINO", "stage": stage, "alpha": "", **eval_method(stage, c_lab, c_ahl, t_lab, t_ahl)})

        # bridge
        alpha = bridge_alpha(stage)
        if alpha is not None:
            adp_c, adp_s = robust_norm_params(c_adp[c_lab == 0]); ahl_c, ahl_s = robust_norm_params(c_ahl[c_lab == 0])
            c_bridge = alpha * robust_norm(c_adp, adp_c, adp_s) + (1 - alpha) * robust_norm(c_ahl, ahl_c, ahl_s)
            t_bridge = alpha * robust_norm(t_adp, adp_c, adp_s) + (1 - alpha) * robust_norm(t_ahl, ahl_c, ahl_s)
            perf_rows.append({"method": "ADP-AHL-bridge-v2", "stage": stage, "alpha": alpha, **eval_method(stage, c_lab, c_bridge, t_lab, t_bridge)})

    args.summary_root.mkdir(parents=True, exist_ok=True)
    # CSV
    with (args.summary_root / "gpu_offline_master_table.csv").open("w", encoding="utf-8", newline="") as f:
        keys = ["method", "stage", "alpha", "threshold", "precision", "recall", "f1", "accuracy", "auc_roc", "auc_pr", "tp", "fp", "tn", "fn"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in perf_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # MD with v2 compare
    lines = ["# GPU offline (uint8, retrained) vs v2 main table (CPU)", "",
             "GPU 列=本轮离线(GPU uint8 预处理+重训 AHL)；v2 列=CPU 离线权威表。同口径可并读。", "",
             "| method | stage | alpha | GPU P | GPU R | GPU F1 | GPU AUPR | v2 F1 | v2 R | v2 AUPR | ΔF1 | ΔR |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in perf_rows:
        ref = v2.get((r["method"], r["stage"]), {})
        v2f1 = ref.get("F1", ""); v2r = ref.get("R", ""); v2pr = ref.get("AUC-PR", "")
        df1 = abs(r["f1"] - float(v2f1)) if v2f1 not in ("", None) else float("nan")
        dr = abs(r["recall"] - float(v2r)) if v2r not in ("", None) else float("nan")
        lines.append(f"| {r['method']} | {r['stage']} | {fnum(r['alpha'],2)} | {fnum(r['precision'])} | {fnum(r['recall'])} | "
                     f"{fnum(r['f1'])} | {fnum(r['auc_pr'])} | {fnum(v2f1)} | {fnum(v2r)} | {fnum(v2pr)} | {fnum(df1)} | {fnum(dr)} |")
    (args.summary_root / "gpu_offline_master_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(perf_rows), "summary": str(args.summary_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

