#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval-only threshold policy comparison for qm_xiepai 6_1 results.

This script is intentionally decoupled from training. It reads existing
metrics.json and scores.csv files, recalculates deployable threshold policies,
and writes compact comparison tables. It does not retrain or re-score models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from threshold_policies import apply_thresholds, best_f1_threshold, metric_at

STAGES = [f"S{i}" for i in range(9)]
DEFAULT_POLICIES = [
    "production_normal_p95_0",
    "strategy_stage_adaptive",
    "strategy_mild_stage_v2",
    "strategy_stage_v3",
    "recall_at_95_highest_threshold",
    "recall_at_90_highest_threshold",
    "strategy_balanced_f1",
]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_idx(stage: str) -> int:
    text = str(stage).strip().upper()
    if text.startswith("S") and text[1:].isdigit():
        return int(text[1:])
    raise ValueError(f"invalid stage: {stage}")


def load_scores(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"empty scores file: {path}")
    fields = set(rows[0].keys())
    if "split" in fields:
        calib = [r for r in rows if r.get("split") == "val"]
        test = [r for r in rows if r.get("split") == "test"]
    elif "role" in fields:
        calib = [r for r in rows if str(r.get("role", "")).startswith("calib_")]
        test = [r for r in rows if str(r.get("role", "")).startswith("test_")]
    else:
        raise ValueError(f"unknown scores schema: {path}")
    if not calib or not test:
        raise ValueError(f"scores file lacks calib/test rows: {path}")

    def arr(part: Sequence[Dict[str, str]], col: str) -> np.ndarray:
        return np.asarray([float(r[col]) for r in part], dtype=float)

    return arr(calib, "label").astype(int), arr(calib, "score"), arr(test, "label").astype(int), arr(test, "score")


def fpr_from_metric(row: Dict[str, float]) -> float:
    fp = float(row.get("fp", 0.0))
    tn = float(row.get("tn", 0.0))
    denom = fp + tn
    return fp / denom if denom else 0.0


def constrained_recall_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    recall_floor: float,
    fpr_cap: float,
) -> Dict[str, float]:
    """Choose a recall-aware threshold without collapsing into all-positive.

    Selection order:
    1. candidates with recall >= floor and FPR <= cap, ranked by validation F1;
    2. if none, candidates with recall >= floor, ranked by validation F1 and lower FPR;
    3. if none, validation balanced F1.
    """
    candidates: List[Dict[str, float]] = []
    for threshold in np.unique(scores):
        row = metric_at(labels, scores, float(threshold))
        row["calibration_fpr"] = float(fpr_from_metric(row))
        candidates.append(row)
    strict = [r for r in candidates if r["recall"] >= recall_floor and r["calibration_fpr"] <= fpr_cap]
    if strict:
        row = max(strict, key=lambda r: (r["f1"], r["precision"], -r["calibration_fpr"], r["threshold"]))
        rule = "max_f1_with_recall_floor_and_fpr_cap"
    else:
        loose = [r for r in candidates if r["recall"] >= recall_floor]
        if loose:
            row = max(loose, key=lambda r: (r["f1"], -r["calibration_fpr"], r["precision"], r["threshold"]))
            rule = "fallback_max_f1_with_recall_floor"
        else:
            row = best_f1_threshold(labels, scores)
            row["calibration_fpr"] = float(fpr_from_metric(row))
            rule = "fallback_validation_best_f1"
    row = dict(row)
    row["policy_family"] = "strategy_mild_stage_v2"
    row["recall_floor"] = float(recall_floor)
    row["fpr_cap"] = float(fpr_cap)
    row["selection_rule"] = rule
    return row


def mild_stage_v2_threshold(labels: np.ndarray, scores: np.ndarray, stage: str) -> Dict[str, float]:
    idx = stage_idx(stage)
    if idx <= 2:
        row = constrained_recall_threshold(labels, scores, recall_floor=0.90, fpr_cap=0.20)
        row["source_policy"] = "mild_recall90_fpr20"
    elif idx <= 6:
        row = constrained_recall_threshold(labels, scores, recall_floor=0.85, fpr_cap=0.10)
        row["source_policy"] = "mild_recall85_fpr10"
    else:
        row = best_f1_threshold(labels, scores)
        row["policy_family"] = "strategy_mild_stage_v2"
        row["selection_rule"] = "validation_balanced_f1"
        row["source_policy"] = "balanced_f1_late"
        row["calibration_fpr"] = float(fpr_from_metric(row))
    row["stage"] = stage
    return row


def policy_results(
    calib_labels: np.ndarray,
    calib_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    stage: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out = apply_thresholds(calib_labels, calib_scores, test_labels, test_scores, stage=stage)
    mild = mild_stage_v2_threshold(calib_labels, calib_scores, stage)
    out["strategy_mild_stage_v2"] = {
        "calibration": mild,
        "test": metric_at(test_labels, test_scores, float(mild["threshold"])),
    }
    # Stable alias for late-stage balanced policy.
    if "strategy_balanced_f1" not in out:
        balanced = best_f1_threshold(calib_labels, calib_scores)
        balanced["policy_family"] = "strategy_balanced"
        balanced["selection_rule"] = "validation_best_f1"
        out["strategy_balanced_f1"] = {
            "calibration": balanced,
            "test": metric_at(test_labels, test_scores, float(balanced["threshold"])),
        }
    return out


def auc_or_none(labels: np.ndarray, scores: np.ndarray, kind: str) -> Optional[float]:
    try:
        if len(set(labels.tolist())) < 2:
            return None
        if kind == "roc":
            return float(roc_auc_score(labels, scores))
        return float(average_precision_score(labels, scores))
    except Exception:
        return None


def find_stage_metrics(root: Path, stage: str) -> Optional[Path]:
    candidates = [root / "stages" / stage / "metrics" / "metrics.json", root / stage / "metrics" / "metrics.json"]
    for p in candidates:
        if p.exists():
            return p
    found = sorted(root.glob(f"**/{stage}/metrics/metrics.json"))
    return found[0] if found else None


def normalize_policy_name(name: str) -> str:
    return "production_normal_p95" if name == "production_normal_p95_0" else name


def fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def parse_source(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("source must be METHOD=/path/to/output_root")
    method, path = text.split("=", 1)
    return method, Path(path)


def build_rows(sources: Sequence[Tuple[str, Path]], policies: Sequence[str]) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    details: List[Dict] = []
    seen_ok = set()
    for method, src in sources:
        for stage in STAGES:
            metrics_path = find_stage_metrics(src, stage)
            key = (method, stage)
            if metrics_path is None:
                if key not in seen_ok:
                    rows.append({"method": method, "stage": stage, "policy": "ALL", "status": "missing", "reason": f"metrics not found under {src}"})
                continue
            try:
                metrics = read_json(metrics_path)
                status = metrics.get("status", "ok")
                if status != "ok":
                    if key not in seen_ok:
                        rows.append({"method": method, "stage": stage, "policy": "ALL", "status": status, "reason": metrics.get("reason", ""), "metrics_json": str(metrics_path)})
                    continue
                score_file = Path(metrics.get("score_file") or metrics_path.parent / "scores.csv")
                calib_labels, calib_scores, test_labels, test_scores = load_scores(score_file)
                recalculated = policy_results(calib_labels, calib_scores, test_labels, test_scores, stage=stage)
                rows = [r for r in rows if not (r.get("method") == method and r.get("stage") == stage)]
                for policy in policies:
                    item = recalculated.get(policy)
                    if not item:
                        rows.append({"method": method, "stage": stage, "policy": normalize_policy_name(policy), "status": "missing_policy", "reason": f"{policy} not available", "metrics_json": str(metrics_path)})
                        continue
                    test = item.get("test", {})
                    calib = item.get("calibration", {})
                    rows.append({
                        "method": method,
                        "stage": stage,
                        "policy": normalize_policy_name(policy),
                        "status": "ok",
                        "reason": "",
                        "accuracy": test.get("accuracy"),
                        "precision": test.get("precision"),
                        "recall": test.get("recall"),
                        "time_ms": metrics.get("time_ms") or metrics.get("inference_time_ms") or metrics.get("test_time_ms"),
                        "f1": test.get("f1"),
                        "tp": test.get("tp"),
                        "fp": test.get("fp"),
                        "tn": test.get("tn"),
                        "fn": test.get("fn"),
                        "auc_roc": metrics.get("auc_roc") if metrics.get("auc_roc") is not None else auc_or_none(test_labels, test_scores, "roc"),
                        "auc_pr": metrics.get("auc_pr") if metrics.get("auc_pr") is not None else auc_or_none(test_labels, test_scores, "pr"),
                        "threshold": test.get("threshold", calib.get("threshold")),
                        "calib_threshold": calib.get("threshold"),
                        "calib_precision": calib.get("precision"),
                        "calib_recall": calib.get("recall"),
                        "calib_f1": calib.get("f1"),
                        "calib_fpr": calib.get("calibration_fpr"),
                        "source_policy": calib.get("source_policy"),
                        "selection_rule": calib.get("selection_rule"),
                        "policy_family": calib.get("policy_family"),
                        "metrics_json": str(metrics_path),
                        "score_file": str(score_file),
                    })
                details.append({"method": method, "stage": stage, "metrics_json": str(metrics_path), "score_file": str(score_file)})
                seen_ok.add(key)
            except Exception as e:
                rows = [r for r in rows if not (r.get("method") == method and r.get("stage") == stage)]
                rows.append({"method": method, "stage": stage, "policy": "ALL", "status": "failed", "reason": repr(e), "metrics_json": str(metrics_path)})
    return rows, details


def write_outputs(rows: List[Dict], details: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "stage", "policy", "status", "reason", "accuracy", "precision", "recall", "time_ms", "f1",
        "tp", "fp", "tn", "fn", "auc_roc", "auc_pr", "threshold", "calib_threshold", "calib_precision",
        "calib_recall", "calib_f1", "calib_fpr", "source_policy", "selection_rule", "policy_family", "metrics_json", "score_file",
    ]
    with (output_dir / "policy_compare.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "# 20260518 Threshold Policy V2 Eval",
        "",
        "Only existing scores/metrics are reused. No retraining or re-scoring is performed.",
        "",
        "method stage policy status acc prec recall time_ms f1 tp fp tn fn auc_roc auc_pr source_policy",
    ]
    for row in rows:
        lines.append(" ".join([
            str(row.get("method", "")), str(row.get("stage", "")), str(row.get("policy", "")), str(row.get("status", "")),
            fmt(row.get("accuracy")), fmt(row.get("precision")), fmt(row.get("recall")), fmt(row.get("time_ms")), fmt(row.get("f1")),
            str(row.get("tp", "") or ""), str(row.get("fp", "") or ""), str(row.get("tn", "") or ""), str(row.get("fn", "") or ""),
            fmt(row.get("auc_roc")), fmt(row.get("auc_pr")), str(row.get("source_policy", "") or ""),
        ]))
    (output_dir / "policy_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "policy_compare.json", {"status": "ok", "rows": rows, "details": details})


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval threshold policy variants from existing qm_xiepai scores.")
    parser.add_argument("--source", action="append", type=parse_source, required=True, help="METHOD=/path/to/output_root")
    parser.add_argument("--policy", action="append", default=None, help="Policy key to output. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    policies = args.policy or DEFAULT_POLICIES
    rows, details = build_rows(args.source, policies)
    write_outputs(rows, details, args.output_dir)
    print(json.dumps({"status": "ok", "rows": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
