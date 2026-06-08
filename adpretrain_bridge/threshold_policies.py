#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold policies shared by qm_xiepai offline and val-calibrated evaluators.

The deployable policies are intentionally separated from the offline oracle:

* production_normal_pXX: common production baseline, calibrated from normal
  validation scores to control the validation-side false positive rate.
* strategy_stage_adaptive: line-start strategy. Early stages prefer high recall
  to avoid missed defects and collect anomaly samples; later stages return to a
  balanced validation F1 policy.

test_best_f1_oracle is kept only for offline comparison with older result
tables. It must not be used as a production threshold.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score


def metric_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2.0 * precisions * recalls) / (precisions + recalls + 1e-12)
    if len(thresholds) == 0:
        row = metric_at(labels, scores, float(scores.max()) if len(scores) else 0.0)
        row["source_precision_recall_curve_f1"] = float(f1_scores[0]) if len(f1_scores) else row["f1"]
        return row
    valid = f1_scores[:-1]
    idx = int(np.nanargmax(valid))
    row = metric_at(labels, scores, float(thresholds[idx]))
    row["source_precision_recall_curve_f1"] = float(valid[idx])
    return row


def normal_percentile_threshold(labels: np.ndarray, scores: np.ndarray, percentile: float) -> Dict[str, float]:
    normal_scores = scores[labels == 0]
    if len(normal_scores) == 0:
        raise ValueError("normal percentile threshold requires at least one normal validation score")
    row = metric_at(labels, scores, float(np.percentile(normal_scores, percentile)))
    row["policy_family"] = "production_normal_percentile"
    row["normal_percentile"] = float(percentile)
    row["expected_validation_fpr"] = float((100.0 - percentile) / 100.0)
    return row


def recall_floor_threshold(labels: np.ndarray, scores: np.ndarray, recall_floor: float) -> Dict[str, float]:
    """Choose the minimum threshold that reaches the validation recall floor."""
    anomaly_scores = scores[labels == 1]
    if len(anomaly_scores) == 0:
        raise ValueError("recall floor threshold requires at least one anomaly validation score")
    candidates = np.unique(scores)
    viable = []
    for threshold in candidates:
        row = metric_at(labels, scores, float(threshold))
        if row["recall"] >= recall_floor:
            viable.append(row)
    if not viable:
        threshold = float(np.min(anomaly_scores))
        row = metric_at(labels, scores, threshold)
    else:
        # Early production rollout favors catching defects over suppressing
        # reports, so use the lowest viable threshold.
        row = min(viable, key=lambda x: (x["threshold"], -x["recall"], -x["f1"]))
    row["policy_family"] = "strategy_recall_first"
    row["recall_floor"] = float(recall_floor)
    row["selection_rule"] = "min_threshold_with_recall_floor"
    return row



def recall_floor_highest_threshold(labels: np.ndarray, scores: np.ndarray, recall_floor: float) -> Dict[str, float]:
    """Choose the highest threshold whose validation recall reaches the floor."""
    anomaly_scores = scores[labels == 1]
    if len(anomaly_scores) == 0:
        raise ValueError("recall floor threshold requires at least one anomaly validation score")
    viable = []
    for threshold in np.unique(scores):
        row = metric_at(labels, scores, float(threshold))
        if row["recall"] >= recall_floor:
            viable.append(row)
    if not viable:
        row = best_f1_threshold(labels, scores)
        row["fallback_reason"] = "no_threshold_reaches_recall_floor"
    else:
        row = max(viable, key=lambda x: (x["threshold"], x["precision"], x["f1"]))
    row["policy_family"] = "recall_highest_threshold"
    row["recall_floor"] = float(recall_floor)
    row["selection_rule"] = "highest_threshold_with_recall_floor"
    return row


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
    policy_family: str,
) -> Dict[str, float]:
    candidates = []
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
    row["policy_family"] = policy_family
    row["recall_floor"] = float(recall_floor)
    row["fpr_cap"] = float(fpr_cap)
    row["selection_rule"] = rule
    return row


def best_f1_with_fpr_cap(labels: np.ndarray, scores: np.ndarray, fpr_cap: float) -> Dict[str, float]:
    candidates = []
    for threshold in np.unique(scores):
        row = metric_at(labels, scores, float(threshold))
        row["calibration_fpr"] = float(fpr_from_metric(row))
        candidates.append(row)
    strict = [r for r in candidates if r["calibration_fpr"] <= fpr_cap]
    if strict:
        row = max(strict, key=lambda r: (r["f1"], r["precision"], -r["calibration_fpr"], r["threshold"]))
        rule = "max_f1_with_fpr_cap"
    else:
        row = best_f1_threshold(labels, scores)
        row["calibration_fpr"] = float(fpr_from_metric(row))
        rule = "fallback_validation_best_f1"
    row = dict(row)
    row["fpr_cap"] = float(fpr_cap)
    row["selection_rule"] = rule
    return row


def mild_stage_v2_1_safe_threshold(labels: np.ndarray, scores: np.ndarray, stage: Optional[str]) -> Dict[str, float]:
    idx = stage_index(stage)
    if idx is None or idx <= 2:
        row = constrained_recall_threshold(labels, scores, 0.90, 0.20, "strategy_mild_stage_v2_1_safe")
        row["source_policy"] = "mild_recall90_fpr20"
    elif idx <= 6:
        row = constrained_recall_threshold(labels, scores, 0.85, 0.10, "strategy_mild_stage_v2_1_safe")
        row["source_policy"] = "mild_recall85_fpr10"
    else:
        row = best_f1_with_fpr_cap(labels, scores, 0.10)
        row["policy_family"] = "strategy_mild_stage_v2_1_safe"
        row["source_policy"] = "late_best_f1_fpr10"
    row["stage"] = "" if stage is None else str(stage)
    return row

def production_thresholds(calib_labels: np.ndarray, calib_scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for percentile in [95.0, 97.5, 99.0]:
        suffix = str(percentile).replace(".", "_")
        row = normal_percentile_threshold(calib_labels, calib_scores, percentile)
        out["production_normal_p" + suffix] = row
        # Legacy aliases keep old command-line policies usable.
        out["val_normal_p" + suffix] = dict(row)
    return out


def strategy_thresholds(calib_labels: np.ndarray, calib_scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {
        "strategy_recall_first_r95": recall_floor_threshold(calib_labels, calib_scores, 0.95),
        "strategy_recall_first_r90": recall_floor_threshold(calib_labels, calib_scores, 0.90),
        "recall_at_95_highest_threshold": recall_floor_highest_threshold(calib_labels, calib_scores, 0.90),
        "recall_at_90_highest_threshold": recall_floor_highest_threshold(calib_labels, calib_scores, 0.85),
        "strategy_balanced_f1": best_f1_threshold(calib_labels, calib_scores),
    }
    out["strategy_balanced_f1"]["policy_family"] = "strategy_balanced"
    out["strategy_balanced_f1"]["selection_rule"] = "validation_best_f1"
    return out


def stage_index(stage: Optional[str]) -> Optional[int]:
    if stage is None:
        return None
    text = str(stage).strip().upper()
    if text.startswith("S") and text[1:].isdigit():
        return int(text[1:])
    return None


def adaptive_strategy_source(stage: Optional[str]) -> str:
    idx = stage_index(stage)
    if idx is None or idx <= 4:
        return "strategy_recall_first_r95"
    if idx <= 6:
        return "strategy_recall_first_r90"
    return "strategy_balanced_f1"



def stage_v3_strategy_source(stage: Optional[str]) -> str:
    idx = stage_index(stage)
    if idx is None or idx <= 4:
        return "recall_at_95_highest_threshold"
    if idx <= 6:
        return "recall_at_90_highest_threshold"
    return "strategy_balanced_f1"


def choose_thresholds(calib_labels: np.ndarray, calib_scores: np.ndarray, stage: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    out.update(production_thresholds(calib_labels, calib_scores))
    strategy = strategy_thresholds(calib_labels, calib_scores)
    out.update(strategy)
    source_policy = adaptive_strategy_source(stage)
    adaptive = dict(strategy[source_policy])
    adaptive["policy_family"] = "strategy_stage_adaptive"
    adaptive["source_policy"] = source_policy
    adaptive["stage"] = "" if stage is None else str(stage)
    out["strategy_stage_adaptive"] = adaptive

    v3_source_policy = stage_v3_strategy_source(stage)
    v3 = dict(strategy[v3_source_policy])
    v3["policy_family"] = "strategy_stage_v3_highest_threshold"
    v3["source_policy"] = v3_source_policy
    v3["stage"] = "" if stage is None else str(stage)
    out["strategy_stage_v3"] = v3

    out["strategy_mild_stage_v2_1_safe"] = mild_stage_v2_1_safe_threshold(calib_labels, calib_scores, stage)

    legacy_best = best_f1_threshold(calib_labels, calib_scores)
    legacy_best["policy_family"] = "offline_reference"
    legacy_best["selection_rule"] = "validation_best_f1"
    out["val_best_f1"] = legacy_best
    return out


def apply_thresholds(
    calib_labels: np.ndarray,
    calib_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    stage: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, calib_metric in choose_thresholds(calib_labels, calib_scores, stage=stage).items():
        threshold = calib_metric["threshold"]
        out[name] = {
            "calibration": calib_metric,
            "test": metric_at(test_labels, test_scores, threshold),
        }
    out["test_best_f1_oracle"] = {
        "calibration": {},
        "test": best_f1_threshold(test_labels, test_scores),
        "note": "offline reference only; threshold is selected on the held-out test labels",
    }
    return out
