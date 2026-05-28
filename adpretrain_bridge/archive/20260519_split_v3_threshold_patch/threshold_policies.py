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
