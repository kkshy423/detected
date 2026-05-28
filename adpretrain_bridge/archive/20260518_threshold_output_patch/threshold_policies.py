#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold policies shared by qm_xiepai offline and val-calibrated evaluators."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score


def metric_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
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
    return metric_at(labels, scores, float(np.percentile(normal_scores, percentile)))


def choose_thresholds(calib_labels: np.ndarray, calib_scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {"val_best_f1": best_f1_threshold(calib_labels, calib_scores)}
    for p in [95.0, 97.5, 99.0]:
        key = "val_normal_p" + str(p).replace(".", "_")
        out[key] = normal_percentile_threshold(calib_labels, calib_scores, p)
    return out


def apply_thresholds(
    calib_labels: np.ndarray,
    calib_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, calib_metric in choose_thresholds(calib_labels, calib_scores).items():
        threshold = calib_metric["threshold"]
        out[name] = {
            "calibration": calib_metric,
            "test": metric_at(test_labels, test_scores, threshold),
        }
    out["test_best_f1_oracle"] = {
        "calibration": {},
        "test": best_f1_threshold(test_labels, test_scores),
    }
    return out
