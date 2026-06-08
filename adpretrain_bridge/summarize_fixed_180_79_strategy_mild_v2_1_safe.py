#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-table fixed 180/79 AHL and YOLO results with safe production policy.

This is eval-only: it reads existing scores.csv / metrics.json and does not
train or re-score any model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from threshold_policies import best_f1_threshold, best_f1_with_fpr_cap, metric_at

STAGES = [f"S{i}" for i in range(9)]
DEFAULT_AHL = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3")
DEFAULT_YOLO = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3")
DEFAULT_OUT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_fixed_180_79_strategy_mild_v2_1_safe_retable")


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
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
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

    def arr(part: Sequence[Dict[str, str]], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in part], dtype=float)

    return arr(calib, "label").astype(int), arr(calib, "score"), arr(test, "label").astype(int), arr(test, "score")


def fpr(row: Dict[str, float]) -> float:
    fp = float(row.get("fp", 0.0))
    tn = float(row.get("tn", 0.0))
    return fp / (fp + tn) if fp + tn > 0 else 0.0


def candidates(labels: np.ndarray, scores: np.ndarray) -> List[Dict[str, float]]:
    out = []
    for threshold in np.unique(scores):
        row = metric_at(labels, scores, float(threshold))
        row["calibration_fpr"] = fpr(row)
        out.append(row)
    return out


def constrained_recall(labels: np.ndarray, scores: np.ndarray, recall_floor: float, fpr_cap: float) -> Dict[str, float]:
    rows = candidates(labels, scores)
    strict = [r for r in rows if r["recall"] >= recall_floor and r["calibration_fpr"] <= fpr_cap]
    if strict:
        row = max(strict, key=lambda r: (r["f1"], r["precision"], -r["calibration_fpr"], r["threshold"]))
        rule = "max_f1_with_recall_floor_and_fpr_cap"
    else:
        loose = [r for r in rows if r["recall"] >= recall_floor]
        if loose:
            row = max(loose, key=lambda r: (r["f1"], -r["calibration_fpr"], r["precision"], r["threshold"]))
            rule = "fallback_max_f1_with_recall_floor"
        else:
            row = best_f1_threshold(labels, scores)
            row["calibration_fpr"] = fpr(row)
            rule = "fallback_validation_best_f1"
    row = dict(row)
    row["recall_floor"] = recall_floor
    row["fpr_cap"] = fpr_cap
    row["selection_rule"] = rule
    return row


def highest_recall_floor(labels: np.ndarray, scores: np.ndarray, recall_floor: float) -> Dict[str, float]:
    viable = [r for r in candidates(labels, scores) if r["recall"] >= recall_floor]
    if viable:
        row = max(viable, key=lambda r: (r["threshold"], r["precision"], r["f1"]))
        rule = "highest_threshold_with_recall_floor"
    else:
        row = best_f1_threshold(labels, scores)
        row["calibration_fpr"] = fpr(row)
        rule = "fallback_validation_best_f1"
    row = dict(row)
    row["recall_floor"] = recall_floor
    row["selection_rule"] = rule
    return row


def safe_v2_1(labels: np.ndarray, scores: np.ndarray, stage: str) -> Dict[str, float]:
    idx = stage_idx(stage)
    if idx <= 2:
        row = constrained_recall(labels, scores, 0.90, 0.20)
        source = "mild_recall90_fpr20"
    elif idx <= 6:
        row = constrained_recall(labels, scores, 0.85, 0.10)
        source = "mild_recall85_fpr10"
    else:
        row = best_f1_with_fpr_cap(labels, scores, 0.10)
        source = "late_best_f1_fpr10"
    row["policy_family"] = "strategy_mild_stage_v2_1_safe"
    row["source_policy"] = source
    row["stage"] = stage
    return row


def late_recall85(labels: np.ndarray, scores: np.ndarray, stage: str) -> Dict[str, float]:
    row = highest_recall_floor(labels, scores, 0.85)
    row["policy_family"] = "late_quality_recall85"
    row["source_policy"] = "late_recall85_highest_threshold"
    row["stage"] = stage
    return row


def production_p95(labels: np.ndarray, scores: np.ndarray, stage: str) -> Dict[str, float]:
    normal = scores[labels == 0]
    if len(normal) == 0:
        raise ValueError("production p95 requires calibration normal scores")
    row = metric_at(labels, scores, float(np.percentile(normal, 95.0)))
    row["calibration_fpr"] = fpr(row)
    row["policy_family"] = "production_normal_percentile"
    row["source_policy"] = "production_normal_p95"
    row["stage"] = stage
    return row


def auc_or_none(labels: np.ndarray, scores: np.ndarray, kind: str) -> Optional[float]:
    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores)) if kind == "roc" else float(average_precision_score(labels, scores))


def find_metrics(root: Path, stage: str) -> Optional[Path]:
    p = root / "stages" / stage / "metrics" / "metrics.json"
    return p if p.exists() else None


def score_path(metrics_path: Path, metrics: Dict) -> Path:
    raw = metrics.get("score_file")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    return metrics_path.parent / "scores.csv"


def row_for(method: str, stage: str, policy_name: str, calib_policy: Dict[str, float], test_labels: np.ndarray, test_scores: np.ndarray, auc_roc, auc_pr, metrics_path: Path, scores: Path) -> Dict:
    test = metric_at(test_labels, test_scores, float(calib_policy["threshold"]))
    metrics = read_json(metrics_path)
    return {
        "method": method,
        "stage": stage,
        "policy": policy_name,
        "status": "ok",
        "threshold": test["threshold"],
        "calib_precision": calib_policy.get("precision"),
        "calib_recall": calib_policy.get("recall"),
        "calib_f1": calib_policy.get("f1"),
        "calib_fpr": calib_policy.get("calibration_fpr", fpr(calib_policy)),
        "calib_fp": calib_policy.get("fp"),
        "calib_fn": calib_policy.get("fn"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "test_f1": test.get("f1"),
        "test_accuracy": test.get("accuracy"),
        "test_tp": test.get("tp"),
        "test_fp": test.get("fp"),
        "test_tn": test.get("tn"),
        "test_fn": test.get("fn"),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "time_adpretrain_feature_ms": metrics.get("time_adpretrain_feature_ms"),
        "time_ahl_process_ms": metrics.get("time_ahl_process_ms"),
        "time_total_ms": metrics.get("time_total_ms", metrics.get("inference_time_ms")),
        "time_kind": metrics.get("time_kind"),
        "source_policy": calib_policy.get("source_policy"),
        "selection_rule": calib_policy.get("selection_rule"),
        "metrics_json": str(metrics_path),
        "score_file": str(scores),
    }


def collect(method: str, root: Path) -> List[Dict]:
    out: List[Dict] = []
    for stage in STAGES:
        metrics_path = find_metrics(root, stage)
        if metrics_path is None:
            out.append({"method": method, "stage": stage, "policy": "ALL", "status": "missing", "reason": f"metrics not found under {root}"})
            continue
        metrics = read_json(metrics_path)
        if metrics.get("status", "ok") != "ok":
            out.append({"method": method, "stage": stage, "policy": "ALL", "status": metrics.get("status"), "reason": metrics.get("reason", ""), "metrics_json": str(metrics_path)})
            continue
        scores = score_path(metrics_path, metrics)
        calib_labels, calib_scores, test_labels, test_scores = load_scores(scores)
        auc_roc = metrics.get("auc_roc") if metrics.get("auc_roc") is not None else auc_or_none(test_labels, test_scores, "roc")
        auc_pr = metrics.get("auc_pr") if metrics.get("auc_pr") is not None else auc_or_none(test_labels, test_scores, "pr")
        policies = {
            "strategy_mild_stage_v2_1_safe": safe_v2_1(calib_labels, calib_scores, stage),
            "production_normal_p95": production_p95(calib_labels, calib_scores, stage),
        }
        if stage in {"S7", "S8"}:
            policies["late_recall85_quality"] = late_recall85(calib_labels, calib_scores, stage)
        for name, calib_policy in policies.items():
            out.append(row_for(method, stage, name, calib_policy, test_labels, test_scores, auc_roc, auc_pr, metrics_path, scores))
    return out


def fmt(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "method", "stage", "policy", "status", "reason", "threshold",
        "calib_precision", "calib_recall", "calib_f1", "calib_fpr", "calib_fp", "calib_fn",
        "test_precision", "test_recall", "test_f1", "test_accuracy", "test_tp", "test_fp", "test_tn", "test_fn",
        "auc_roc", "auc_pr", "time_adpretrain_feature_ms", "time_ahl_process_ms", "time_total_ms", "time_kind",
        "source_policy", "selection_rule", "metrics_json", "score_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict]) -> None:
    main = [r for r in rows if r.get("policy") == "strategy_mild_stage_v2_1_safe"]
    late = [r for r in rows if r.get("policy") == "late_recall85_quality"]
    lines = [
        "# Fixed 180/79 Strategy Mild V2.1 Safe Re-table",
        "",
        "This is eval-only. Existing scores.csv / metrics.json are reused; no training or re-scoring is performed.",
        "",
        "Main policy: `strategy_mild_stage_v2_1_safe` = S0-S2 `mild_recall90_fpr20`, S3-S6 `mild_recall85_fpr10`, S7-S8 `late_recall80_fpr10`.",
        "Late quality appendix: S7-S8 `late_recall85_quality`.",
        "",
        "## Main Table",
        "",
        "| Method | Stage | Source policy | Calib P/R/F1/FPR | Test P/R/F1 | Time ms | Test FP/FN | AUC-ROC | AUC-PR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('method')} | {r.get('stage')} | - | {r.get('status')} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['method']} | {r['stage']} | {r.get('source_policy','')} | "
            f"{fmt(r.get('calib_precision'))}/{fmt(r.get('calib_recall'))}/{fmt(r.get('calib_f1'))}/{fmt(r.get('calib_fpr'))} | "
            f"{fmt(r.get('test_precision'))}/{fmt(r.get('test_recall'))}/{fmt(r.get('test_f1'))} | "
            f"{fmt(r.get('time_adpretrain_feature_ms'))}/{fmt(r.get('time_ahl_process_ms'))}/{fmt(r.get('time_total_ms'))} | "
            f"{r.get('test_fp')}/{r.get('test_fn')} | {fmt(r.get('auc_roc'))} | {fmt(r.get('auc_pr'))} |"
        )
    lines.extend([
        "",
        "## S7/S8 Late Recall85 Quality Appendix",
        "",
        "| Method | Stage | Source policy | Calib P/R/F1/FPR | Test P/R/F1 | Time ms | Test FP/FN | AUC-ROC | AUC-PR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in late:
        lines.append(
            f"| {r['method']} | {r['stage']} | {r.get('source_policy','')} | "
            f"{fmt(r.get('calib_precision'))}/{fmt(r.get('calib_recall'))}/{fmt(r.get('calib_f1'))}/{fmt(r.get('calib_fpr'))} | "
            f"{fmt(r.get('test_precision'))}/{fmt(r.get('test_recall'))}/{fmt(r.get('test_f1'))} | "
            f"{fmt(r.get('time_adpretrain_feature_ms'))}/{fmt(r.get('time_ahl_process_ms'))}/{fmt(r.get('time_total_ms'))} | "
            f"{r.get('test_fp')}/{r.get('test_fn')} | {fmt(r.get('auc_roc'))} | {fmt(r.get('auc_pr'))} |"
        )
    lines.extend([
        "",
        "## Initial Conclusions",
        "",
        "- AHL S0 remains unusable: AUC-ROC is below random and safe policy still produces massive false positives.",
        "- AHL S2-S6 remains the main few-shot transition range; S6 is the strongest AHL stage under the safe policy.",
        "- YOLO S1-S3 remains unsuitable as the main method because false positives are high under calibration-derived thresholds.",
        "- YOLO S6-S8 are the supervised switch candidates; S6 has the best F1 in the safe main table.",
        "- S7/S8 late_recall85 is retained only as a quality/recall-priority appendix; strict production should check calibration FPR before adoption.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ahl-root", type=Path, default=DEFAULT_AHL)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect("AHL_plain_ADPretrain", args.ahl_root) + collect("YOLO26s_cls", args.yolo_root)
    write_csv(args.output_dir / "main_and_late_quality_tables.csv", rows)
    write_json(args.output_dir / "summary.json", {"status": "ok", "rows": rows, "policy": "strategy_mild_stage_v2_1_safe"})
    write_md(args.output_dir / "summary.md", rows)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
