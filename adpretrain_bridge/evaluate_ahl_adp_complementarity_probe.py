#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-wise AHL vs ADP score complementarity probe on existing scores."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = {
    "AHL-DINO": Path("output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages"),
    "ADP-only-DINO": Path(
        "output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe/stages"
    ),
}

STAGES = ["S5", "S6", "S7", "S8"]
ALPHAS = np.round(np.linspace(0.0, 1.0, 101), 2)

BASELINE_SUMMARY = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/summary/20260530_stage_recall_bestf1_policy/metrics_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/summary/20260530_ahl_adp_complementarity_probe"
)


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
    rows = read_csv(path)
    out: Dict[str, Dict[str, Dict[str, object]]] = {"calib": {}, "test": {}}
    for row in rows:
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
    ids = sorted(data.keys())
    labels = np.asarray([int(data[sid]["label"]) for sid in ids], dtype=np.int64)
    scores = np.asarray([float(data[sid]["score"]) for sid in ids], dtype=np.float64)
    return SplitData(labels=labels, scores=scores, ids=ids)


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
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
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


def robust_norm_params(calib_normal_scores: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(calib_normal_scores))
    mad = float(np.median(np.abs(calib_normal_scores - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q75, q25 = np.percentile(calib_normal_scores, [75, 25])
        scale = float((q75 - q25) / 1.349) if (q75 - q25) > 1e-12 else float(np.std(calib_normal_scores))
    if scale <= 1e-12:
        scale = 1.0
    return med, scale


def robust_normalize(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (scores - center) / scale


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    candidates = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores, float(threshold))
        row["threshold"] = float(threshold)
        candidates.append(row)
    selected = max(candidates, key=lambda r: (r["f1"], r["recall"], r["precision"], r["threshold"]))
    return dict(selected)


def format_float(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def load_baselines() -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv(BASELINE_SUMMARY)
    out = {}
    for row in rows:
        if row["method"] in METHODS and row["stage"] in STAGES and row["status"] == "ok":
            out[(row["method"], row["stage"])] = row
    return out


def align_split(adp: Dict[str, Dict[str, object]], ahl: Dict[str, Dict[str, object]]) -> Tuple[SplitData, SplitData]:
    adp_data = pack_split(adp)
    ahl_data = pack_split(ahl)
    if adp_data.ids != ahl_data.ids:
        adp_set = set(adp_data.ids)
        ahl_set = set(ahl_data.ids)
        missing_adp = sorted(ahl_set - adp_set)
        missing_ahl = sorted(adp_set - ahl_set)
        raise ValueError(
            f"Sample id mismatch between ADP and AHL. Missing in ADP: {missing_adp[:5]}, "
            f"missing in AHL: {missing_ahl[:5]}"
        )
    if not np.array_equal(adp_data.labels, ahl_data.labels):
        diff = np.where(adp_data.labels != ahl_data.labels)[0][:5]
        raise ValueError(f"Label mismatch between ADP and AHL at indices {diff.tolist()}")
    return adp_data, ahl_data


def choose_fusion_alpha(
    adp_calib: np.ndarray,
    ahl_calib: np.ndarray,
    adp_test: np.ndarray,
    ahl_test: np.ndarray,
    calib_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None
    for alpha in ALPHAS:
        fusion_calib = alpha * adp_calib + (1.0 - alpha) * ahl_calib
        fusion_test = alpha * adp_test + (1.0 - alpha) * ahl_test
        threshold_row = best_f1_threshold(calib_labels, fusion_calib)
        test_metric = confusion(test_labels, fusion_test, threshold_row["threshold"])
        auc_roc, auc_pr = aucs(test_labels, fusion_test)
        row = {
            "alpha": float(alpha),
            "threshold": float(threshold_row["threshold"]),
            "calib_precision": float(threshold_row["precision"]),
            "calib_recall": float(threshold_row["recall"]),
            "calib_f1": float(threshold_row["f1"]),
            "calib_accuracy": float(threshold_row["accuracy"]),
            "calib_fpr": float(threshold_row["fpr"]),
            "test_precision": float(test_metric["precision"]),
            "test_recall": float(test_metric["recall"]),
            "test_f1": float(test_metric["f1"]),
            "test_accuracy": float(test_metric["accuracy"]),
            "test_fpr": float(test_metric["fpr"]),
            "test_tp": int(test_metric["tp"]),
            "test_fp": int(test_metric["fp"]),
            "test_tn": int(test_metric["tn"]),
            "test_fn": int(test_metric["fn"]),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
        }
        rows.append(row)
        key = (row["calib_f1"], row["calib_recall"], row["calib_precision"], row["alpha"])
        if best is None or key > (
            best["calib_f1"],
            best["calib_recall"],
            best["calib_precision"],
            best["alpha"],
        ):
            best = row
    assert best is not None
    return rows, best


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in fieldnames} for row in rows])


def build_markdown(
    baseline_rows: Sequence[Dict[str, str]],
    fusion_rows: Sequence[Dict[str, float]],
    overlap_rows: Sequence[Dict[str, int]],
    alpha_rows: Sequence[Dict[str, float]],
) -> str:
    lines = [
        "# 20260530 AHL vs ADP Score Complementarity Probe",
        "",
        "Eval-only diagnostic. Alpha and fusion threshold are selected from calibration scores only; test labels are only used for final reporting.",
        "",
        "Robust normalization: calibration-normal median/MAD per method and stage.",
        "",
        "## Baseline and Fusion Metrics",
        "",
        "| Method | Stage | Alpha | Threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stage_to_fusion = {(r["stage"]): r for r in fusion_rows}
    for stage in STAGES:
        for method in ["ADP-only-DINO", "AHL-DINO"]:
            row = baseline_rows[(method, stage)]
            lines.append(
                f"| {method} | {stage} |  | {format_float(row['threshold'])} | {format_float(row['precision'])} | "
                f"{format_float(row['recall'])} | {format_float(row['f1'])} | {format_float(row['accuracy'])} | "
                f"{format_float(row['auc_roc'])} | {format_float(row['auc_pr'])} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} |"
            )
        fus = stage_to_fusion[stage]
        lines.append(
            f"| fusion | {stage} | {format_float(fus['alpha'])} | {format_float(fus['threshold'])} | "
            f"{format_float(fus['test_precision'])} | {format_float(fus['test_recall'])} | {format_float(fus['test_f1'])} | "
            f"{format_float(fus['test_accuracy'])} | {format_float(fus['auc_roc'])} | {format_float(fus['auc_pr'])} | "
            f"{fus['test_tp']} | {fus['test_fp']} | {fus['test_tn']} | {fus['test_fn']} |"
        )
    lines.extend([
        "",
        "## FN Overlap",
        "",
        "| Stage | ADP FN | AHL FN | Common FN | AHL rescues ADP FN | ADP rescues AHL FN | Fusion FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in overlap_rows:
        lines.append(
            f"| {row['stage']} | {row['adp_fn']} | {row['ahl_fn']} | {row['common_fn']} | "
            f"{row['ahl_recovers_adp_fn']} | {row['adp_recovers_ahl_fn']} | {row['fusion_fn']} |"
        )
    lines.extend([
        "",
        "## Best Alpha",
        "",
        "| Stage | Best Alpha | Fusion Threshold | Calib F1 | Calib Recall | Calib Precision | Test F1 | Test Recall | Test Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in alpha_rows:
        lines.append(
            f"| {row['stage']} | {format_float(row['alpha'])} | {format_float(row['threshold'])} | "
            f"{format_float(row['calib_f1'])} | {format_float(row['calib_recall'])} | {format_float(row['calib_precision'])} | "
            f"{format_float(row['test_f1'])} | {format_float(row['test_recall'])} | {format_float(row['test_precision'])} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- `AHL rescues ADP FN` = defects missed by ADP but recovered by AHL.",
        "- `ADP rescues AHL FN` = defects missed by AHL but recovered by ADP.",
        "- `fusion` threshold is chosen on calibration best-F1 after robust normalization.",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_root / "metrics_summary.csv"
    metrics_md = args.output_root / "metrics_summary.md"
    alpha_csv = args.output_root / "alpha_probe.csv"
    overlap_csv = args.output_root / "fn_overlap.csv"
    if not args.force:
        existing = [str(p) for p in [metrics_csv, metrics_md, alpha_csv, overlap_csv] if p.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {existing}")

    baselines = load_baselines()
    metrics_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, int]] = []
    alpha_probe_rows: List[Dict[str, float]] = []
    fusion_best_rows: List[Dict[str, float]] = []

    for stage in STAGES:
        adp = load_scores(METHODS["ADP-only-DINO"] / stage / "metrics" / "scores.csv")
        ahl = load_scores(METHODS["AHL-DINO"] / stage / "metrics" / "scores.csv")
        adp_calib, adp_test = split = (pack_split(adp["calib"]), pack_split(adp["test"]))
        ahl_calib, ahl_test = (pack_split(ahl["calib"]), pack_split(ahl["test"]))
        align_split(adp["calib"], ahl["calib"])
        align_split(adp["test"], ahl["test"])

        adp_center, adp_scale = robust_norm_params(adp_calib.scores[adp_calib.labels == 0])
        ahl_center, ahl_scale = robust_norm_params(ahl_calib.scores[ahl_calib.labels == 0])
        adp_calib_norm = robust_normalize(adp_calib.scores, adp_center, adp_scale)
        adp_test_norm = robust_normalize(adp_test.scores, adp_center, adp_scale)
        ahl_calib_norm = robust_normalize(ahl_calib.scores, ahl_center, ahl_scale)
        ahl_test_norm = robust_normalize(ahl_test.scores, ahl_center, ahl_scale)

        baseline_adp = baselines[("ADP-only-DINO", stage)]
        baseline_ahl = baselines[("AHL-DINO", stage)]
        adp_threshold = float(baseline_adp["threshold"])
        ahl_threshold = float(baseline_ahl["threshold"])
        adp_test_metric = confusion(adp_test.labels, adp_test.scores, adp_threshold)
        ahl_test_metric = confusion(ahl_test.labels, ahl_test.scores, ahl_threshold)
        adp_auc_roc, adp_auc_pr = aucs(adp_test.labels, adp_test.scores)
        ahl_auc_roc, ahl_auc_pr = aucs(ahl_test.labels, ahl_test.scores)

        adp_fn_ids = {
            sid
            for sid, label, score in zip(adp_test.ids, adp_test.labels, adp_test.scores)
            if label == 1 and score < adp_threshold
        }
        ahl_fn_ids = {
            sid
            for sid, label, score in zip(ahl_test.ids, ahl_test.labels, ahl_test.scores)
            if label == 1 and score < ahl_threshold
        }
        common_fn = adp_fn_ids & ahl_fn_ids
        fusion_rows, best_fusion = choose_fusion_alpha(
            adp_calib_norm,
            ahl_calib_norm,
            adp_test_norm,
            ahl_test_norm,
            adp_calib.labels,
            adp_test.labels,
        )
        for row in fusion_rows:
            alpha_probe_rows.append({
                "stage": stage,
                **row,
            })
        fusion_best_rows.append({
            "stage": stage,
            **best_fusion,
        })
        overlap_rows.append({
            "stage": stage,
            "adp_fn": len(adp_fn_ids),
            "ahl_fn": len(ahl_fn_ids),
            "common_fn": len(common_fn),
            "ahl_recovers_adp_fn": len(adp_fn_ids - ahl_fn_ids),
            "adp_recovers_ahl_fn": len(ahl_fn_ids - adp_fn_ids),
            "fusion_fn": int(best_fusion["test_fn"]),
        })

        metrics_rows.extend([
            {
                "method": "ADP-only-DINO",
                "stage": stage,
                "status": "ok",
                "alpha": "",
                "threshold": adp_threshold,
                "precision": adp_test_metric["precision"],
                "recall": adp_test_metric["recall"],
                "f1": adp_test_metric["f1"],
                "accuracy": adp_test_metric["accuracy"],
                "auc_roc": adp_auc_roc,
                "auc_pr": adp_auc_pr,
                "tp": adp_test_metric["tp"],
                "fp": adp_test_metric["fp"],
                "tn": adp_test_metric["tn"],
                "fn": adp_test_metric["fn"],
            },
            {
                "method": "AHL-DINO",
                "stage": stage,
                "status": "ok",
                "alpha": "",
                "threshold": ahl_threshold,
                "precision": ahl_test_metric["precision"],
                "recall": ahl_test_metric["recall"],
                "f1": ahl_test_metric["f1"],
                "accuracy": ahl_test_metric["accuracy"],
                "auc_roc": ahl_auc_roc,
                "auc_pr": ahl_auc_pr,
                "tp": ahl_test_metric["tp"],
                "fp": ahl_test_metric["fp"],
                "tn": ahl_test_metric["tn"],
                "fn": ahl_test_metric["fn"],
            },
            {
                "method": "fusion",
                "stage": stage,
                "status": "ok",
                "alpha": best_fusion["alpha"],
                "threshold": best_fusion["threshold"],
                "precision": best_fusion["test_precision"],
                "recall": best_fusion["test_recall"],
                "f1": best_fusion["test_f1"],
                "accuracy": best_fusion["test_accuracy"],
                "auc_roc": best_fusion["auc_roc"],
                "auc_pr": best_fusion["auc_pr"],
                "tp": best_fusion["test_tp"],
                "fp": best_fusion["test_fp"],
                "tn": best_fusion["test_tn"],
                "fn": best_fusion["test_fn"],
            },
        ])

    metrics_fields = [
        "method", "stage", "status", "alpha", "threshold", "precision", "recall", "f1", "accuracy",
        "auc_roc", "auc_pr", "tp", "fp", "tn", "fn",
    ]
    alpha_fields = [
        "stage", "alpha", "threshold", "calib_precision", "calib_recall", "calib_f1", "calib_accuracy",
        "calib_fpr", "test_precision", "test_recall", "test_f1", "test_accuracy", "test_fpr",
        "test_tp", "test_fp", "test_tn", "test_fn", "auc_roc", "auc_pr",
    ]
    overlap_fields = [
        "stage", "adp_fn", "ahl_fn", "common_fn", "ahl_recovers_adp_fn", "adp_recovers_ahl_fn", "fusion_fn",
    ]

    write_csv(metrics_csv, metrics_fields, metrics_rows)
    write_csv(alpha_csv, alpha_fields, alpha_probe_rows)
    write_csv(overlap_csv, overlap_fields, overlap_rows)
    metrics_md.write_text(
        build_markdown(baselines, fusion_best_rows, overlap_rows, fusion_best_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "output_root": str(args.output_root),
        "metrics_rows": len(metrics_rows),
        "alpha_rows": len(alpha_probe_rows),
        "overlap_rows": len(overlap_rows),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
