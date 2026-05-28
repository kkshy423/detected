#!/usr/bin/env python3
"""Collect DRA, CLIP plain, and CLIP CHMM metrics into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


GROUPS = [
    ("dra_baseline", "DRA baseline"),
    ("clip_plain", "CLIP plain"),
    ("clip_chmm", "CLIP CHMM"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize three-way Xidun three-class experiment outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(rows, key):
    vals = [row.get(key) for row in rows or [] if isinstance(row.get(key), (int, float))]
    return float(mean(vals)) if vals else None


def latency_mean(rows, key):
    vals = [row.get(key) for row in rows or [] if isinstance(row.get(key), (int, float))]
    return float(mean(vals)) if vals else None


def main() -> None:
    args = parse_args()
    summary = []
    for group_id, group_label in GROUPS:
        metric_rows = read_json(args.output_root / "metrics" / group_id / "ahl_aligned_metrics_full.json")
        latency_rows = read_json(args.output_root / "metrics" / group_id / "adpretrain_latency.json")
        ahl_ms = mean_metric(metric_rows, "inference_time_ms")
        clip_ms = latency_mean(latency_rows, "clip_feature_ms")
        chmm_ms = latency_mean(latency_rows, "chmm_affine_ms")
        end_ms = None
        if ahl_ms is not None:
            end_ms = ahl_ms + (clip_ms or 0.0) + (chmm_ms or 0.0)
        summary.append({
            "group": group_id,
            "label": group_label,
            "class_count_ok": len([r for r in metric_rows or [] if r.get("status") == "ok"]),
            "accuracy": mean_metric(metric_rows, "classification_accuracy"),
            "precision": mean_metric(metric_rows, "precision"),
            "recall": mean_metric(metric_rows, "recall"),
            "f1": mean_metric(metric_rows, "f1"),
            "auc_roc": mean_metric(metric_rows, "auc_roc"),
            "auc_pr": mean_metric(metric_rows, "auc_pr"),
            "ahl_forward_ms": ahl_ms,
            "clip_feature_ms": clip_ms,
            "chmm_affine_ms": chmm_ms,
            "end_to_end_estimated_ms": end_ms,
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-Way Experiment Summary",
        "",
        "| Group | OK Classes | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | AHL ms | CLIP ms | CHMM ms | End-to-End ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {label} | {class_count_ok} | {accuracy} | {precision} | {recall} | {f1} | {auc_roc} | {auc_pr} | {ahl_forward_ms} | {clip_feature_ms} | {chmm_affine_ms} | {end_to_end_estimated_ms} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.output_json}")
    print(f"WROTE {args.output_md}")


if __name__ == "__main__":
    main()
