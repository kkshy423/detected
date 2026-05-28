#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize validation-threshold stage results into compact CSV/Markdown."""

import argparse
import csv
from pathlib import Path

from fewshot_qm_xiepai_common import read_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ahl-output-root", type=Path, required=True)
    p.add_argument("--yolo-output-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260517"))
    return p.parse_args()


def fmt(x):
    return "" if x is None else f"{float(x):.4f}"


def load_method(method, root: Path):
    rows = []
    for i in range(9):
        stage = f"S{i}"
        p = root / "stages" / stage / "metrics" / "metrics.json"
        if not p.exists():
            rows.append({"method": method, "stage": stage, "status": "missing"})
            continue
        m = read_json(p)
        primary = m.get("primary") or {}
        rows.append({
            "method": method,
            "stage": stage,
            "status": m.get("status", "ok"),
            "train": f"{(m.get('train_counts') or {}).get('normal', '')}/{(m.get('train_counts') or {}).get('anomaly', '')}",
            "calib": f"{(m.get('calibration_counts') or {}).get('normal', '')}/{(m.get('calibration_counts') or {}).get('anomaly', '')}",
            "test": f"{(m.get('test_counts') or {}).get('normal', '')}/{(m.get('test_counts') or {}).get('anomaly', '')}",
            "precision": primary.get("precision"),
            "recall": primary.get("recall"),
            "f1": primary.get("f1"),
            "accuracy": primary.get("accuracy"),
            "auc_roc": m.get("auc_roc"),
            "auc_pr": m.get("auc_pr"),
            "primary_policy": m.get("primary_policy", ""),
            "metrics_json": str(p),
        })
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_method("AHL_PLAIN_VAL", args.ahl_output_root) + load_method("YOLO26S_VAL", args.yolo_output_root)
    fields = ["method", "stage", "status", "train", "calib", "test", "precision", "recall", "f1", "accuracy", "auc_roc", "auc_pr", "primary_policy", "metrics_json"]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# qm_xiepai 6_1 Validation-Threshold Summary",
        "",
        "Primary policy: validation-selected threshold applied to held-out test.",
        "",
        "| Method | Stage | Status | Train N/A | Calib N/A | Test N/A | P | R | F1 | Acc | AUC-ROC | AUC-PR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['stage']} | {r['status']} | {r.get('train','')} | {r.get('calib','')} | {r.get('test','')} | "
            f"{fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('f1'))} | {fmt(r.get('accuracy'))} | {fmt(r.get('auc_roc'))} | {fmt(r.get('auc_pr'))} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output_dir / "summary.json", {"status": "ok", "rows": rows})
    print(args.output_dir)


if __name__ == "__main__":
    main()
