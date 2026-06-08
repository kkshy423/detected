#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path


ROOTS = {
    "AHL-DINO": Path("output/20260529_ahl_dino_large_180_70_val49_stage_v1"),
    "ADP-only-DINO": Path("output/20260529_adpretrain_only_dino_large_180_70_val49_all_stage_p95_safe"),
    "YOLO26s-eval": Path("output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only"),
}
STAGES = [f"S{i}" for i in range(9)]
OUT_DIR = Path("summary/20260530_dino_large_val49")


def f4(v):
    if v is None:
        return ""
    return f"{float(v):.4f}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pick_auc(m, key):
    if key in m:
        return m.get(key)
    return (m.get("test_auc") or {}).get(key)


def counts(m):
    tc = m.get("test_counts") or {}
    cc = m.get("calibration_counts") or {}
    tr = m.get("train_counts") or {}
    return {
        "train_normal": tr.get("normal"),
        "train_anomaly": tr.get("anomaly"),
        "calib_normal": cc.get("normal"),
        "calib_anomaly": cc.get("anomaly"),
        "test_normal": tc.get("normal"),
        "test_anomaly": tc.get("anomaly"),
    }


def row_for(method, stage, path):
    if not path.exists():
        return {"method": method, "stage": stage, "status": "missing", "metrics_json": str(path)}
    m = read_json(path)
    row = {"method": method, "stage": stage, "status": m.get("status", "ok"), "metrics_json": str(path)}
    if row["status"] != "ok":
        row["reason"] = m.get("reason") or m.get("error")
        row.update(counts(m))
        return row
    pri = m.get("primary") or {}
    row.update({
        "policy": m.get("primary_policy"),
        "threshold": pri.get("threshold"),
        "precision": pri.get("precision"),
        "recall": pri.get("recall"),
        "f1": pri.get("f1"),
        "accuracy": pri.get("accuracy"),
        "tp": pri.get("tp"),
        "fp": pri.get("fp"),
        "tn": pri.get("tn"),
        "fn": pri.get("fn"),
        "auc_roc": pick_auc(m, "auc_roc"),
        "auc_pr": pick_auc(m, "auc_pr"),
        "time_adpretrain_feature_ms": m.get("time_adpretrain_feature_ms"),
        "time_ahl_process_ms": m.get("time_ahl_process_ms"),
        "time_total_ms": m.get("time_total_ms") or m.get("inference_time_ms"),
        "weights": m.get("weights"),
    })
    row.update(counts(m))
    return row


def main():
    rows = []
    for method, root in ROOTS.items():
        for stage in STAGES:
            rows.append(row_for(method, stage, root / "stages" / stage / "metrics" / "metrics.json"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "metrics_summary.csv"
    fields = [
        "method", "stage", "status", "policy", "threshold", "precision", "recall", "f1", "accuracy",
        "tp", "fp", "tn", "fn", "auc_roc", "auc_pr",
        "train_normal", "train_anomaly", "calib_normal", "calib_anomaly", "test_normal", "test_anomaly",
        "time_adpretrain_feature_ms", "time_ahl_process_ms", "time_total_ms", "weights", "reason", "metrics_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    md = [
        "# 20260530 DINO-large Val49 Results",
        "",
        "- split: `20260529_qm_xiepai_6_1_fixed_180_70_val49`",
        "- S7-S8 threshold: validation best-F1 under `fpr<=0.1` via `strategy_mild_stage_v2_1_safe`",
        "- ADPretrain backbone/projector: `dinov2-large` + `checkpoints/dino-large/checkpoints_img_norm.pth`",
        "",
        "| Method | Stage | Status | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN | Total ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            md.append(f"| {row['method']} | {row['stage']} | {row['status']} |  |  |  |  |  |  |  |  |  |  |  |")
            continue
        md.append(
            f"| {row['method']} | {row['stage']} | ok | {f4(row.get('precision'))} | {f4(row.get('recall'))} | "
            f"{f4(row.get('f1'))} | {f4(row.get('accuracy'))} | {f4(row.get('auc_roc'))} | {f4(row.get('auc_pr'))} | "
            f"{row.get('tp') if row.get('tp') is not None else ''} | {row.get('fp') if row.get('fp') is not None else ''} | "
            f"{row.get('tn') if row.get('tn') is not None else ''} | {row.get('fn') if row.get('fn') is not None else ''} | "
            f"{f4(row.get('time_total_ms'))} |"
        )
    md.extend(["", f"- CSV: `{csv_path}`", ""])
    md_path = OUT_DIR / "metrics_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "ok", "csv": str(csv_path), "md": str(md_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
