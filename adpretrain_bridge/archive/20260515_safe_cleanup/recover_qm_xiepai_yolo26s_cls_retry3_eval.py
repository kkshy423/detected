#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover YOLO26s-cls metrics from retry3 best.pt weights without retraining."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from fewshot_qm_xiepai_common import STAGE_SPECS, read_json, write_json
from run_qm_xiepai_yolo26s_cls_stage import ensure_ultralytics, evaluate


DEFAULT_DATA_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_cls_fewshot_train_test_20260512")
DEFAULT_RETRY3_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_yolo26s_cls_train_test_20260513_retry3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_RETRY3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stages", nargs="*", default=list(STAGE_SPECS.keys()))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def write_result_md(path, title, metrics):
    primary = metrics["primary"]
    lines = [
        f"# {title}",
        "",
        "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {primary['precision']:.4f} | {primary['recall']:.4f} | "
            f"{primary['f1']:.4f} | {primary['accuracy']:.4f} | "
            f"{metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} | "
            f"{metrics['inference_time_ms']:.4f} |"
        ),
        "",
        f"- Weights: `{metrics['weights']}`",
        f"- Scores: `{metrics['score_file']}`",
        "- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def stage_counts(data_root, stage):
    manifest = read_json(data_root / stage / "manifest.json")
    counts = manifest["counts"]
    return {
        "train_normal": counts["available_train_normal"],
        "train_anomaly": counts["available_train_anomaly"],
        "fit_train_normal": counts["fit_train_normal"],
        "fit_train_anomaly": counts["fit_train_anomaly"],
        "val_normal": counts["val_normal"],
        "val_anomaly": counts["val_anomaly"],
        "test_normal": counts["test_normal"],
        "test_anomaly": counts["test_anomaly"],
    }


def best_weight(source_output_root, stage):
    stage_root = source_output_root / "stages" / stage / "ultralytics" / "train" / "weights"
    best = stage_root / "best.pt"
    if best.exists():
        return best
    last = stage_root / "last.pt"
    if last.exists():
        return last
    return None


def write_summary(output_root, rows):
    metrics_root = output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    write_json(metrics_root / "yolo26s_cls_recovered_summary.json", {"status": "ok", "rows": rows})

    csv_path = metrics_root / "yolo26s_cls_recovered_summary.csv"
    fieldnames = [
        "stage",
        "status",
        "train_normal",
        "train_anomaly",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "auc_roc",
        "auc_pr",
        "inference_time_ms",
        "weights",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# YOLO26s-cls Retry3 Recovered Eval",
        "",
        "| Stage | Train N/A | Status | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train = f"{row.get('train_normal', '-')}/{row.get('train_anomaly', '-')}"
        if row.get("status") == "ok":
            lines.append(
                f"| {row['stage']} | {train} | ok | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} | "
                f"{row['auc_roc']:.4f} | {row['auc_pr']:.4f} | {row['inference_time_ms']:.4f} |"
            )
        else:
            lines.append(f"| {row['stage']} | {train} | {row.get('status', 'unknown')} |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "This recovery evaluates existing retry3 `best.pt` weights only. No YOLO retraining is performed.",
            "",
        ]
    )
    (metrics_root / "yolo26s_cls_recovered_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "result.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root.exists() and not args.allow_existing:
        existing = [path.name for path in args.output_root.iterdir() if path.name != "logs"]
        if existing:
            raise FileExistsError(f"Refusing to reuse existing output root with content: {args.output_root}")

    (args.output_root / "config").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "YOLO26s-cls retry3 recovered offline eval",
        "data_root": str(args.data_root),
        "source_output_root": str(args.source_output_root),
        "output_root": str(args.output_root),
        "stages": args.stages,
        "imgsz": args.imgsz,
        "device": args.device,
        "note": "No retraining. Existing retry3 best.pt weights are evaluated on the fixed 140 normal + 60 anomaly test split.",
    }
    write_json(args.output_root / "config" / "recovery_config.json", config)

    ensure_ultralytics()

    rows = []
    for stage in args.stages:
        if stage not in STAGE_SPECS:
            raise ValueError(f"Unknown stage: {stage}")
        output_stage = args.output_root / "stages" / stage
        metrics_dir = output_stage / "metrics"
        output_stage.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        counts = stage_counts(args.data_root, stage)
        if counts["train_anomaly"] == 0:
            skipped = {
                "stage": stage,
                "status": "skipped",
                "reason": "No anomaly training sample exists for supervised binary YOLO-cls.",
                "train_counts": {
                    "normal": counts["train_normal"],
                    "anomaly": counts["train_anomaly"],
                },
            }
            write_json(metrics_dir / "metrics.json", skipped)
            rows.append({"stage": stage, "status": "skipped", **counts})
            continue

        weight = best_weight(args.source_output_root, stage)
        if weight is None:
            missing = {
                "stage": stage,
                "status": "missing_weights",
                "reason": "Neither best.pt nor last.pt exists in retry3 output.",
                "train_counts": {
                    "normal": counts["train_normal"],
                    "anomaly": counts["train_anomaly"],
                },
            }
            write_json(metrics_dir / "metrics.json", missing)
            rows.append({"stage": stage, "status": "missing_weights", **counts})
            continue

        metrics = evaluate(weight, args.data_root / stage, metrics_dir, args.imgsz, args.device)
        metrics.update(
            {
                "stage": stage,
                "method": "YOLO26s-cls supervised baseline recovered from retry3 best.pt",
                "weights": str(weight),
                "source_output_root": str(args.source_output_root),
                "data_root": str(args.data_root / stage),
                "train_counts": {
                    "normal": counts["train_normal"],
                    "anomaly": counts["train_anomaly"],
                },
                "fit_counts": {
                    "train_normal": counts["fit_train_normal"],
                    "train_anomaly": counts["fit_train_anomaly"],
                    "val_normal": counts["val_normal"],
                    "val_anomaly": counts["val_anomaly"],
                },
                "score_file": str(metrics_dir / "scores.csv"),
                "recovery_note": "Recovered by evaluating existing retry3 weights; no training was run.",
            }
        )
        write_json(metrics_dir / "metrics.json", metrics)
        write_result_md(output_stage / "result.md", f"YOLO26s-cls {stage} Recovered Eval", metrics)
        primary = metrics["primary"]
        rows.append(
            {
                "stage": stage,
                "status": "ok",
                "train_normal": counts["train_normal"],
                "train_anomaly": counts["train_anomaly"],
                "precision": primary["precision"],
                "recall": primary["recall"],
                "f1": primary["f1"],
                "accuracy": primary["accuracy"],
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "inference_time_ms": metrics["inference_time_ms"],
                "weights": str(weight),
            }
        )

    write_summary(args.output_root, rows)
    print(json.dumps({"status": "ok", "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
