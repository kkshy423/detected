#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one YOLO26s-cls stage and evaluate test metrics using validation-selected thresholds."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fewshot_qm_xiepai_common import read_json, write_json
from run_qm_xiepai_yolo26s_cls_stage import DEFAULT_MODEL, ensure_ultralytics
from threshold_policies import apply_thresholds


DEFAULT_DATA_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_6_1_cls_val_threshold_20260517")
DEFAULT_OUTPUT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=[f"S{i}" for i in range(9)])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--primary-policy", default="val_best_f1")
    return p.parse_args()


def list_images(folder: Path):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if (p.is_file() or p.is_symlink()) and p.suffix.lower() in exts], key=lambda p: p.name)


def split_items(stage_root: Path, split: str):
    items = []
    for cls, label in [("good", 0), ("defect", 1)]:
        for p in list_images(stage_root / split / cls):
            items.append((p, label, cls))
    return items


def defect_class_index(names):
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "defect":
                return int(k)
    else:
        for i, v in enumerate(names):
            if str(v).lower() == "defect":
                return i
    raise ValueError(f"Could not find defect class in model names: {names}")


def score_split(model, defect_idx: int, stage_root: Path, split: str, imgsz: int, device: str):
    rows = []
    elapsed = []
    for path, label, cls in split_items(stage_root, split):
        start = time.perf_counter()
        pred = model.predict(source=str(path), imgsz=imgsz, device=device, verbose=False)[0]
        elapsed.append((time.perf_counter() - start) * 1000.0)
        probs = pred.probs.data.detach().cpu().numpy()
        rows.append({"split": split, "path": str(path), "class": cls, "label": int(label), "score": float(probs[defect_idx])})
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    return rows, labels, scores, float(np.mean(elapsed)) if elapsed else None


def main():
    args = parse_args()
    stage_root = args.data_root / args.stage
    output_stage = args.output_root / "stages" / args.stage
    metrics_dir = output_stage / "metrics"
    if (metrics_dir / "metrics.json").exists():
        old = read_json(metrics_dir / "metrics.json")
        if old.get("status") == "ok":
            raise FileExistsError(f"Refusing to overwrite successful metrics: {metrics_dir / 'metrics.json'}")
    output_stage.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(stage_root / "manifest.json")
    counts = manifest["counts"]
    if counts["train_anomaly"] == 0:
        skipped = {"stage": args.stage, "status": "skipped", "reason": "YOLO supervised training requires at least one anomaly sample.", "train_counts": {"normal": counts["train_normal"], "anomaly": counts["train_anomaly"]}}
        write_json(metrics_dir / "metrics.json", skipped)
        print(json.dumps(skipped, ensure_ascii=False), flush=True)
        return
    config = {
        "stage": args.stage,
        "method": "YOLO26s-cls validation-threshold baseline",
        "model": args.model,
        "data_root": str(stage_root),
        "output_stage": str(output_stage),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "primary_policy": args.primary_policy,
        "counts": counts,
    }
    write_json(output_stage / "run_config.json", config)
    ensure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_project = output_stage / "ultralytics"
    result = model.train(
        data=str(stage_root),
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=str(train_project),
        name="train",
        exist_ok=False,
        verbose=True,
    )
    save_dir = Path(getattr(result, "save_dir", train_project / "train"))
    weights = save_dir / "weights" / "best.pt"
    if not weights.exists():
        weights = save_dir / "weights" / "last.pt"
    eval_model = YOLO(str(weights))
    defect_idx = defect_class_index(eval_model.names)
    val_rows, val_labels, val_scores, val_ms = score_split(eval_model, defect_idx, stage_root, "val", args.imgsz, args.device)
    test_rows, test_labels, test_scores, test_ms = score_split(eval_model, defect_idx, stage_root, "test", args.imgsz, args.device)
    threshold_results = apply_thresholds(val_labels, val_scores, test_labels, test_scores)
    primary = threshold_results[args.primary_policy]["test"]
    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "class", "label", "score"])
        writer.writeheader()
        writer.writerows(val_rows + test_rows)
    metrics = {
        "stage": args.stage,
        "status": "ok",
        "method": "YOLO26s-cls validation-threshold baseline",
        "primary_policy": args.primary_policy,
        "primary": primary,
        "threshold_results": threshold_results,
        "auc_roc": float(roc_auc_score(test_labels, test_scores)),
        "auc_pr": float(average_precision_score(test_labels, test_scores)),
        "calibration_auc_roc": float(roc_auc_score(val_labels, val_scores)),
        "calibration_auc_pr": float(average_precision_score(val_labels, val_scores)),
        "train_counts": {"normal": counts["train_normal"], "anomaly": counts["train_anomaly"]},
        "calibration_counts": {"normal": counts["val_normal"], "anomaly": counts["val_anomaly"]},
        "test_counts": {"normal": counts["test_normal"], "anomaly": counts["test_anomaly"]},
        "score_rule": "YOLO classification probability for the defect class",
        "defect_class_index": int(defect_idx),
        "model_names": {str(k): v for k, v in eval_model.names.items()} if isinstance(eval_model.names, dict) else list(eval_model.names),
        "inference_time_ms": test_ms,
        "calibration_inference_time_ms": val_ms,
        "weights": str(weights),
        "train_save_dir": str(save_dir),
        "score_file": str(metrics_dir / "scores.csv"),
    }
    write_json(metrics_dir / "metrics.json", metrics)
    p = metrics["primary"]
    lines = [
        f"# YOLO26s-cls Val-Threshold {args.stage}",
        "",
        f"Primary policy: `{args.primary_policy}`",
        "",
        "| P | R | F1 | Acc | AUC-ROC | AUC-PR |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {p['accuracy']:.4f} | {metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} |",
        "",
    ]
    (output_stage / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "stage": args.stage, "metrics": str(metrics_dir / "metrics.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
