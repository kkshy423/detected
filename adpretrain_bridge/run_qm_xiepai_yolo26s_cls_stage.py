#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate one YOLO26s-cls baseline stage for qm_xiepai."""

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fewshot_qm_xiepai_common import STAGE_SPECS, read_json, write_json


DEFAULT_DATA_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_cls_fewshot_train_test_20260512")
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/"
    "qm_xiepai_yolo26s_cls_train_test_20260512"
)
DEFAULT_MODEL = "yolo26s-cls.pt"
DEFAULT_YOLO_DEPS_DIR = Path("/gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260513_retry3_nopolars")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGE_SPECS.keys()))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def prepend_yolo_deps():
    deps_dir = Path(os.environ.get("YOLO_PYTHON_DEPS_DIR", str(DEFAULT_YOLO_DEPS_DIR)))
    deps_dir.mkdir(parents=True, exist_ok=True)
    deps_path = str(deps_dir)
    if deps_path in sys.path:
        sys.path.remove(deps_path)
    sys.path.insert(0, deps_path)
    return deps_dir


def purge_yolo_modules():
    for name in list(sys.modules):
        if name in {"cv2", "ultralytics"} or name.startswith("cv2.") or name.startswith("ultralytics."):
            del sys.modules[name]


def import_yolo_from_deps(deps_dir):
    prepend_yolo_deps()
    purge_yolo_modules()
    import cv2  # noqa: F401
    from ultralytics import YOLO  # noqa: F401

    deps_prefix = str(deps_dir.resolve())
    cv2_file = str(Path(cv2.__file__).resolve())
    import ultralytics
    ultralytics_file = str(Path(ultralytics.__file__).resolve())
    if not cv2_file.startswith(deps_prefix):
        raise ImportError(f"cv2 resolved outside headless deps: {cv2_file}")
    if not ultralytics_file.startswith(deps_prefix):
        raise ImportError(f"ultralytics resolved outside headless deps: {ultralytics_file}")
    patch_ultralytics_no_polars()


def patch_ultralytics_no_polars():
    from ultralytics.engine.trainer import BaseTrainer

    if getattr(BaseTrainer.read_results_csv, "_codex_no_polars_patch", False):
        return

    def read_results_csv_no_polars(self):
        """Read results.csv without importing polars.

        Some PBS images lack a CPU-compatible polars build. Ultralytics only
        needs this dictionary when serializing checkpoints, so stdlib csv is
        sufficient here.
        """
        path = Path(self.csv)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return {}
                out = {field: [] for field in reader.fieldnames}
                for row in reader:
                    for field in reader.fieldnames:
                        value = row.get(field, "")
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            pass
                        out[field].append(value)
                return out
        except Exception:
            return {}

    read_results_csv_no_polars._codex_no_polars_patch = True
    BaseTrainer.read_results_csv = read_results_csv_no_polars


def ensure_ultralytics():
    deps_dir = prepend_yolo_deps()
    first_error = None
    try:
        import_yolo_from_deps(deps_dir)
        return
    except Exception as exc:
        first_error = repr(exc)

    lock_root = Path("/ghome/huangjd/code/detected/adpretrain_bridge/.locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_root / "ultralytics_headless_install.lock"
    acquired = False
    while not acquired:
        try:
            lock_dir.mkdir()
            acquired = True
        except FileExistsError:
            time.sleep(10)
    try:
        try:
            import_yolo_from_deps(deps_dir)
            return
        except Exception:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(deps_dir),
                    "--no-deps",
                    "opencv-python-headless==4.13.0.92",
                    "ultralytics-thop==2.0.19",
                    "ultralytics==8.4.49",
                ],
                check=True,
            )
            prepend_yolo_deps()
            importlib.invalidate_caches()
            try:
                import_yolo_from_deps(deps_dir)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import headless YOLO dependencies from "
                    f"{deps_dir}. First import error: {first_error}; "
                    f"post-install error: {repr(exc)}"
                )
    finally:
        shutil.rmtree(str(lock_dir), ignore_errors=True)


def metric_at(labels, scores, threshold):
    preds = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def best_f1(labels, scores):
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


def list_images(folder):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() or p.is_symlink() if p.suffix.lower() in exts], key=lambda p: p.name)


def test_items(stage_root):
    items = []
    for class_name, label in (("good", 0), ("defect", 1)):
        for path in list_images(stage_root / "test" / class_name):
            items.append((path, label, class_name))
    return items


def defect_class_index(names):
    if isinstance(names, dict):
        for key, value in names.items():
            if str(value).lower() == "defect":
                return int(key)
    else:
        for idx, value in enumerate(names):
            if str(value).lower() == "defect":
                return idx
    raise ValueError(f"Could not find defect class in model names: {names}")


def evaluate(model_path, stage_root, metrics_dir, imgsz, device):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    defect_idx = defect_class_index(model.names)
    rows = []
    elapsed = []
    for path, label, class_name in test_items(stage_root):
        start = time.perf_counter()
        pred = model.predict(source=str(path), imgsz=imgsz, device=device, verbose=False)[0]
        elapsed.append((time.perf_counter() - start) * 1000.0)
        probs = pred.probs.data.detach().cpu().numpy()
        rows.append({
            "path": str(path),
            "class": class_name,
            "label": int(label),
            "score": float(probs[defect_idx]),
        })
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    primary = best_f1(labels, scores)
    metrics = {
        "status": "ok",
        "primary_policy": "adpretrain_eval_best_f1",
        "primary": primary,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "test_counts": {
            "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()),
        },
        "score_rule": "YOLO classification probability for the defect class",
        "defect_class_index": int(defect_idx),
        "model_names": {str(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else list(model.names),
        "inference_time_ms": float(np.mean(elapsed)) if elapsed else None,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class", "label", "score"])
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def main():
    args = parse_args()
    stage_root = args.data_root / args.stage
    output_stage = args.output_root / "stages" / args.stage
    metrics_dir = output_stage / "metrics"
    metrics_file = metrics_dir / "metrics.json"
    if metrics_file.exists():
        try:
            old_status = json.loads(metrics_file.read_text(encoding="utf-8")).get("status")
        except Exception:
            old_status = "unknown"
        if old_status == "ok":
            raise FileExistsError(f"Refusing to overwrite successful metrics: {metrics_file}")
    output_stage.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(stage_root / "manifest.json")
    counts = manifest["counts"]
    if counts["available_train_anomaly"] == 0:
        skipped = {
            "stage": args.stage,
            "status": "skipped",
            "reason": "YOLO binary supervised classification requires at least one anomaly sample; S0 has none.",
            "train_counts": {
                "normal": counts["available_train_normal"],
                "anomaly": counts["available_train_anomaly"],
            },
        }
        write_json(metrics_dir / "metrics.json", skipped)
        print(json.dumps(skipped, ensure_ascii=False), flush=True)
        return

    config = {
        "stage": args.stage,
        "method": "YOLO26s-cls supervised baseline",
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
        "train_counts": {
            "normal": counts["available_train_normal"],
            "anomaly": counts["available_train_anomaly"],
        },
        "fit_counts": {
            "train_normal": counts["fit_train_normal"],
            "train_anomaly": counts["fit_train_anomaly"],
            "val_normal": counts["val_normal"],
            "val_anomaly": counts["val_anomaly"],
        },
        "test_counts": {
            "normal": counts["test_normal"],
            "anomaly": counts["test_anomaly"],
        },
        "metric_alignment": "fixed 140 normal + 60 anomaly test; best-F1/AUC computed offline from defect probability",
    }
    write_json(output_stage / "run_config.json", config)

    try:
        ensure_ultralytics()
        from ultralytics import YOLO

        model = YOLO(args.model)
        train_project = output_stage / "ultralytics"
        train_result = model.train(
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
        save_dir = Path(getattr(train_result, "save_dir", train_project / "train"))
        best = save_dir / "weights" / "best.pt"
        if not best.exists():
            best = save_dir / "weights" / "last.pt"
        metrics = evaluate(best, stage_root, metrics_dir, args.imgsz, args.device)
        metrics.update({
            "stage": args.stage,
            "method": "YOLO26s-cls supervised baseline",
            "model": args.model,
            "weights": str(best),
            "train_save_dir": str(save_dir),
            "train_counts": config["train_counts"],
            "fit_counts": config["fit_counts"],
        })
        write_json(metrics_dir / "metrics.json", metrics)
        p = metrics["primary"]
        lines = [
            f"# YOLO26s-cls {args.stage}",
            "",
            "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | "
                f"{p['accuracy']:.4f} | {metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} | "
                f"{metrics['inference_time_ms']:.4f} |"
            ),
            "",
        ]
        (output_stage / "result.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"status": "ok", "stage": args.stage, "metrics": str(metrics_dir / "metrics.json")}, ensure_ascii=False), flush=True)
    except Exception as exc:
        failed = {
            "stage": args.stage,
            "status": "failed",
            "error": repr(exc),
            "config": config,
        }
        write_json(metrics_dir / "metrics.json", failed)
        raise


if __name__ == "__main__":
    main()
