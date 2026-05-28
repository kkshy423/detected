#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one few-shot AHL stage with calibration-only threshold selection."""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fewshot_qm_xiepai_common import (
    AHL_ROOT,
    ALIAS_CLASS,
    FIRST_ROUND_STAGES,
    SPLIT_ROOT,
    STAGE_SPECS,
    config_dir,
    metrics_dir,
    read_json,
    read_result_rows,
    stage_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate qm_xiepai few-shot AHL stage metrics.")
    parser.add_argument("--stage", required=True, choices=list(STAGE_SPECS.keys()))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--ahl-subdir", default="ahl")
    parser.add_argument("--benchmark-model-inference", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=80)
    return parser.parse_args()


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.4f}"
    return str(value)


def metric_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.asarray([0.0], dtype=np.float64)
    vals = np.unique(scores.astype(np.float64))
    eps = max(float(np.std(vals)) * 1e-6, 1e-9)
    return np.unique(np.concatenate([vals, vals + eps, vals - eps]))


def choose_calib_thresholds(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return out
    rows = [metric_at(labels, scores, float(t)) for t in threshold_candidates(scores)]
    out["calib_best_f1"] = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"], -r["threshold"]))
    out["calib_recall_priority"] = max(rows, key=lambda r: (r["recall"], r["f1"], r["precision"], -r["threshold"]))
    out["calib_precision_priority"] = max(rows, key=lambda r: (r["precision"], r["f1"], r["recall"], r["threshold"]))
    return out


def parse_setting_file(path: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if " : " not in line:
                continue
            key, value = line.rstrip("\n").split(" : ", 1)
            if value == "None":
                values[key] = None
            elif value in {"True", "False"}:
                values[key] = value == "True"
            else:
                try:
                    values[key] = int(value)
                except ValueError:
                    try:
                        values[key] = float(value)
                    except ValueError:
                        values[key] = value
    return values


def install_optional_import_stubs() -> None:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        import types
        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules.setdefault("matplotlib", matplotlib_stub)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

    try:
        import einops  # noqa: F401
    except ImportError:
        import types
        einops_stub = types.ModuleType("einops")

        def rearrange(x, pattern, **kwargs):
            normalized = " ".join(pattern.split())
            if normalized == "e k n-> k n e":
                return x.permute(1, 2, 0)
            raise NotImplementedError("Fallback einops.rearrange only supports: e k n-> k n e")

        einops_stub.rearrange = rearrange
        sys.modules.setdefault("einops", einops_stub)


def benchmark_inference(args: argparse.Namespace, manifest: dict) -> Tuple[Optional[float], str]:
    try:
        import torch
        torch.backends.cudnn.enabled = False
        install_optional_import_stubs()
        sys.path.insert(0, str(AHL_ROOT))
        from main import Trainer, setup_seed  # type: ignore

        ahl_dir = stage_output_dir(args.output_root, args.stage) / args.ahl_subdir
        setting_file = ahl_dir / "setting.txt"
        checkpoint = ahl_dir / f"{args.alias}_ctest.pkl"
        if not setting_file.exists() or not checkpoint.exists():
            return None, "missing setting.txt or checkpoint"
        settings = parse_setting_file(setting_file)
        settings["dataset_root"] = manifest["stage_dataset_root"]
        settings["experiment_dir"] = str(ahl_dir)
        settings["classname"] = args.alias
        settings["feat_classname"] = args.alias
        settings["AHL"] = False
        settings["auxiliary"] = False
        settings["workers"] = 0
        settings["no_cuda"] = args.device == "cpu"
        settings["cuda"] = args.device != "cpu" and torch.cuda.is_available()
        setup_seed(int(settings.get("ramdn_seed", 42)))
        old_cwd = os.getcwd()
        os.chdir(str(AHL_ROOT))
        try:
            trainer_args = SimpleNamespace(**settings)
            trainer = Trainer(trainer_args)
            trainer.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
            if trainer_args.cuda:
                trainer.model = trainer.model.cuda()
                trainer.criterion = trainer.criterion.cuda()
                trainer.mse_loss = trainer.mse_loss.cuda()
            trainer.model.eval()
            times: List[float] = []
            processed = 0
            with torch.no_grad():
                for sample in trainer.test_loader:
                    image, image_scale, target = sample["image"], sample["image_scale"], sample["label"]
                    if trainer_args.cuda:
                        image, image_scale, target = image.cuda(), image_scale.cuda(), target.cuda()
                    if int(settings.get("total_heads", 4)) == 4:
                        try:
                            ref_image = next(trainer.ref)["image"]
                            ref_image_scale = next(trainer.ref)["image_scale"]
                        except StopIteration:
                            trainer.ref = iter(trainer.ref_loader)
                            ref_image = next(trainer.ref)["image"]
                            ref_image_scale = next(trainer.ref)["image_scale"]
                        if trainer_args.cuda:
                            ref_image = ref_image.cuda()
                            ref_image_scale = ref_image_scale.cuda()
                        image = torch.cat([ref_image, image], dim=0)
                        image_scale = torch.cat([ref_image_scale, image_scale], dim=0)
                    if trainer_args.cuda:
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = trainer.model.forward(image=image, image_scale=image_scale, label=target, var=trainer.model.parameters())
                    if trainer_args.cuda:
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000.0
                    if processed >= args.warmup:
                        times.append(elapsed / max(int(target.numel()), 1))
                    processed += int(target.numel())
                    if args.max_images and processed >= args.max_images:
                        break
            if not times:
                return None, "no timing samples after warmup"
            return float(mean(times)), "feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O"
        finally:
            os.chdir(old_cwd)
    except Exception as exc:  # timing is reference metric; keep evaluation metrics usable.
        return None, f"benchmark failed: {exc}"


def main() -> None:
    args = parse_args()
    manifest_path = config_dir(args.output_root) / f"{args.stage}_stage_manifest.json"
    manifest = read_json(manifest_path)
    result_file = stage_output_dir(args.output_root, args.stage) / args.ahl_subdir / "result.txt"
    rows = read_result_rows(result_file)
    eval_order = manifest["eval_order"]
    mappings = {row["stage_rel"]: row for row in manifest["mappings"]}
    eval_size = len(eval_order)
    if len(rows) < eval_size:
        raise ValueError(f"Not enough result rows: {len(rows)} < {eval_size}")
    last_rows = rows[-eval_size:]

    score_rows = []
    for stage_rel, (label, score) in zip(eval_order, last_rows):
        meta = mappings[stage_rel]
        score_rows.append({
            "stage_rel": stage_rel,
            "source_rel": meta["source_rel"],
            "role": meta["role"],
            "label": int(label),
            "score": float(score),
        })

    calib_rows = [r for r in score_rows if r["role"].startswith("calib_")]
    test_rows = [r for r in score_rows if r["role"].startswith("test_")]
    calib_normal_scores = np.asarray([r["score"] for r in calib_rows if r["role"] == "calib_normal"], dtype=np.float64)
    calib_labels = np.asarray([r["label"] for r in calib_rows], dtype=np.int64)
    calib_scores = np.asarray([r["score"] for r in calib_rows], dtype=np.float64)
    test_labels = np.asarray([r["label"] for r in test_rows], dtype=np.int64)
    test_scores = np.asarray([r["score"] for r in test_rows], dtype=np.float64)

    if len(test_labels) != 200:
        raise ValueError(f"Expected fixed test size 200, got {len(test_labels)}")
    if len(set(test_labels.tolist())) != 2:
        raise ValueError("Fixed test set must contain both normal and anomaly labels")

    threshold_defs: Dict[str, Dict[str, object]] = {}
    if len(calib_normal_scores) == 0:
        raise ValueError("Calibration normal scores are required for thresholding")
    for percentile in [95.0, 97.5, 99.0]:
        key = "normal_p" + str(percentile).replace(".", "_")
        threshold_defs[key] = {
            "source": "calibration_normal_percentile",
            "percentile": percentile,
            "calibration_metrics": None,
            "threshold": float(np.percentile(calib_normal_scores, percentile)),
        }

    for key, calib_metric in choose_calib_thresholds(calib_labels, calib_scores).items():
        threshold_defs[key] = {
            "source": "calibration_best_search",
            "calibration_metrics": calib_metric,
            "threshold": float(calib_metric["threshold"]),
        }

    threshold_results = {}
    for key, definition in threshold_defs.items():
        threshold_results[key] = metric_at(test_labels, test_scores, float(definition["threshold"]))

    primary_policy = "calib_best_f1" if "calib_best_f1" in threshold_results else "normal_p97_5"
    primary = threshold_results[primary_policy]
    auc_roc = float(roc_auc_score(test_labels, test_scores))
    auc_pr = float(average_precision_score(test_labels, test_scores))
    inference_time_ms = None
    inference_note = "not benchmarked"
    if args.benchmark_model_inference:
        inference_time_ms, inference_note = benchmark_inference(args, manifest)

    out = {
        "stage": args.stage,
        "status": "ok",
        "primary_policy": primary_policy,
        "primary": primary,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "test_counts": {"normal": int((test_labels == 0).sum()), "anomaly": int((test_labels == 1).sum())},
        "calibration_counts": {
            "normal": sum(1 for r in calib_rows if r["label"] == 0),
            "anomaly": sum(1 for r in calib_rows if r["label"] == 1),
        },
        "threshold_definitions": threshold_defs,
        "threshold_results": threshold_results,
        "inference_time_ms": inference_time_ms,
        "inference_time_note": inference_note,
        "score_file": str(metrics_dir(args.output_root, args.stage) / "scores.csv"),
        "result_file": str(result_file),
        "ahl_subdir": args.ahl_subdir,
        "manifest_file": str(manifest_path),
    }

    md = [
        f"# {args.stage} Metrics",
        "",
        f"Primary threshold policy: `{primary_policy}`",
        "",
        "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {fmt(primary['precision'])} | {fmt(primary['recall'])} | {fmt(primary['f1'])} | {fmt(primary['accuracy'])} | {fmt(auc_roc)} | {fmt(auc_pr)} | {fmt(inference_time_ms)} |",
        "",
        "## Thresholds",
        "",
        "| Policy | Threshold | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in threshold_results.items():
        md.append(f"| {key} | {fmt(row['threshold'])} | {fmt(row['accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} |")
    md.extend([
        "",
        "## Scope",
        "",
        "- Thresholds are selected only from calibration scores.",
        "- Fixed test metrics use only `test_normal` and `test_anomaly` roles.",
        f"- Inference timing: {inference_note}",
    ])

    mdir = metrics_dir(args.output_root, args.stage)
    mdir.mkdir(parents=True, exist_ok=True)
    write_json(mdir / "metrics.json", out)
    (mdir / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (mdir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "label", "score", "stage_rel", "source_rel"])
        writer.writeheader()
        writer.writerows(score_rows)
    (stage_output_dir(args.output_root, args.stage) / "result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
