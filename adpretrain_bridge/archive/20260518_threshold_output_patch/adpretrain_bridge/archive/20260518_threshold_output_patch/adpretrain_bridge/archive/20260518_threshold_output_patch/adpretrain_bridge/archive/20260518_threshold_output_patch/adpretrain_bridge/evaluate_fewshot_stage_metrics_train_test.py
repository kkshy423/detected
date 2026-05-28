#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one train/test-only AHL stage with ADPretrain-style best-F1 metrics."""
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fewshot_qm_xiepai_common import (
    AHL_ROOT,
    ALIAS_CLASS,
    STAGE_SPECS,
    config_dir,
    metrics_dir,
    read_json,
    read_result_rows,
    stage_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate qm_xiepai train/test-only few-shot AHL stage metrics.")
    parser.add_argument("--stage", required=True, choices=list(STAGE_SPECS.keys()))
    parser.add_argument("--output-root", type=Path, required=True)
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


def adpretrain_best_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2.0 * precisions * recalls) / (precisions + recalls + 1e-12)
    if len(thresholds) == 0:
        threshold = float(scores.max()) if len(scores) else 0.0
        row = metric_at(labels, scores, threshold)
        row["source_precision_recall_curve_f1"] = float(f1_scores[0]) if len(f1_scores) else row["f1"]
        return row
    valid_f1 = f1_scores[:-1]
    best_index = int(np.nanargmax(valid_f1))
    threshold = float(thresholds[best_index])
    row = metric_at(labels, scores, threshold)
    row["source_precision_recall_curve_f1"] = float(valid_f1[best_index])
    return row


def normal_percentile_thresholds(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    normal_scores = scores[labels == 0]
    out = {}
    if len(normal_scores) == 0:
        return out
    for percentile in [95.0, 97.5, 99.0]:
        key = "test_normal_p" + str(percentile).replace(".", "_")
        out[key] = metric_at(labels, scores, float(np.percentile(normal_scores, percentile)))
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
    except Exception as exc:
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

    test_rows = [r for r in score_rows if r["role"].startswith("test_")]
    test_labels = np.asarray([r["label"] for r in test_rows], dtype=np.int64)
    test_scores = np.asarray([r["score"] for r in test_rows], dtype=np.float64)

    if len(test_labels) != 200:
        raise ValueError(f"Expected fixed test size 200, got {len(test_labels)}")
    if len(set(test_labels.tolist())) != 2:
        raise ValueError("Fixed test set must contain both normal and anomaly labels")

    threshold_results = {"adpretrain_eval_best_f1": adpretrain_best_f1(test_labels, test_scores)}
    threshold_results.update(normal_percentile_thresholds(test_labels, test_scores))
    primary_policy = "adpretrain_eval_best_f1"
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
        "eval_policy": "ADPretrain-style image-level best-F1 on fixed test scores",
        "primary_policy": primary_policy,
        "primary": primary,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "test_counts": {"normal": int((test_labels == 0).sum()), "anomaly": int((test_labels == 1).sum())},
        "train_counts": {
            "normal": int(manifest["counts"]["train_normal"]),
            "anomaly": int(manifest["counts"]["train_anomaly"]),
        },
        "threshold_results": threshold_results,
        "inference_time_ms": inference_time_ms,
        "inference_time_note": inference_note,
        "score_file": str(metrics_dir(args.output_root, args.stage) / "scores.csv"),
        "result_file": str(result_file),
        "ahl_subdir": args.ahl_subdir,
        "manifest_file": str(manifest_path),
        "notes": [
            "AHL original code reports anomaly scores plus AUC-ROC/AUC-PR and does not define a deployable classification threshold.",
            "ADPretrain downstream metrics compute image F1 from the best point on precision_recall_curve(labels, scores).",
            "This evaluation follows that ADPretrain-style metric for parity with the no-AHL baseline; it is an evaluation-set oracle threshold, not a deployable production threshold.",
        ],
    }

    md = [
        f"# {args.stage} Train/Test Metrics",
        "",
        f"Primary policy: `{primary_policy}`",
        "",
        "| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {fmt(primary['precision'])} | {fmt(primary['recall'])} | {fmt(primary['f1'])} | {fmt(primary['accuracy'])} | {fmt(auc_roc)} | {fmt(auc_pr)} | {fmt(inference_time_ms)} |",
        "",
        "## Threshold Results",
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
        "- No calibration samples are used or present.",
        "- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.",
        "- This is an evaluation metric, not a deployable threshold calibration mechanism.",
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
