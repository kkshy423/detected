#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and run one qm_xiepai 6_1 AHL stage with validation-threshold calibration."""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from fewshot_qm_xiepai_common import (
    ALIAS_CLASS,
    AUG_ALIAS_PAIRS,
    BRIDGE_ROOT,
    ahl_eval_order,
    config_dir,
    ensure_empty_dir,
    ensure_symlink,
    read_json,
    read_lines,
    role_stage_rel,
    source_feature_path,
    source_image_path,
    stage_output_dir,
    stage_feature_rel,
    write_json,
)


DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_6_1_val_threshold")
DEFAULT_CACHE_ROOT = Path("/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/plain_official_img_angle")
DEFAULT_STAGE_ROOT_BASE = Path("/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/stage_roots_plain_official_img_angle_val_threshold")
LEGACY_AHL_WRAPPER = BRIDGE_ROOT / "_archive/cleanup_20260525/root_legacy_scripts/run_ahl_no_cudnn.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=[f"S{i}" for i in range(9)])
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--stage-root-base", type=Path, default=DEFAULT_STAGE_ROOT_BASE)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--alias", default=ALIAS_CLASS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cluster-num", type=int, default=2)
    p.add_argument("--n-ref", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--ahl-subdir", default="ahl")
    p.add_argument("--reuse-stage-root", action="store_true")
    p.add_argument("--reuse-model-path", type=Path, default=None, help="existing trained AHL weights; when present, skip training and run eval-only")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def load_split(split_root: Path, stage: str):
    root = split_root / stage
    data = {}
    for role in ["train_normal", "train_anomaly", "calib_normal", "calib_anomaly"]:
        data[role] = read_lines(root / f"{role}.txt")
    data.update({
        "test_normal": read_lines(split_root / "test_normal.txt"),
        "test_anomaly": read_lines(split_root / "test_anomaly.txt"),
    })
    return data


def link_role(cache_class_root: Path, stage_class_root: Path, role: str, source_rels: List[str]) -> List[dict]:
    rows = []
    for source_rel in source_rels:
        stage_rel = role_stage_rel(role, source_rel)
        image_src = source_image_path(cache_class_root, source_rel)
        image_dst = stage_class_root / stage_rel
        if not image_src.exists():
            raise FileNotFoundError(image_src)
        ensure_symlink(image_src, image_dst)
        for scale in ["feature", "feature_scale"]:
            feature_src = source_feature_path(cache_class_root, scale, source_rel)
            feature_dst = stage_class_root / stage_feature_rel(scale, stage_rel)
            if not feature_src.exists():
                raise FileNotFoundError(feature_src)
            ensure_symlink(feature_src, feature_dst)
        rows.append({"role": role, "source_rel": source_rel, "stage_rel": stage_rel})
    return rows


def build_stage_root(args) -> Tuple[Path, Path]:
    cache_class_root = args.cache_root / args.alias
    if not cache_class_root.exists():
        raise FileNotFoundError(cache_class_root)
    split = load_split(args.split_root, args.stage)
    stage_dataset_root = args.stage_root_base / args.stage
    stage_class_root = stage_dataset_root / args.alias
    if stage_class_root.exists() and not args.reuse_stage_root:
        raise FileExistsError(f"Stage root exists; pass --reuse-stage-root only after verifying it: {stage_class_root}")
    for rel_dir in [
        "train/good", "test/good", "test/defect", "test/train_defect",
        "feature/train/good", "feature/test/good", "feature/test/defect", "feature/test/train_defect",
        "feature_scale/train/good", "feature_scale/test/good", "feature_scale/test/defect", "feature_scale/test/train_defect",
    ]:
        ensure_empty_dir(stage_class_root / rel_dir)
    mappings = []
    for role in ["train_normal", "train_anomaly", "calib_normal", "calib_anomaly", "test_normal", "test_anomaly"]:
        mappings.extend(link_role(cache_class_root, stage_class_root, role, split[role]))
    for alias_name, source_name in AUG_ALIAS_PAIRS:
        ensure_symlink(stage_class_root / source_name, stage_class_root / alias_name)
    eval_order = ahl_eval_order(stage_class_root, know_class="train_defect")
    role_by_stage_rel = {row["stage_rel"]: row["role"] for row in mappings}
    eval_roles = [role_by_stage_rel.get(rel, "unknown") for rel in eval_order]
    if any(role == "unknown" for role in eval_roles):
        raise RuntimeError("Eval order contains untracked files")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "alias": args.alias,
        "stage_dataset_root": str(stage_dataset_root),
        "stage_class_root": str(stage_class_root),
        "source_cache_root": str(args.cache_root),
        "split_root": str(args.split_root),
        "split_policy": "6_1_val_threshold",
        "know_class": "train_defect",
        "counts": {k: len(v) for k, v in split.items()},
        "eval_total": len(eval_order),
        "mappings": mappings,
        "eval_order": eval_order,
        "eval_roles": eval_roles,
        "note": "Calibration rows use val/good plus held-out val/defect; final test rows use test/good plus test/defect.",
    }
    write_json(stage_class_root / "stage_manifest.json", manifest)
    write_json(config_dir(args.output_root) / f"{args.stage}_stage_manifest.json", manifest)
    return stage_dataset_root, stage_class_root


def ahl_entrypoint() -> Path:
    current = BRIDGE_ROOT / "run_ahl_no_cudnn.py"
    if current.exists():
        return current
    if LEGACY_AHL_WRAPPER.exists():
        return LEGACY_AHL_WRAPPER
    raise FileNotFoundError(f"AHL wrapper not found: {current} or {LEGACY_AHL_WRAPPER}")


def run_ahl(args, stage_dataset_root: Path, train_anomaly: int) -> None:
    out_dir = stage_output_dir(args.output_root, args.stage)
    ahl_dir = out_dir / args.ahl_subdir
    ahl_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    result_file = ahl_dir / "result.txt"
    if result_file.exists():
        raise FileExistsError(f"Refusing to append to existing AHL result file: {result_file}")
    reuse_model_path = args.reuse_model_path
    train_skipped = False
    if reuse_model_path is not None:
        if not reuse_model_path.exists():
            raise FileNotFoundError(reuse_model_path)
        train_skipped = True
    cmd = [
        sys.executable, str(ahl_entrypoint()),
        "--dataset", "mvtecad",
        "--dataset_root", str(stage_dataset_root),
        "--experiment_dir", str(ahl_dir),
        "--classname", args.alias,
        "--feat_classname", args.alias,
        "--nAnomaly", str(train_anomaly),
        "--test_threshold", str(train_anomaly),
        "--know_class", "train_defect",
        "--cluster_num", str(args.cluster_num),
        "--nRef", str(args.n_ref),
        "--epochs", str(args.epochs),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--backbone", "resnet18",
        "--ramdn_seed", str(args.seed),
    ]
    if train_anomaly == 0:
        cmd.extend(["--auxiliary", "False"])
    if train_skipped:
        cmd.extend(["--eval_only", "--load_weights", str(reuse_model_path), "--preload_ref_eval"])
    config_path = config_dir(args.output_root) / f"{args.stage}_run_config.json"
    write_json(config_path, {
        "stage": args.stage,
        "command": cmd,
        "stage_dataset_root": str(stage_dataset_root),
        "ahl_dir": str(ahl_dir),
        "train_anomaly": train_anomaly,
        "execution_mode": "eval_only_reuse_model" if train_skipped else "train_then_eval",
        "train_skipped": train_skipped,
        "reuse_model_path": str(reuse_model_path) if train_skipped else None,
        "threshold_policy": "strategy_mild_stage_v2_1_safe primary; production_normal_p95 and val_best_f1 kept for comparison only",
    })
    print("RUN", " ".join(cmd), flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(BRIDGE_ROOT), check=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    feature_runtime_path = args.cache_root / args.alias / "feature_runtime.json"
    feature_runtime = read_json(feature_runtime_path) if feature_runtime_path.exists() else {}
    ahl_runtime_path = ahl_dir / "ahl_runtime.json"
    ahl_runtime = read_json(ahl_runtime_path) if ahl_runtime_path.exists() else {}
    time_adp = feature_runtime.get("time_adpretrain_feature_ms")
    time_ahl = ahl_runtime.get("time_ahl_process_ms")
    time_ahl_median = ahl_runtime.get("time_ahl_process_median_ms")
    time_ahl_p90 = ahl_runtime.get("time_ahl_process_p90_ms")
    time_total = None
    if time_adp is not None and time_ahl is not None:
        time_total = float(time_adp) + float(time_ahl)
    time_total_typical = None
    if time_adp is not None and time_ahl_median is not None:
        time_total_typical = float(time_adp) + float(time_ahl_median)
    time_total_p90 = None
    if time_adp is not None and time_ahl_p90 is not None:
        time_total_p90 = float(time_adp) + float(time_ahl_p90)
    write_json(out_dir / "runtime.json", {
        "stage": args.stage,
        "time_ms": float(elapsed_ms),
        "time_kind": "stage_wall_clock",
        "stage_wall_clock_ms": float(elapsed_ms),
        "execution_mode": "eval_only_reuse_model" if train_skipped else "train_then_eval",
        "train_skipped": train_skipped,
        "train_time_ms": 0.0 if train_skipped else None,
        "reuse_model_path": str(reuse_model_path) if train_skipped else None,
        "time_adpretrain_feature_ms": time_adp,
        "time_adpretrain_projected_feature_ms": feature_runtime.get("time_adpretrain_projected_feature_ms", time_adp),
        "time_adpretrain_encoder_ms": feature_runtime.get("time_adpretrain_encoder_ms"),
        "time_adpretrain_projector_ms": feature_runtime.get("time_adpretrain_projector_ms"),
        "time_ahl_process_ms": time_ahl,
        "time_ahl_process_median_ms": time_ahl_median,
        "time_ahl_process_p90_ms": time_ahl_p90,
        "time_total_ms": time_total,
        "time_total_typical_ms": time_total_typical,
        "time_total_p90_ms": time_total_p90,
        "time_total_note": "Online inference estimate sums projected ADPretrain feature generation and AHL processing. Feature file saving is excluded; eval-only AHL preloads reference batches to approximate deployment.",
        "ahl_subdir": args.ahl_subdir,
        "feature_runtime_file": str(feature_runtime_path) if feature_runtime_path.exists() else None,
        "feature_runtime": feature_runtime,
        "ahl_runtime_file": str(ahl_runtime_path) if ahl_runtime_path.exists() else None,
        "ahl_runtime": ahl_runtime,
        "command": cmd,
    })


def main() -> None:
    args = parse_args()
    stage_dataset_root, _ = build_stage_root(args)
    manifest = (config_dir(args.output_root) / f"{args.stage}_stage_manifest.json")
    train_anomaly = len(read_lines(args.split_root / args.stage / "train_anomaly.txt"))
    if args.prepare_only:
        print(f"Prepared val-threshold stage root: {stage_dataset_root}")
        return
    run_ahl(args, stage_dataset_root, train_anomaly)
    print(f"Stage finished: {args.stage}; manifest={manifest}")


if __name__ == "__main__":
    main()
