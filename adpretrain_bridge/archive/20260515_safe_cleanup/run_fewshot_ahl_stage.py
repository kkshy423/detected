#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one stage-specific AHL input root and run AHL without editing AHL."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from fewshot_qm_xiepai_common import (
    ALIAS_CLASS,
    AUG_ALIAS_PAIRS,
    BRIDGE_ROOT,
    CACHE_BASE,
    FIRST_ROUND_STAGES,
    SPLIT_ROOT,
    STAGE_SPECS,
    ahl_eval_order,
    config_dir,
    ensure_empty_dir,
    ensure_symlink,
    load_stage_split,
    role_stage_rel,
    source_feature_path,
    source_image_path,
    stage_feature_rel,
    stage_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one qm_xiepai few-shot AHL stage.")
    parser.add_argument("--stage", required=True, choices=list(STAGE_SPECS.keys()))
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_BASE / "chmm_official_img_angle_draref")
    parser.add_argument("--stage-root-base", type=Path, default=CACHE_BASE / "stage_roots_chmm_official_img_angle_draref")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cluster-num", type=int, default=2)
    parser.add_argument("--n-ref", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ahl-subdir", default="ahl")
    parser.add_argument("--reuse-stage-root", action="store_true", help="Allow existing identical symlinks in the stage root.")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


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


def build_stage_root(args: argparse.Namespace) -> Tuple[Path, Path]:
    cache_class_root = args.cache_root / args.alias
    if not cache_class_root.exists():
        raise FileNotFoundError(cache_class_root)
    stage_dataset_root = args.stage_root_base / args.stage
    stage_class_root = stage_dataset_root / args.alias
    if stage_class_root.exists() and not args.reuse_stage_root:
        raise FileExistsError(f"Stage root exists; pass --reuse-stage-root only after verifying it: {stage_class_root}")

    split = load_stage_split(args.split_root, args.stage)
    spec = STAGE_SPECS[args.stage]
    mappings = []
    for rel_dir in [
        "train/good",
        "test/good",
        "test/defect",
        "test/train_defect",
        "feature/train/good",
        "feature/test/good",
        "feature/test/defect",
        "feature/test/train_defect",
        "feature_scale/train/good",
        "feature_scale/test/good",
        "feature_scale/test/defect",
        "feature_scale/test/train_defect",
    ]:
        ensure_empty_dir(stage_class_root / rel_dir)

    for role in ["train_normal", "train_anomaly", "calib_normal", "calib_anomaly", "test_normal", "test_anomaly"]:
        mappings.extend(link_role(cache_class_root, stage_class_root, role, split[role]))

    for alias_name, source_name in AUG_ALIAS_PAIRS:
        ensure_symlink(stage_class_root / source_name, stage_class_root / alias_name)

    eval_order = ahl_eval_order(stage_class_root, know_class="train_defect")
    role_by_stage_rel = {row["stage_rel"]: row["role"] for row in mappings}
    eval_roles = [role_by_stage_rel.get(rel, "unknown") for rel in eval_order]
    unknown = [rel for rel, role in zip(eval_order, eval_roles) if role == "unknown"]
    if unknown:
        raise RuntimeError(f"Eval order contains untracked files: {unknown[:10]}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "alias": args.alias,
        "stage_dataset_root": str(stage_dataset_root),
        "stage_class_root": str(stage_class_root),
        "source_cache_root": str(args.cache_root),
        "split_root": str(args.split_root),
        "know_class": "train_defect",
        "counts": {
            "train_normal": len(split["train_normal"]),
            "train_anomaly": len(split["train_anomaly"]),
            "calib_normal": len(split["calib_normal"]),
            "calib_anomaly": len(split["calib_anomaly"]),
            "test_normal": len(split["test_normal"]),
            "test_anomaly": len(split["test_anomaly"]),
            "eval_total": len(eval_order),
        },
        "stage_spec": spec.__dict__,
        "mappings": mappings,
        "eval_order": eval_order,
        "eval_roles": eval_roles,
        "note": "Calibration samples are included in AHL test scoring only for threshold calibration; fixed test metrics are computed from test_* roles only.",
    }
    write_json(stage_class_root / "stage_manifest.json", manifest)
    write_json(config_dir(args.output_root) / f"{args.stage}_stage_manifest.json", manifest)
    return stage_dataset_root, stage_class_root


def run_ahl(args: argparse.Namespace, stage_dataset_root: Path) -> None:
    spec = STAGE_SPECS[args.stage]
    out_dir = stage_output_dir(args.output_root, args.stage)
    ahl_dir = out_dir / args.ahl_subdir
    log_dir = out_dir / "logs"
    ahl_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_file = ahl_dir / "result.txt"
    if result_file.exists():
        raise FileExistsError(f"Refusing to append to existing AHL result file: {result_file}")

    cmd = [
        sys.executable,
        str(BRIDGE_ROOT / "run_ahl_no_cudnn.py"),
        "--dataset", "mvtecad",
        "--dataset_root", str(stage_dataset_root),
        "--experiment_dir", str(ahl_dir),
        "--classname", args.alias,
        "--feat_classname", args.alias,
        "--nAnomaly", str(spec.train_anomaly),
        "--test_threshold", str(spec.train_anomaly),
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
    auxiliary_enabled = spec.train_anomaly > 0
    if not auxiliary_enabled:
        cmd.extend(["--auxiliary", "False"])
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "command": cmd,
        "stage_dataset_root": str(stage_dataset_root),
        "ahl_dir": str(ahl_dir),
        "seed": args.seed,
        "cluster_num": args.cluster_num,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "ahl_subdir": args.ahl_subdir,
        "auxiliary_enabled": auxiliary_enabled,
        "feature_source_backbone": "clip-base",
        "ahl_backbone_arg": "resnet18",
        "bridge_runtime_notes": [
            "AHL argparse bool parsing is normalized by run_ahl_no_cudnn.py.",
            "AHL random.sample falls back to replacement only when k is larger than the population, for very small known-anomaly stages.",
            "S0 disables auxiliary reward because the original AHL auxiliary reward path has no real anomaly examples in that stage.",
        ],
    }
    write_json(config_dir(args.output_root) / f"{args.stage}_run_config.json", config)
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(BRIDGE_ROOT), check=True)


def main() -> None:
    args = parse_args()
    stage_dataset_root, _ = build_stage_root(args)
    if args.prepare_only:
        print(f"Prepared stage root: {stage_dataset_root}")
        return
    run_ahl(args, stage_dataset_root)
    print(f"Stage finished: {args.stage}")


if __name__ == "__main__":
    main()
