#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare an AHL-readable full original single-class root from the plain cache."""
import argparse
from datetime import datetime
from pathlib import Path

from fewshot_qm_xiepai_common import (
    ALIAS_CLASS,
    AUG_ALIAS_PAIRS,
    CACHE_BASE,
    config_dir,
    ensure_empty_dir,
    ensure_symlink,
    list_supported_entries,
    source_feature_path,
    source_image_path,
    stage_feature_rel,
    write_json,
)

DEFAULT_STAGE_ROOT = CACHE_BASE / "full_original_plain_official_img_angle"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare full original plain-cache AHL root.")
    parser.add_argument("--cache-root", type=Path, default=CACHE_BASE / "plain_official_img_angle")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--reuse-stage-root", action="store_true")
    return parser.parse_args()


def rels(cache_class_root: Path, rel_dir: str):
    base = cache_class_root / rel_dir
    return [str(p.relative_to(cache_class_root)).replace("\\", "/") for p in list_supported_entries(base)]


def link_rel(cache_class_root: Path, stage_class_root: Path, source_rel: str):
    image_src = source_image_path(cache_class_root, source_rel)
    image_dst = stage_class_root / source_rel
    ensure_symlink(image_src, image_dst)
    for scale in ["feature", "feature_scale"]:
        feature_src = source_feature_path(cache_class_root, scale, source_rel)
        feature_dst = stage_class_root / stage_feature_rel(scale, source_rel)
        ensure_symlink(feature_src, feature_dst)
    return {"source_rel": source_rel, "stage_rel": source_rel}


def main():
    args = parse_args()
    cache_class_root = args.cache_root / args.alias
    stage_class_root = args.stage_root / args.alias
    if stage_class_root.exists() and not args.reuse_stage_root:
        raise FileExistsError(f"Stage root exists; pass --reuse-stage-root only after verifying it: {stage_class_root}")
    for rel_dir in [
        "train/good",
        "test/good",
        "test/defect",
        "feature/train/good",
        "feature/test/good",
        "feature/test/defect",
        "feature_scale/train/good",
        "feature_scale/test/good",
        "feature_scale/test/defect",
    ]:
        ensure_empty_dir(stage_class_root / rel_dir)

    train_normal = rels(cache_class_root, "train/good")
    test_normal = rels(cache_class_root, "test/good")
    test_anomaly = rels(cache_class_root, "test/defect")
    mappings = []
    for rel in train_normal + test_normal + test_anomaly:
        mappings.append(link_rel(cache_class_root, stage_class_root, rel))
    for alias_name, source_name in AUG_ALIAS_PAIRS:
        ensure_symlink(stage_class_root / source_name, stage_class_root / alias_name)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alias": args.alias,
        "stage_dataset_root": str(args.stage_root),
        "stage_class_root": str(stage_class_root),
        "source_cache_root": str(args.cache_root),
        "split_policy": "full_original_plain_cache_direct_ahl_split",
        "counts": {
            "train_normal": len(train_normal),
            "all_test_normal": len(test_normal),
            "all_test_anomaly": len(test_anomaly),
            "nAnomaly": 139,
            "expected_eval_normal": len(test_normal),
            "expected_eval_anomaly": len(test_anomaly) - 139,
            "expected_eval_total": len(test_normal) + len(test_anomaly) - 139,
        },
        "note": "AHL original split is used: nAnomaly=139 is sampled from test/defect for training and the remaining 60 anomalies are evaluated with all 140 test/good normals.",
        "mappings": mappings,
    }
    write_json(stage_class_root / "stage_manifest.json", manifest)
    write_json(config_dir(args.output_root) / "full_original_stage_manifest.json", manifest)
    print(f"Prepared full original root: {args.stage_root}")


if __name__ == "__main__":
    main()
