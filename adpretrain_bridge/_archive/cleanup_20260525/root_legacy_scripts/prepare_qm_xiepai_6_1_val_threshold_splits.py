#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare qm_xiepai 6_1 staged splits with validation-side threshold calibration."""

import argparse
import json
import os
from pathlib import Path
from typing import List

from fewshot_qm_xiepai_common import ALIAS_CLASS, natural_key, write_json, write_lines


SOURCE_CLASS_ROOT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1/models__球面斜拍")
SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_6_1_val_threshold")
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
STAGE_COUNTS = {
    "S0": (50, 0),
    "S1": (100, 1),
    "S2": (150, 3),
    "S3": (200, 5),
    "S4": (300, 10),
    "S5": (400, 20),
    "S6": (490, 40),
    "S7": (490, 60),
    "S8": (490, 80),
}
CALIB_ANOMALY_COUNT = 19


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    p.add_argument("--output-root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--alias", default=ALIAS_CLASS)
    p.add_argument("--calib-anomaly-count", type=int, default=CALIB_ANOMALY_COUNT)
    return p.parse_args()


def rel_images(folder: Path, base: Path) -> List[str]:
    return [
        str(p.relative_to(base)).replace(os.sep, "/")
        for p in sorted(folder.iterdir(), key=lambda x: natural_key(x.name))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def assert_prefix(prev: List[str], cur: List[str], name: str) -> None:
    if cur[: len(prev)] != prev:
        raise ValueError(f"{name} is not cumulative")


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty split root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    normal_pool = rel_images(args.source_class_root / "train/good", args.source_class_root)
    val_normal = rel_images(args.source_class_root / "val/good", args.source_class_root)
    val_defect = rel_images(args.source_class_root / "val/defect", args.source_class_root)
    test_normal = rel_images(args.source_class_root / "test/good", args.source_class_root)
    test_anomaly = rel_images(args.source_class_root / "test/defect", args.source_class_root)
    if len(val_defect) <= args.calib_anomaly_count:
        raise ValueError("Not enough val/defect samples to reserve calibration anomalies")
    train_anomaly_pool = val_defect[: -args.calib_anomaly_count]
    calib_anomaly = val_defect[-args.calib_anomaly_count :]

    write_lines(args.output_root / "production_normal_pool.txt", normal_pool)
    write_lines(args.output_root / "production_anomaly_pool_from_val_defect.txt", train_anomaly_pool)
    write_lines(args.output_root / "calib_normal.txt", val_normal)
    write_lines(args.output_root / "calib_anomaly.txt", calib_anomaly)
    write_lines(args.output_root / "test_normal.txt", test_normal)
    write_lines(args.output_root / "test_anomaly.txt", test_anomaly)

    prev_n: List[str] = []
    prev_a: List[str] = []
    stage_specs = {}
    for stage, (n_count, a_count) in STAGE_COUNTS.items():
        train_normal = normal_pool[:n_count]
        train_anomaly = train_anomaly_pool[:a_count]
        assert_prefix(prev_n, train_normal, f"{stage}/train_normal")
        assert_prefix(prev_a, train_anomaly, f"{stage}/train_anomaly")
        prev_n, prev_a = train_normal, train_anomaly
        root = args.output_root / stage
        write_lines(root / "train_normal.txt", train_normal)
        write_lines(root / "train_anomaly.txt", train_anomaly)
        write_lines(root / "calib_normal.txt", val_normal)
        write_lines(root / "calib_anomaly.txt", calib_anomaly)
        stage_specs[stage] = {
            "stage": stage,
            "train_normal": len(train_normal),
            "train_anomaly": len(train_anomaly),
            "calib_normal": len(val_normal),
            "calib_anomaly": len(calib_anomaly),
            "test_normal": len(test_normal),
            "test_anomaly": len(test_anomaly),
        }

    train_all = set(normal_pool) | set(train_anomaly_pool)
    calib_all = set(val_normal) | set(calib_anomaly)
    test_all = set(test_normal) | set(test_anomaly)
    if train_all & calib_all:
        raise ValueError("train/calib overlap")
    if train_all & test_all:
        raise ValueError("train/test overlap")
    if calib_all & test_all:
        raise ValueError("calib/test overlap")

    config = {
        "status": "ok",
        "alias": args.alias,
        "source_class_root": str(args.source_class_root),
        "split_root": str(args.output_root),
        "split_policy": "6_1_fixed_test_val_threshold_with_val_defect_split_for_train_and_calibration",
        "stage_specs": stage_specs,
        "data_inventory": {
            "train_good_pool": len(normal_pool),
            "val_good_as_calib_normal": len(val_normal),
            "val_defect_total": len(val_defect),
            "val_defect_train_pool": len(train_anomaly_pool),
            "val_defect_reserved_calib": len(calib_anomaly),
            "test_good": len(test_normal),
            "test_defect": len(test_anomaly),
        },
        "notes": [
            "train/good supplies cumulative normal training samples.",
            "val/defect is split into a cumulative known-anomaly training pool and a fixed anomaly calibration set.",
            "val/good is the fixed normal calibration set.",
            "test/good and test/defect are held out for final evaluation only.",
        ],
    }
    write_json(args.output_root / "split_config.json", config)
    write_json(args.output_root / "split_check.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
