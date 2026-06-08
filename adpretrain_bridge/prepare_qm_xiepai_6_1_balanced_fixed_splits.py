#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare qm_xiepai 6_1 staged splits with fixed train/calib/test counts.

This split consumes all available normal samples and all available defect
samples from the 6_1 class folder, then creates cumulative staged training
sets plus fixed calibration/test sets:

S0..S8 train normal/anomaly follow the requested production accumulation table;
calibration is fixed at 100 normal + 40 anomaly; test is fixed at
180 normal + 79 anomaly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from fewshot_qm_xiepai_common import ALIAS_CLASS, natural_key, write_json, write_lines

SOURCE_BASE = Path("/gdata1/huangjd/xidun_mvtec_format_6_1")
SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/20260529_qm_xiepai_6_1_fixed_180_70_val49")
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
STAGE_COUNTS = {
    "S0": {"train_normal": 50, "train_anomaly": 0, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S1": {"train_normal": 100, "train_anomaly": 1, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S2": {"train_normal": 150, "train_anomaly": 3, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S3": {"train_normal": 200, "train_anomaly": 5, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S4": {"train_normal": 300, "train_anomaly": 10, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S5": {"train_normal": 400, "train_anomaly": 20, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S6": {"train_normal": 420, "train_anomaly": 40, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S7": {"train_normal": 420, "train_anomaly": 60, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
    "S8": {"train_normal": 420, "train_anomaly": 80, "calib_normal": 100, "calib_anomaly": 49, "test_normal": 180, "test_anomaly": 70},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-base", type=Path, default=SOURCE_BASE)
    p.add_argument("--source-class-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--alias", default=ALIAS_CLASS)
    return p.parse_args()


def resolve_source_class_root(args) -> Path:
    if args.source_class_root is not None:
        return args.source_class_root
    roots = sorted([p for p in args.source_base.glob("models__*") if p.is_dir()])
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one models__* folder under {args.source_base}, got {roots}")
    return roots[0]


def rel_images(folder: Path, base: Path) -> List[str]:
    return [
        str(p.relative_to(base)).replace(os.sep, "/")
        for p in sorted(folder.iterdir(), key=lambda x: natural_key(x.name))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def assert_prefix(prev: List[str], cur: List[str], name: str) -> None:
    if cur[: len(prev)] != prev:
        raise ValueError(f"{name} is not cumulative")


def assert_disjoint(groups: Dict[str, List[str]]) -> None:
    names = list(groups)
    for i, a in enumerate(names):
        sa = set(groups[a])
        if len(sa) != len(groups[a]):
            raise ValueError(f"{a} has duplicate entries")
        for b in names[i + 1:]:
            overlap = sa & set(groups[b])
            if overlap:
                raise ValueError(f"{a}/{b} overlap: {sorted(overlap)[:5]}")


def main() -> None:
    args = parse_args()
    source_class_root = resolve_source_class_root(args)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty split root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    normal_pool = (
        rel_images(source_class_root / "train/good", source_class_root)
        + rel_images(source_class_root / "val/good", source_class_root)
        + rel_images(source_class_root / "test/good", source_class_root)
    )
    anomaly_pool = (
        rel_images(source_class_root / "val/defect", source_class_root)
        + rel_images(source_class_root / "test/defect", source_class_root)
    )
    need_normal = 420 + 100 + 180
    need_anomaly = 80 + 49 + 70
    if len(normal_pool) < need_normal:
        raise ValueError(f"Need {need_normal} normal samples, got {len(normal_pool)}")
    if len(anomaly_pool) < need_anomaly:
        raise ValueError(f"Need {need_anomaly} anomaly samples, got {len(anomaly_pool)}")

    train_normal_pool = normal_pool[:420]
    calib_normal = normal_pool[420:520]
    test_normal = normal_pool[520:700]
    train_anomaly_pool = anomaly_pool[:80]
    calib_anomaly = anomaly_pool[80:129]
    test_anomaly = anomaly_pool[129:199]

    assert_disjoint({
        "train_normal_pool": train_normal_pool,
        "calib_normal": calib_normal,
        "test_normal": test_normal,
        "train_anomaly_pool": train_anomaly_pool,
        "calib_anomaly": calib_anomaly,
        "test_anomaly": test_anomaly,
    })

    write_lines(args.output_root / "train_normal_pool.txt", train_normal_pool)
    write_lines(args.output_root / "train_anomaly_pool.txt", train_anomaly_pool)
    write_lines(args.output_root / "calib_normal.txt", calib_normal)
    write_lines(args.output_root / "calib_anomaly.txt", calib_anomaly)
    write_lines(args.output_root / "test_normal.txt", test_normal)
    write_lines(args.output_root / "test_anomaly.txt", test_anomaly)

    prev_normal: List[str] = []
    prev_anomaly: List[str] = []
    stage_specs = {}
    for stage, spec in STAGE_COUNTS.items():
        train_normal = train_normal_pool[: spec["train_normal"]]
        train_anomaly = train_anomaly_pool[: spec["train_anomaly"]]
        assert_prefix(prev_normal, train_normal, f"{stage}/train_normal")
        assert_prefix(prev_anomaly, train_anomaly, f"{stage}/train_anomaly")
        prev_normal, prev_anomaly = train_normal, train_anomaly
        root = args.output_root / stage
        write_lines(root / "train_normal.txt", train_normal)
        write_lines(root / "train_anomaly.txt", train_anomaly)
        write_lines(root / "calib_normal.txt", calib_normal)
        write_lines(root / "calib_anomaly.txt", calib_anomaly)
        stage_specs[stage] = {
            "stage": stage,
            "train_normal": len(train_normal),
            "train_anomaly": len(train_anomaly),
            "calib_normal": len(calib_normal),
            "calib_anomaly": len(calib_anomaly),
            "test_normal": len(test_normal),
            "test_anomaly": len(test_anomaly),
        }

    config = {
        "status": "ok",
        "alias": args.alias,
        "source_class_root": str(source_class_root),
        "split_root": str(args.output_root),
        "split_policy": "20260529_fixed_counts_cumulative_train_fixed_calib_test_180_70_49",
        "stage_specs": stage_specs,
        "data_inventory": {
            "normal_total": len(normal_pool),
            "anomaly_total": len(anomaly_pool),
            "normal_source_order": ["train/good", "val/good", "test/good"],
            "anomaly_source_order": ["val/defect", "test/defect"],
            "train_normal_pool": len(train_normal_pool),
            "train_anomaly_pool": len(train_anomaly_pool),
            "calib_normal": len(calib_normal),
            "calib_anomaly": len(calib_anomaly),
            "test_normal": len(test_normal),
            "test_anomaly": len(test_anomaly),
        },
        "notes": [
            "Later stages are strict prefixes over earlier stage train lists.",
            "Calibration and test sets are fixed across all stages.",
            "This is a new full-pool redistribution split; it is comparable by trend to the previous 6_1 split but not sample-identical.",
            "Text label lists are stored under the split root for reproducibility.",
        ],
    }
    write_json(args.output_root / "split_config.json", config)
    write_json(args.output_root / "split_check.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
