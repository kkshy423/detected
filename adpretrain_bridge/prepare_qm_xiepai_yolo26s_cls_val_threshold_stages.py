#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare YOLO classification stage datasets for the 6_1 validation-threshold split."""

import argparse
import os
from pathlib import Path

from fewshot_qm_xiepai_common import ALIAS_CLASS, read_lines, write_json


DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_6_1_val_threshold")
DEFAULT_SOURCE_CLASS_ROOT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1/models__球面斜拍")
DEFAULT_OUTPUT_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_6_1_cls_val_threshold_20260517")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--source-class-root", type=Path, default=DEFAULT_SOURCE_CLASS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--stages", nargs="+", default=[f"S{i}" for i in range(9)])
    p.add_argument("--alias", default=ALIAS_CLASS)
    return p.parse_args()


def symlink_once(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            target = os.readlink(str(dst))
            resolved = (dst.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
            if resolved == src.resolve():
                return
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    os.symlink(str(src), str(dst))


def safe_name(role: str, rel: str) -> str:
    p = Path(rel)
    return f"{role}__{'__'.join(p.with_suffix('').parts)}{p.suffix.lower()}"


def link_group(root: Path, split: str, cls: str, role: str, rels, source_class_root: Path):
    rows = []
    for rel in rels:
        src = source_class_root / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = root / split / cls / safe_name(role, rel)
        symlink_once(src, dst)
        rows.append({"split": split, "class": cls, "role": role, "source_rel": rel, "path": str(dst)})
    return rows


def prepare_stage(args, stage: str):
    stage_root = args.output_root / stage
    if stage_root.exists():
        raise FileExistsError(f"Stage dataset already exists: {stage_root}")
    sroot = args.split_root / stage
    train_normal = read_lines(sroot / "train_normal.txt")
    train_anomaly = read_lines(sroot / "train_anomaly.txt")
    calib_normal = read_lines(sroot / "calib_normal.txt")
    calib_anomaly = read_lines(sroot / "calib_anomaly.txt")
    test_normal = read_lines(args.split_root / "test_normal.txt")
    test_anomaly = read_lines(args.split_root / "test_anomaly.txt")
    rows = []
    rows += link_group(stage_root, "train", "good", "train_normal", train_normal, args.source_class_root)
    rows += link_group(stage_root, "train", "defect", "train_anomaly", train_anomaly, args.source_class_root)
    rows += link_group(stage_root, "val", "good", "calib_normal", calib_normal, args.source_class_root)
    rows += link_group(stage_root, "val", "defect", "calib_anomaly", calib_anomaly, args.source_class_root)
    rows += link_group(stage_root, "test", "good", "test_normal", test_normal, args.source_class_root)
    rows += link_group(stage_root, "test", "defect", "test_anomaly", test_anomaly, args.source_class_root)
    data_yaml = stage_root / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {stage_root}",
            "train: train",
            "val: val",
            "test: test",
            "names:",
            "  0: defect",
            "  1: good",
            "",
        ]),
        encoding="utf-8",
    )
    manifest = {
        "stage": stage,
        "stage_root": str(stage_root),
        "data_yaml": str(data_yaml),
        "split_root": str(args.split_root),
        "source_class_root": str(args.source_class_root),
        "task": "YOLO classification binary good-vs-defect with validation-side threshold calibration",
        "counts": {
            "train_normal": len(train_normal),
            "train_anomaly": len(train_anomaly),
            "val_normal": len(calib_normal),
            "val_anomaly": len(calib_anomaly),
            "test_normal": len(test_normal),
            "test_anomaly": len(test_anomaly),
        },
        "rows": rows,
        "notes": [
            "Validation split is used for model validation and threshold calibration.",
            "Held-out test split is not used for threshold selection.",
            "S0 has no anomaly training sample and is expected to be skipped by supervised YOLO.",
        ],
    }
    write_json(stage_root / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = [prepare_stage(args, stage) for stage in args.stages]
    write_json(args.output_root / "summary_manifest.json", {"status": "ok", "stages": manifests})
    print({"status": "ok", "output_root": str(args.output_root), "stages": args.stages})


if __name__ == "__main__":
    main()
