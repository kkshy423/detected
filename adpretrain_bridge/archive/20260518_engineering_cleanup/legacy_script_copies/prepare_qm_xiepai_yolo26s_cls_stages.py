#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare YOLO classification stage datasets from the qm_xiepai train/test split."""

import argparse
import json
import os
from pathlib import Path

from fewshot_qm_xiepai_common import SOURCE_CLASS_ROOT, STAGE_SPECS, read_lines, write_json


DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test")
DEFAULT_OUTPUT_ROOT = Path("/gdata1/huangjd/data/yolo_qm_xiepai_cls_fewshot_train_test_20260512")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stages", nargs="+", default=list(STAGE_SPECS.keys()))
    parser.add_argument("--val-normal-frac", type=float, default=0.2)
    parser.add_argument("--val-anomaly-frac", type=float, default=0.2)
    return parser.parse_args()


def symlink_force(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            target = os.readlink(str(dst))
            resolved = (dst.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
            if resolved == src.resolve():
                return
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    os.symlink(str(src), str(dst))


def safe_name(role, rel):
    p = Path(rel)
    return f"{role}__{'__'.join(p.with_suffix('').parts)}{p.suffix.lower()}"


def split_train_val(items, val_frac, allow_val_overlap=False):
    items = list(items)
    if not items:
        return [], []
    if len(items) == 1 and allow_val_overlap:
        return items, items
    val_n = int(round(len(items) * val_frac))
    val_n = max(1, val_n)
    val_n = min(val_n, len(items) - 1)
    val_items = items[-val_n:]
    train_items = items[:-val_n]
    return train_items, val_items


def link_group(root, split_name, class_name, role, rels, source_class_root):
    rows = []
    for rel in rels:
        src = source_class_root / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = root / split_name / class_name / safe_name(role, rel)
        symlink_force(src, dst)
        rows.append({"split": split_name, "class": class_name, "role": role, "source_rel": rel, "path": str(dst)})
    return rows


def prepare_stage(args, stage):
    stage_root = args.output_root / stage
    if stage_root.exists():
        raise FileExistsError(f"Stage dataset already exists: {stage_root}")
    split_stage_root = args.split_root / stage
    train_normal_all = read_lines(split_stage_root / "train_normal.txt")
    train_anomaly_all = read_lines(split_stage_root / "train_anomaly.txt")
    test_normal = read_lines(args.split_root / "test_normal.txt")
    test_anomaly = read_lines(args.split_root / "test_anomaly.txt")

    train_normal, val_normal = split_train_val(train_normal_all, args.val_normal_frac, allow_val_overlap=False)
    if len(train_anomaly_all) == 1:
        train_anomaly, val_anomaly = split_train_val(train_anomaly_all, args.val_anomaly_frac, allow_val_overlap=True)
        val_policy = "single_anomaly_reused_for_train_and_val"
    else:
        train_anomaly, val_anomaly = split_train_val(train_anomaly_all, args.val_anomaly_frac, allow_val_overlap=False)
        val_policy = "holdout_from_stage_train"

    rows = []
    rows += link_group(stage_root, "train", "good", "train_normal", train_normal, args.source_class_root)
    rows += link_group(stage_root, "train", "defect", "train_anomaly", train_anomaly, args.source_class_root)
    rows += link_group(stage_root, "val", "good", "val_normal", val_normal, args.source_class_root)
    rows += link_group(stage_root, "val", "defect", "val_anomaly", val_anomaly, args.source_class_root)
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
        "task": "YOLO classification binary good-vs-defect",
        "val_policy": val_policy,
        "counts": {
            "available_train_normal": len(train_normal_all),
            "available_train_anomaly": len(train_anomaly_all),
            "fit_train_normal": len(train_normal),
            "fit_train_anomaly": len(train_anomaly),
            "val_normal": len(val_normal),
            "val_anomaly": len(val_anomaly),
            "test_normal": len(test_normal),
            "test_anomaly": len(test_anomaly),
        },
        "rows": rows,
        "notes": [
            "S0 is prepared but is not suitable for binary supervised YOLO-cls training because it has no anomaly sample.",
            "The fixed test split is identical to the AHL train/test-only evaluation: 140 normal and 60 anomaly images.",
            "Validation is taken only from the stage training pool; fixed test images are not used for model selection.",
        ],
    }
    write_json(stage_root / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for stage in args.stages:
        manifests.append(prepare_stage(args, stage))
    write_json(args.output_root / "summary_manifest.json", {"status": "ok", "stages": manifests})
    print(json.dumps({"status": "ok", "output_root": str(args.output_root), "stages": args.stages}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
