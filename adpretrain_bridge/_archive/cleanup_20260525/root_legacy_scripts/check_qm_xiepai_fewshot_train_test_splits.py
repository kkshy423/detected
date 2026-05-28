#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate qm_xiepai train/test-only split counts, disjointness, and accumulation."""
import argparse
from pathlib import Path

from fewshot_qm_xiepai_common import (
    SOURCE_CLASS_ROOT,
    STAGE_SPECS,
    ensure_no_duplicates,
    read_lines,
    stage_dir,
    write_json,
)

DEFAULT_SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check qm_xiepai train/test-only split files.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def assert_count(name: str, values, expected: int) -> None:
    if len(values) != expected:
        raise AssertionError(f"{name}: expected {expected}, got {len(values)}")


def assert_disjoint(left_name: str, left, right_name: str, right) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise AssertionError(f"{left_name} overlaps {right_name}: {sorted(overlap)[:10]}")


def assert_exists(source_root: Path, name: str, values) -> None:
    missing = [rel for rel in values if not (source_root / rel).exists()]
    if missing:
        raise AssertionError(f"{name} has missing source files: {missing[:10]}")


def main() -> None:
    args = parse_args()
    split_root = args.split_root
    source_root = args.source_class_root
    if not (split_root / "split_config.json").exists():
        raise FileNotFoundError(split_root / "split_config.json")

    test_normal = read_lines(split_root / "test_normal.txt")
    test_anomaly = read_lines(split_root / "test_anomaly.txt")
    assert_count("test_normal", test_normal, 140)
    assert_count("test_anomaly", test_anomaly, 60)
    ensure_no_duplicates("test_normal", test_normal)
    ensure_no_duplicates("test_anomaly", test_anomaly)
    assert_disjoint("test_normal", test_normal, "test_anomaly", test_anomaly)
    assert_exists(source_root, "test_normal", test_normal)
    assert_exists(source_root, "test_anomaly", test_anomaly)

    fixed_test = set(test_normal) | set(test_anomaly)
    previous_normal = set()
    previous_anomaly = set()
    stage_report = {}
    for stage, spec in STAGE_SPECS.items():
        root = stage_dir(split_root, stage)
        train_normal = read_lines(root / "train_normal.txt")
        train_anomaly = read_lines(root / "train_anomaly.txt")
        ensure_no_duplicates(f"{stage}/train_normal", train_normal)
        ensure_no_duplicates(f"{stage}/train_anomaly", train_anomaly)
        assert_exists(source_root, f"{stage}/train_normal", train_normal)
        assert_exists(source_root, f"{stage}/train_anomaly", train_anomaly)
        assert_count(f"{stage}/train_normal", train_normal, spec.available_normal)
        assert_count(f"{stage}/train_anomaly", train_anomaly, spec.available_anomaly)
        assert_disjoint(f"{stage}/train", set(train_normal) | set(train_anomaly), "fixed_test", fixed_test)
        if not previous_normal.issubset(set(train_normal)):
            raise AssertionError(f"{stage}: normal train pool is not cumulative")
        if not previous_anomaly.issubset(set(train_anomaly)):
            raise AssertionError(f"{stage}: anomaly train pool is not cumulative")
        previous_normal = set(train_normal)
        previous_anomaly = set(train_anomaly)
        stage_report[stage] = {
            "train_normal": len(train_normal),
            "train_anomaly": len(train_anomaly),
        }

    report = {
        "status": "ok",
        "split_root": str(split_root),
        "source_class_root": str(source_root),
        "test_normal": len(test_normal),
        "test_anomaly": len(test_anomaly),
        "stages": stage_report,
    }
    output_json = args.output_json or split_root / "split_check.json"
    write_json(output_json, report)
    print(f"Train/test split check OK: {output_json}")


if __name__ == "__main__":
    main()
