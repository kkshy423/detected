#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate qm_xiepai few-shot split counts, disjointness, and accumulation."""
import argparse
from pathlib import Path

from fewshot_qm_xiepai_common import (
    SOURCE_CLASS_ROOT,
    SPLIT_ROOT,
    STAGE_SPECS,
    ROLE_TO_FILE,
    ensure_no_duplicates,
    read_lines,
    stage_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check qm_xiepai few-shot split files.")
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
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

    previous_normal_available = set()
    previous_anomaly_available = set()
    stage_report = {}
    fixed_test = set(test_normal) | set(test_anomaly)

    for stage, spec in STAGE_SPECS.items():
        root = stage_dir(split_root, stage)
        data = {role: read_lines(root / filename) for role, filename in ROLE_TO_FILE.items()}
        for role, values in data.items():
            ensure_no_duplicates(f"{stage}/{role}", values)
            assert_exists(source_root, f"{stage}/{role}", values)

        assert_count(f"{stage}/train_normal", data["train_normal"], spec.train_normal)
        assert_count(f"{stage}/train_anomaly", data["train_anomaly"], spec.train_anomaly)
        assert_count(f"{stage}/calib_normal", data["calib_normal"], spec.calib_normal)
        assert_count(f"{stage}/calib_anomaly", data["calib_anomaly"], spec.calib_anomaly)

        train_all = set(data["train_normal"]) | set(data["train_anomaly"])
        calib_all = set(data["calib_normal"]) | set(data["calib_anomaly"])
        assert_disjoint(f"{stage}/train", train_all, f"{stage}/calib", calib_all)
        assert_disjoint(f"{stage}/train+calib", train_all | calib_all, "fixed_test", fixed_test)

        normal_available = set(data["train_normal"]) | set(data["calib_normal"])
        anomaly_available = set(data["train_anomaly"]) | set(data["calib_anomaly"])
        assert_count(f"{stage}/available_normal", normal_available, spec.available_normal)
        assert_count(f"{stage}/available_anomaly", anomaly_available, spec.available_anomaly)
        if not previous_normal_available.issubset(normal_available):
            raise AssertionError(f"{stage}: normal pool is not cumulative")
        if not previous_anomaly_available.issubset(anomaly_available):
            raise AssertionError(f"{stage}: anomaly pool is not cumulative")
        previous_normal_available = normal_available
        previous_anomaly_available = anomaly_available

        stage_report[stage] = {
            "train_normal": len(data["train_normal"]),
            "train_anomaly": len(data["train_anomaly"]),
            "calib_normal": len(data["calib_normal"]),
            "calib_anomaly": len(data["calib_anomaly"]),
            "available_normal": len(normal_available),
            "available_anomaly": len(anomaly_available),
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
    print(f"Split check OK: {output_json}")


if __name__ == "__main__":
    main()