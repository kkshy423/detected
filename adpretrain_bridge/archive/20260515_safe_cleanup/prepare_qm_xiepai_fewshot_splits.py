#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fixed-test, staged few-shot splits for models_qiumianxiepai."""
import argparse
import random
from datetime import datetime
from pathlib import Path

from fewshot_qm_xiepai_common import (
    ALIAS_CLASS,
    SOURCE_CLASS_ROOT,
    SPLIT_ROOT,
    STAGE_SPECS,
    ensure_no_duplicates,
    rel_list,
    stage_dir,
    stage_spec_dict,
    write_json,
    write_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare qm_xiepai few-shot split files.")
    parser.add_argument("--source-class-root", type=Path, default=SOURCE_CLASS_ROOT)
    parser.add_argument("--output-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--test-normal-count", type=int, default=140)
    parser.add_argument("--test-anomaly-count", type=int, default=60)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    return parser.parse_args()


def shuffled(values, seed: int):
    out = list(values)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def main() -> None:
    args = parse_args()
    root = args.source_class_root
    if not root.exists():
        raise FileNotFoundError(root)

    production_normal = rel_list(root / "train" / "good", root)
    fixed_test_normal = rel_list(root / "test" / "good", root)
    all_anomaly = rel_list(root / "test" / "defect", root)

    ensure_no_duplicates("production_normal", production_normal)
    ensure_no_duplicates("fixed_test_normal", fixed_test_normal)
    ensure_no_duplicates("all_anomaly", all_anomaly)

    if len(production_normal) != 560:
        raise ValueError(f"Expected 560 production normal samples, got {len(production_normal)}")
    if len(fixed_test_normal) != args.test_normal_count:
        raise ValueError(f"Expected {args.test_normal_count} fixed test normal samples, got {len(fixed_test_normal)}")
    if len(all_anomaly) != 199:
        raise ValueError(f"Expected 199 anomaly samples, got {len(all_anomaly)}")

    anomaly_shuffled = shuffled(all_anomaly, args.seed + 17)
    fixed_test_anomaly = anomaly_shuffled[: args.test_anomaly_count]
    production_anomaly = anomaly_shuffled[args.test_anomaly_count :]
    if len(production_anomaly) != 139:
        raise ValueError(f"Expected 139 production anomaly samples, got {len(production_anomaly)}")

    production_normal_order = shuffled(production_normal, args.seed + 31)
    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty split root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    write_lines(output_root / "test_normal.txt", fixed_test_normal)
    write_lines(output_root / "test_anomaly.txt", fixed_test_anomaly)
    write_lines(output_root / "production_normal_pool.txt", production_normal_order)
    write_lines(output_root / "production_anomaly_pool.txt", production_anomaly)

    for stage, spec in STAGE_SPECS.items():
        normals_available = production_normal_order[: spec.available_normal]
        anomalies_available = production_anomaly[: spec.available_anomaly]
        train_normal = normals_available[: spec.train_normal]
        calib_normal = normals_available[spec.train_normal : spec.train_normal + spec.calib_normal]
        train_anomaly = anomalies_available[: spec.train_anomaly]
        calib_anomaly = anomalies_available[spec.train_anomaly : spec.train_anomaly + spec.calib_anomaly]

        if len(train_normal) != spec.train_normal or len(calib_normal) != spec.calib_normal:
            raise ValueError(f"Normal count mismatch for {stage}")
        if len(train_anomaly) != spec.train_anomaly or len(calib_anomaly) != spec.calib_anomaly:
            raise ValueError(f"Anomaly count mismatch for {stage}")

        root_stage = stage_dir(output_root, stage)
        write_lines(root_stage / "train_normal.txt", train_normal)
        write_lines(root_stage / "train_anomaly.txt", train_anomaly)
        write_lines(root_stage / "calib_normal.txt", calib_normal)
        write_lines(root_stage / "calib_anomaly.txt", calib_anomaly)

    config = {
        "alias": args.alias,
        "source_class_root": str(root),
        "split_root": str(output_root),
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "path_format": "relative_to_source_class_root",
        "data_inventory": {
            "production_normal_from_train_good": len(production_normal),
            "fixed_test_normal_from_test_good": len(fixed_test_normal),
            "all_anomaly_from_test_defect": len(all_anomaly),
            "fixed_test_anomaly": len(fixed_test_anomaly),
            "production_anomaly_pool": len(production_anomaly),
        },
        "stage_specs": stage_spec_dict(),
        "notes": [
            "test/good is used as the fixed 140-normal test set.",
            "test/defect is shuffled once; 60 anomalies become fixed test anomalies and the remaining 139 form the production anomaly pool.",
            "Stage train/calibration sets are cumulative prefixes of the production pools.",
        ],
    }
    write_json(output_root / "split_config.json", config)
    print(f"Split generation OK: {output_root}")


if __name__ == "__main__":
    main()