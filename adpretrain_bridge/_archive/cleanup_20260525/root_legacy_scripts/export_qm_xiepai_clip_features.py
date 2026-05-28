#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export/reuse CLIP-B16 official ADPretrain cache and derive CHMM cache."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fewshot_qm_xiepai_common import (
    ADPRETRAIN_ROOT,
    ALIAS_CLASS,
    BRIDGE_ROOT,
    CACHE_BASE,
    OFFICIAL_CLIP_PROJECTOR,
    REFERENCE_DATASET_ROOT,
    SOURCE_DATASET_ROOT,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qm_xiepai CLIP-B16 feature caches for few-shot AHL.")
    parser.add_argument("--dataset-root", type=Path, default=SOURCE_DATASET_ROOT)
    parser.add_argument("--output-base", type=Path, default=CACHE_BASE)
    parser.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CLIP_PROJECTOR)
    parser.add_argument("--alias", default=ALIAS_CLASS)
    parser.add_argument("--plain-name", default="plain_official_img_angle")
    parser.add_argument("--chmm-name", default="chmm_official_img_angle_draref")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_DATASET_ROOT)
    parser.add_argument("--reference-discover-prefix", default="models__")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--skip-plain-export", action="store_true")
    parser.add_argument("--skip-chmm", action="store_true")
    parser.add_argument("--check-cache", action="store_true", default=True)
    parser.add_argument("--no-check-cache", dest="check_cache", action="store_false")
    return parser.parse_args()


def run(cmd, cwd: Path) -> None:
    print("RUN", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def cache_counts(root: Path, alias: str) -> dict:
    class_root = root / alias
    out = {}
    for scale in ["feature", "feature_scale"]:
        for rel in ["train/good", "test/good", "test/defect"]:
            folder = class_root / scale / rel
            out[f"{scale}/{rel}"] = len(list(folder.glob("*.npy"))) if folder.exists() else 0
    return out


def main() -> None:
    args = parse_args()
    plain_root = args.output_base / args.plain_name
    chmm_root = args.output_base / args.chmm_name
    args.output_base.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    if not args.skip_plain_export:
        cmd = [
            sys.executable,
            BRIDGE_ROOT / "export_plain_features.py",
            "--adpretrain-root", args.adpretrain_root,
            "--dataset-root", args.dataset_root,
            "--output-root", plain_root,
            "--backbone", "clip-base",
            "--checkpoint", args.checkpoint,
            "--discover-prefix", "models__",
            "--alias-class", args.alias,
            "--num-ref", str(args.num_ref),
            "--device", args.device,
            "--splits", *args.splits,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        run(cmd, BRIDGE_ROOT)

    for root in [plain_root]:
        cmd = [
            sys.executable,
            BRIDGE_ROOT / "prepare_plain_aug_aliases.py",
            "--dataset-root", root,
            "--classes", args.alias,
            "--mode", "all",
        ]
        run(cmd, BRIDGE_ROOT)

    if not args.skip_chmm:
        cmd = [
            sys.executable,
            BRIDGE_ROOT / "channel_moment_match_cache.py",
            "--source-root", plain_root,
            "--reference-root", args.reference_root,
            "--output-root", chmm_root,
            "--class-name", args.alias,
            "--reference-discover-prefix", args.reference_discover_prefix,
            "--alias-class", args.alias,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        run(cmd, BRIDGE_ROOT)
        cmd = [
            sys.executable,
            BRIDGE_ROOT / "prepare_plain_aug_aliases.py",
            "--dataset-root", chmm_root,
            "--classes", args.alias,
            "--mode", "all",
        ]
        run(cmd, BRIDGE_ROOT)

    if args.check_cache:
        for root in [plain_root, chmm_root]:
            if root.exists():
                cmd = [
                    sys.executable,
                    BRIDGE_ROOT / "check_feature_cache.py",
                    "--dataset-root", root,
                    "--classes", args.alias,
                ]
                run(cmd, BRIDGE_ROOT)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(args.dataset_root),
        "alias": args.alias,
        "backbone": "clip-base",
        "checkpoint": str(args.checkpoint),
        "plain_root": str(plain_root),
        "chmm_root": str(chmm_root),
        "reference_root": str(args.reference_root),
        "reference_discover_prefix": args.reference_discover_prefix,
        "num_ref": args.num_ref,
        "device": args.device,
        "skip_existing": args.skip_existing,
        "counts": {
            "plain": cache_counts(plain_root, args.alias) if plain_root.exists() else {},
            "chmm": cache_counts(chmm_root, args.alias) if chmm_root.exists() else {},
        },
        "note": "Plain cache uses the fixed official CLIP-B16 projector without projector fine-tuning; CHMM is per-channel affine matching from source train/good to the DRA reference cache.",
    }
    write_json(args.output_base / "feature_cache_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
