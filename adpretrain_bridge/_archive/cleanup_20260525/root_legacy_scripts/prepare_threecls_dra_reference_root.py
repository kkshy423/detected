#!/usr/bin/env python3
"""Prepare a three-class DRA reference cache root without touching raw data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xidun_aliases import suffixed_aliases


LINK_DIRS = ("train", "test", "ground_truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DRA reference cache root for the Xidun three-class dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alias-profile", default="xidun3", choices=["xidun3"])
    return parser.parse_args()


def ensure_link(src: Path, dst: Path) -> str:
    if not src.exists():
        return "source_missing"
    if dst.is_symlink():
        target = os.readlink(dst)
        resolved = (dst.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
        if resolved == src.resolve():
            return "exists"
        raise RuntimeError(f"symlink conflict: {dst} -> {target}, expected {src}")
    if dst.exists():
        raise RuntimeError(f"refusing to replace non-symlink path: {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "created"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "alias_profile": args.alias_profile,
        "classes": {},
        "note": "train/test/ground_truth are symlinks; feature/feature_scale are written here by AHL extraction.",
    }
    for real_class, _ in suffixed_aliases("", args.alias_profile):
        class_src = args.dataset_root / real_class
        class_dst = args.output_root / real_class
        class_dst.mkdir(parents=True, exist_ok=True)
        links = {}
        for name in LINK_DIRS:
            links[name] = ensure_link(class_src / name, class_dst / name)
        meta["classes"][real_class] = {"links": links}
    (args.output_root / "dra_reference_root_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
