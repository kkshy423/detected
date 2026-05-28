#!/usr/bin/env python3
"""Create a per-channel moment-matched ADPretrain feature cache for AHL.

This script is a decoupled cache post-processing step. It reads an existing
ADPretrain cache, estimates per-channel mean/std from train/good only, matches
those moments to a reference AHL/DRA cache, and writes a new cache root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

SCALES = ("feature", "feature_scale")
LINK_DIRS = ("train", "test", "ground_truth")
NOAUG_ALIAS_PAIRS = (("aug", "feature"), ("aug_scale", "feature_scale"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-channel moment-match ADPretrain cache to reference AHL cache.")
    parser.add_argument("--source-root", type=Path, required=True, help="Existing ADPretrain cache root.")
    parser.add_argument("--reference-root", type=Path, required=True, help="Original/reference AHL cache root.")
    parser.add_argument("--output-root", type=Path, required=True, help="New output cache root.")
    parser.add_argument("--class-name", default=None, help="Real source/reference class name. If omitted, use --discover-prefix.")
    parser.add_argument("--discover-prefix", default="models__", help="Prefix used to discover class without non-ASCII shell args.")
    parser.add_argument("--reference-class-name", default=None, help="Reference class name. Defaults to --class-name/source class.")
    parser.add_argument("--reference-discover-prefix", default=None, help="Prefix used to discover the reference class when it differs from source.")
    parser.add_argument("--alias-class", required=True, help="ASCII class name written under output root.")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-aug-alias", action="store_true", help="Do not create aug/aug_scale aliases.")
    return parser.parse_args()


def discover_class(root: Path, prefix: str) -> str:
    matches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"No class directory starts with {prefix!r} under {root}")
    if len(matches) > 1:
        print(f"WARNING: multiple matches for prefix {prefix!r}, using {matches[0]!r}: {matches}")
    return matches[0]


def npy_files(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.rglob("*.npy") if p.is_file())


def ensure_link(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            target = os.readlink(dst)
            resolved = (dst.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
            if resolved == src.resolve():
                return "exists"
        if dst.is_dir() and not dst.is_symlink():
            raise FileExistsError(f"Refusing to replace real directory: {dst}")
        dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "created"


def channel_mean_std(files: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    total = None
    total_sq = None
    count = 0
    file_count = 0
    channel_count = None
    for path in files:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"Expected CHW array, got shape={arr.shape} at {path}")
        if channel_count is None:
            channel_count = int(arr.shape[0])
            total = np.zeros(channel_count, dtype=np.float64)
            total_sq = np.zeros(channel_count, dtype=np.float64)
        elif int(arr.shape[0]) != channel_count:
            raise ValueError(f"Channel mismatch at {path}: {arr.shape[0]} != {channel_count}")
        arr64 = arr.astype(np.float64, copy=False)
        total += arr64.sum(axis=(1, 2))
        total_sq += np.square(arr64).sum(axis=(1, 2))
        count += int(arr.shape[1] * arr.shape[2])
        file_count += 1
    if file_count == 0 or channel_count is None or total is None or total_sq is None:
        raise ValueError("No files found for per-channel stats")
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    return mean, np.sqrt(var), file_count, channel_count


def write_scale(
    src_class: Path,
    ref_class: Path,
    out_class: Path,
    scale: str,
    eps: float,
    skip_existing: bool,
    stats_dir: Optional[Path] = None,
) -> Dict[str, object]:
    src_train = src_class / scale / "train" / "good"
    ref_train = ref_class / scale / "train" / "good"
    src_mean, src_std, src_files, src_channels = channel_mean_std(npy_files(src_train))
    ref_mean, ref_std, ref_files, ref_channels = channel_mean_std(npy_files(ref_train))
    if src_channels != ref_channels:
        raise ValueError(f"Channel mismatch for {scale}: source={src_channels}, reference={ref_channels}")

    if stats_dir is not None:
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stats_dir / f"{scale}_channel_moments.npz",
            source_mean=src_mean.astype(np.float32),
            source_std=src_std.astype(np.float32),
            reference_mean=ref_mean.astype(np.float32),
            reference_std=ref_std.astype(np.float32),
        )

    src_scale_root = src_class / scale
    out_scale_root = out_class / scale
    mean = src_mean[:, None, None]
    denom = (src_std + eps)[:, None, None]
    target_std = ref_std[:, None, None]
    target_mean = ref_mean[:, None, None]

    written = 0
    skipped = 0
    for src_path in npy_files(src_scale_root):
        rel = src_path.relative_to(src_scale_root)
        out_path = out_scale_root / rel
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.load(src_path).astype(np.float32, copy=False)
        if arr.ndim != 3 or arr.shape[0] != src_channels:
            raise ValueError(f"Unexpected shape={arr.shape} at {src_path}")
        matched = (arr - mean) / denom * target_std + target_mean
        np.save(out_path, matched.astype(np.float32, copy=False))
        written += 1

    return {
        "channel_count": src_channels,
        "source_train_files": src_files,
        "reference_train_files": ref_files,
        "source_mean_avg": float(src_mean.mean()),
        "source_std_avg": float(src_std.mean()),
        "reference_mean_avg": float(ref_mean.mean()),
        "reference_std_avg": float(ref_std.mean()),
        "source_mean_min": float(src_mean.min()),
        "source_mean_max": float(src_mean.max()),
        "source_std_min": float(src_std.min()),
        "source_std_max": float(src_std.max()),
        "reference_mean_min": float(ref_mean.min()),
        "reference_mean_max": float(ref_mean.max()),
        "reference_std_min": float(ref_std.min()),
        "reference_std_max": float(ref_std.max()),
        "stats_npz": str(stats_dir / f"{scale}_channel_moments.npz") if stats_dir is not None else None,
        "written": written,
        "skipped": skipped,
    }


def main() -> None:
    args = parse_args()
    class_name = args.class_name or discover_class(args.source_root, args.discover_prefix)
    if args.reference_class_name:
        reference_class_name = args.reference_class_name
    elif args.reference_discover_prefix:
        reference_class_name = discover_class(args.reference_root, args.reference_discover_prefix)
    else:
        reference_class_name = class_name
    src_class = args.source_root / class_name
    ref_class = args.reference_root / reference_class_name
    out_class = args.output_root / args.alias_class
    if not src_class.exists():
        raise FileNotFoundError(src_class)
    if not ref_class.exists():
        raise FileNotFoundError(ref_class)
    out_class.mkdir(parents=True, exist_ok=True)

    linked: Dict[str, str] = {}
    for name in LINK_DIRS:
        src = src_class / name
        if src.exists():
            linked[name] = ensure_link(src, out_class / name)

    stats: Dict[str, Dict[str, object]] = {}
    for scale in SCALES:
        stats[scale] = write_scale(src_class, ref_class, out_class, scale, args.eps, args.skip_existing, out_class / "moment_stats")

    noaug_aliases: Dict[str, str] = {}
    if not args.no_aug_alias:
        for alias_name, source_name in NOAUG_ALIAS_PAIRS:
            noaug_aliases[alias_name] = ensure_link(out_class / source_name, out_class / alias_name)

    meta = {
        "method": "per_channel_moment_match_train_good_to_reference",
        "source_root": str(args.source_root),
        "reference_root": str(args.reference_root),
        "output_root": str(args.output_root),
        "source_class": class_name,
        "reference_class": reference_class_name,
        "alias_class": args.alias_class,
        "eps": args.eps,
        "linked_dirs": linked,
        "noaug_aliases": noaug_aliases,
        "scales": stats,
        "note": "Uses train/good only for per-channel moments; applies same affine transform to train/test npy files per scale.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "channel_moment_match_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
