#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export DINO-large projected residual feature caches for qm_xiepai."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from common import (
    add_adpretrain_to_path,
    build_reference_features,
    build_transform,
    compress_four_to_two,
    ensure_dataset_links,
    list_image_files,
    load_image_tensor,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
    write_json,
)


DEFAULT_SOURCE_DATASET_ROOT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1")
DEFAULT_CACHE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm"
)
DEFAULT_ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
DEFAULT_PROJECTOR = Path("/ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth")
DEFAULT_ALIAS = "models_qiumianxiepai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-class-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--adpretrain-root", type=Path, default=DEFAULT_ADPRETRAIN_ROOT)
    parser.add_argument("--projector-checkpoint", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--backbone", default="dinov2-large")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_source_class_root(args: argparse.Namespace) -> Path:
    if args.source_class_root is not None:
        return args.source_class_root
    candidates = sorted([p for p in args.source_dataset_root.glob("models__*") if p.is_dir()])
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one models__* folder under {args.source_dataset_root}, got {candidates}")
    return candidates[0]


def list_all_images(class_root: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for split in ["train", "val", "test"]:
        split_root = class_root / split
        if not split_root.exists():
            continue
        for class_dir in sorted([p for p in split_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for image_path in list_image_files(class_dir):
                rel = image_path.relative_to(class_root).as_posix()
                rows.append((rel, image_path))
    return rows


def save_feature_pair(cache_class_root: Path, rel_path: str, feature: torch.Tensor, feature_scale: torch.Tensor) -> None:
    rel = Path(rel_path)
    feature_path = cache_class_root / "feature" / rel.with_suffix(".npy")
    feature_scale_path = cache_class_root / "feature_scale" / rel.with_suffix(".npy")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_scale_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature.squeeze(0).detach().cpu().numpy().astype(np.float32))
    np.save(feature_scale_path, feature_scale.squeeze(0).detach().cpu().numpy().astype(np.float32))


def stage_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean_ms": None, "median_ms": None, "p90_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    source_class_root = resolve_source_class_root(args)
    cache_class_root = args.cache_root / args.alias
    if cache_class_root.exists() and any(cache_class_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache root: {cache_class_root}")

    add_adpretrain_to_path(str(args.adpretrain_root))
    ensure_dataset_links(source_class_root, cache_class_root)
    cache_class_root.mkdir(parents=True, exist_ok=True)
    (cache_class_root / "feature").mkdir(parents=True, exist_ok=True)
    (cache_class_root / "feature_scale").mkdir(parents=True, exist_ok=True)

    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    encoder.eval()
    projector.eval()

    train_good = sorted(list_image_files(source_class_root / "train" / "good"), key=lambda p: p.name)
    ref_paths = train_good[: args.num_ref]
    if not ref_paths:
        raise ValueError(f"No reference images found under {source_class_root / 'train' / 'good'}")
    refs = build_reference_features(encoder, ref_paths, transform, device, num_ref=len(ref_paths))
    ref_norm = [torch.nn.functional.normalize(ref, p=2, dim=1) for ref in refs]

    all_items = list_all_images(source_class_root)
    if not all_items:
        raise ValueError(f"No images found under {source_class_root}")

    phase_load: List[float] = []
    phase_encoder: List[float] = []
    phase_match_project: List[float] = []
    total_times: List[float] = []
    counts: Dict[str, int] = {}

    with torch.no_grad():
        for rel_path, image_path in all_items:
            counts[rel_path.split("/", 1)[0]] = counts.get(rel_path.split("/", 1)[0], 0) + 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            batch = load_image_tensor(image_path, transform, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            features = encoder.encode_image_from_tensors(batch, return_global=False, shape="img")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            matched = []
            for feature, ref, ref_n in zip(features, refs, ref_norm):
                b, c, h, w = feature.shape
                flat = feature.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
                flat_n = torch.nn.functional.normalize(flat, p=2, dim=1)
                index = torch.argmax(flat_n @ ref_n.T, dim=1)
                picked = ref[index].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
                matched.append(picked)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            feature, feature_scale = compress_four_to_two(projected)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            save_feature_pair(cache_class_root, rel_path, feature, feature_scale)
            phase_load.append((t1 - t0) * 1000.0)
            phase_encoder.append((t2 - t1) * 1000.0)
            phase_match_project.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

    runtime = {
        "status": "ok",
        "time_kind": "single_image_adpretrain_feature_mean_ms",
        "time_adpretrain_feature_ms": float(np.mean(total_times)),
        "time_adpretrain_encoder_ms": float(np.mean(phase_encoder)),
        "time_adpretrain_projector_ms": float(np.mean(phase_match_project)),
        "time_adpretrain_projected_feature_ms": float(np.mean(total_times)),
        "time_load_transform_ms": float(np.mean(phase_load)),
        "sample_count": int(len(total_times)),
        "reference_count": int(len(ref_paths)),
        "reference_paths": [str(p) for p in ref_paths],
    }
    write_json(cache_class_root / "feature_runtime.json", runtime)

    meta = {
        "status": "ok",
        "source_dataset_root": str(args.source_dataset_root),
        "source_class_root": str(source_class_root),
        "output_root": str(args.cache_root),
        "cache_class_root": str(cache_class_root),
        "alias": args.alias,
        "backbone": args.backbone,
        "checkpoint": str(args.projector_checkpoint),
        "num_ref": args.num_ref,
        "device_requested": args.device,
        "device_used": str(device),
        "reference_count": len(ref_paths),
        "feature_rule": "projected residual cached pair; layer3/layer4 resized to 14x14 then averaged as feature; layer1/layer2 resized to 7x7 then averaged as feature_scale; projector output is saved as float32 npy",
        "counts": counts,
    }
    write_json(args.cache_root / "feature_cache_manifest.json", meta)
    write_json(cache_class_root / "adpretrain_bridge_meta.json", meta)
    print(json.dumps({"status": "ok", "cache_class_root": str(cache_class_root), "sample_count": len(total_times)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
