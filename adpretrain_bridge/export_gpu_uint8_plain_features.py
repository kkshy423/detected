#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export DINO-large projected residual feature caches for qm_xiepai using the
GPU uint8 preprocessing path (gpu_tensor_uint8_aa_true).

Mirrors export_qm_xiepai_dino_large_plain_features.py exactly EXCEPT the image
preprocessing: instead of CPU-PIL build_transform/load_image_tensor, both the
reference bank and every query image are preprocessed via
prepare_backend_from_pil('gpu_tensor_uint8_aa_true', ...). Everything downstream
(encoder, reference matching, residual, projector, compress_four_to_two, npy
save, runtime/meta json) is identical so the resulting cache is a drop-in GPU
counterpart of plain_dino_large_norm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    add_adpretrain_to_path,
    compress_four_to_two,
    ensure_dataset_links,
    list_image_files,
    make_encoder,
    make_projector,
    residual_features,
    write_json,
)
from benchmark_preprocess_equivalence_paired import (
    MEAN,
    STD,
    flatten_feature_map,
    load_pil_image,
    prepare_backend_from_pil,
)

GPU_BACKEND = "gpu_tensor_uint8_aa_true"
DEFAULT_SOURCE_DATASET_ROOT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1")
DEFAULT_CACHE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_uint8_offline_v1"
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
    parser.add_argument("--max-images", type=int, default=None, help="smoke cap on number of exported images")
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


def build_reference_features_gpu(encoder, ref_paths, device, mean_gpu, std_gpu) -> List[torch.Tensor]:
    refs = None
    with torch.no_grad():
        for path in ref_paths:
            pil = load_pil_image(path)
            tensor, _, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
            features = list(encoder.encode_image_from_tensors(tensor, return_global=False, shape="img"))
            flat = [flatten_feature_map(x).detach() for x in features]
            if refs is None:
                refs = [[] for _ in flat]
            for idx, fmap in enumerate(flat):
                refs[idx].append(fmap)
    if refs is None:
        raise ValueError("No reference features built")
    return [torch.cat(items, dim=0).to(device) for items in refs]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU uint8 export requires CUDA.")
    source_class_root = resolve_source_class_root(args)
    cache_class_root = args.cache_root / args.alias
    if cache_class_root.exists() and any(cache_class_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache root: {cache_class_root}")

    add_adpretrain_to_path(str(args.adpretrain_root))
    ensure_dataset_links(source_class_root, cache_class_root)
    cache_class_root.mkdir(parents=True, exist_ok=True)
    (cache_class_root / "feature").mkdir(parents=True, exist_ok=True)
    (cache_class_root / "feature_scale").mkdir(parents=True, exist_ok=True)

    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    encoder.eval()
    projector.eval()
    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    train_good = sorted(list_image_files(source_class_root / "train" / "good"), key=lambda p: p.name)
    ref_paths = train_good[: args.num_ref]
    if not ref_paths:
        raise ValueError(f"No reference images found under {source_class_root / 'train' / 'good'}")
    refs = build_reference_features_gpu(encoder, ref_paths, device, mean_gpu, std_gpu)
    ref_norm = [F.normalize(ref, p=2, dim=1) for ref in refs]

    all_items = list_all_images(source_class_root)
    if args.max_images is not None:
        all_items = all_items[: args.max_images]
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
            pil = load_pil_image(image_path)
            batch, _, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
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
                flat_n = F.normalize(flat, p=2, dim=1)
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
        "preprocess_backend": GPU_BACKEND,
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
        "preprocess_backend": GPU_BACKEND,
        "device_requested": args.device,
        "device_used": str(device),
        "reference_count": len(ref_paths),
        "feature_rule": "projected residual cached pair; GPU uint8 preprocessing (resize/crop in uint8 domain, cast after crop); identical downstream to plain_dino_large_norm",
        "counts": counts,
    }
    write_json(args.cache_root / "feature_cache_manifest.json", meta)
    write_json(cache_class_root / "adpretrain_bridge_meta.json", meta)
    print(json.dumps({"status": "ok", "cache_class_root": str(cache_class_root), "sample_count": len(total_times)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

