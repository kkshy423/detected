#!/usr/bin/env python3
"""Benchmark ADPretrain/CLIP feature extraction and optional CHMM affine time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np
import torch

from common import (
    build_reference_features,
    build_transform,
    compress_four_to_two,
    encode_multiscale,
    iter_mvtec_images,
    list_image_files,
    load_image_tensor,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
)
from xidun_aliases import suffixed_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure per-image ADPretrain feature pipeline latency.")
    parser.add_argument("--adpretrain-root", default="/ghome/huangjd/code/detected/ADPretrain")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backbone", default="clip-base", choices=["clip-base", "dinov2-base", "dinov2-large"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alias-profile", default="xidun3", choices=["xidun3", "xidun6"])
    parser.add_argument("--alias-suffix", default="clip_official_effective_plain")
    parser.add_argument("--chmm-root", type=Path, default=None, help="CHMM cache root containing moment_stats; omit for plain.")
    parser.add_argument("--num-ref", type=int, default=8)
    parser.add_argument("--max-images-per-class", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_chmm_stats(chmm_root: Optional[Path], alias_class: str, device: torch.device):
    if chmm_root is None:
        return None
    stats = {}
    for scale in ("feature", "feature_scale"):
        path = chmm_root / alias_class / "moment_stats" / f"{scale}_channel_moments.npz"
        data = np.load(path)
        stats[scale] = {
            "source_mean": torch.from_numpy(data["source_mean"]).to(device=device, dtype=torch.float32)[:, None, None],
            "source_std": torch.from_numpy(data["source_std"]).to(device=device, dtype=torch.float32)[:, None, None],
            "reference_mean": torch.from_numpy(data["reference_mean"]).to(device=device, dtype=torch.float32)[:, None, None],
            "reference_std": torch.from_numpy(data["reference_std"]).to(device=device, dtype=torch.float32)[:, None, None],
        }
    return stats


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, args.adpretrain_root, device)
    projector = make_projector(encoder, args.adpretrain_root, args.checkpoint, device)
    rows = []

    for real_class, alias_class in suffixed_aliases(args.alias_suffix, args.alias_profile):
        class_root = args.dataset_root / real_class
        ref_paths = list_image_files(class_root / "train" / "good")
        refs = build_reference_features(encoder, ref_paths, transform, device, args.num_ref)
        images = []
        for _, img_path in iter_mvtec_images(class_root, "test"):
            images.append(load_image_tensor(img_path, transform, device))
            if len(images) >= args.max_images_per_class:
                break
        if not images:
            raise ValueError(f"No benchmark images for {real_class}")
        stats = load_chmm_stats(args.chmm_root, alias_class, device)
        feature_times = []
        affine_times = []
        outputs = []

        def forward_one(image):
            with torch.no_grad():
                features = encode_multiscale(encoder, image)
                matched = match_reference_features(features, refs)
                residual = residual_features(features, matched)
                projected = projector(*residual)
                return compress_four_to_two(projected)

        for idx, image in enumerate(images):
            out = None
            ms = timed_ms(device, lambda: outputs.append(forward_one(image)))
            out = outputs[-1]
            if idx >= args.warmup:
                feature_times.append(ms)
            if stats is not None:
                def affine():
                    feature, feature_scale = out
                    for scale, tensor in (("feature", feature), ("feature_scale", feature_scale)):
                        st = stats[scale]
                        _ = (tensor - st["source_mean"]) / (st["source_std"] + 1e-6) * st["reference_std"] + st["reference_mean"]
                affine_ms = timed_ms(device, affine)
                if idx >= args.warmup:
                    affine_times.append(affine_ms)

        rows.append({
            "class": alias_class,
            "real_class": real_class,
            "samples": len(images),
            "warmup": args.warmup,
            "clip_feature_ms": float(mean(feature_times)) if feature_times else None,
            "chmm_affine_ms": float(mean(affine_times)) if affine_times else None,
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ADPretrain Latency",
        "",
        "| Class | Samples | CLIP Feature ms | CHMM Affine ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        feature = "N/A" if row["clip_feature_ms"] is None else f"{row['clip_feature_ms']:.4f}"
        affine = "N/A" if row["chmm_affine_ms"] is None else f"{row['chmm_affine_ms']:.4f}"
        lines.append(f"| {row['class']} | {row['samples']} | {feature} | {affine} |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.output_json}")
    print(f"WROTE {args.output_md}")


if __name__ == "__main__":
    main()
