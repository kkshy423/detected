#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADP-only-DINO offline scores using GPU uint8 preprocessing, per stage.

Mirrors evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py's ADP-only score math
(sqrt(||patch||+1)-1 -> upsample+avg+gaussian sigma4 -> spatial max) but uses the
gpu_tensor_uint8_aa_true preprocessing for both reference and query. Writes per
stage scores.csv (role,split,label,score,source_rel) so bridge can fuse with the
GPU AHL scores in the same sample order.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from common import (
    add_adpretrain_to_path,
    compress_four_to_two,  # noqa: F401 (kept for parity, unused)
    encode_multiscale,
    make_encoder,
    make_projector,
    residual_features,
)
from benchmark_preprocess_equivalence_paired import (
    MEAN,
    STD,
    flatten_feature_map,
    load_pil_image,
    prepare_backend_from_pil,
)
from fewshot_qm_xiepai_common import read_lines

GPU_BACKEND = "gpu_tensor_uint8_aa_true"
SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/20260529_qm_xiepai_6_1_fixed_180_70_val49")
SOURCE_PARENT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1")
ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
PROJECTOR = Path("/ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--source-parent", type=Path, default=SOURCE_PARENT)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    p.add_argument("--projector-checkpoint", type=Path, default=PROJECTOR)
    p.add_argument("--backbone", default="dinov2-large")
    p.add_argument("--num-ref", type=int, default=8)
    p.add_argument("--feature-levels", type=int, default=4)
    p.add_argument("--stages", nargs="+", default=[f"S{i}" for i in range(9)])
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def resolve_source_class_root(source_parent: Path) -> Path:
    cands = sorted([p for p in source_parent.glob("models__*") if p.is_dir()])
    if len(cands) != 1:
        raise RuntimeError(f"expected 1 models__* under {source_parent}, got {cands}")
    return cands[0]


def aggregate_score_maps(score_maps, size):
    maps = []
    for sm in score_maps:
        up = F.interpolate(sm.unsqueeze(1), size=size, mode="bilinear", align_corners=True)
        maps.append(up.squeeze(1).detach().cpu().numpy())
    scores = np.zeros_like(maps[0])
    for m in maps:
        scores += m
    scores /= len(maps)
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)
    return scores


def adp_only_image_score(projected, image_size, feature_levels):
    score_maps = []
    for feature in projected:
        bsz, dim, h, w = feature.size()
        patch = feature.permute(0, 2, 3, 1).reshape(-1, dim)
        patch_score = torch.sqrt(torch.linalg.norm(patch, dim=1) + 1.0) - 1.0
        score_maps.append(patch_score.reshape(bsz, h, w).float())
    amap = aggregate_score_maps(score_maps[:feature_levels], size=image_size)
    return float(amap.reshape(amap.shape[0], -1).max(axis=1)[0])


def build_refs_gpu(encoder, ref_paths, device, mean_gpu, std_gpu):
    refs = None
    with torch.no_grad():
        for path in ref_paths:
            pil = load_pil_image(path)
            tensor, _, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
            feats = encode_multiscale(encoder, tensor)
            flat = [flatten_feature_map(x).detach() for x in feats]
            if refs is None:
                refs = [[] for _ in flat]
            for i, fm in enumerate(flat):
                refs[i].append(fm)
    return [torch.cat(items, dim=0).to(device) for items in refs]


def stage_eval_rows(split_root: Path, stage: str, source_class_root: Path) -> List[Dict]:
    """calib (top-level) + test (top-level) eval rows. ADP-only is training-free."""
    spec = [
        ("calib_normal", split_root / "calib_normal.txt", "calib", 0),
        ("calib_anomaly", split_root / "calib_anomaly.txt", "calib", 1),
        ("test_normal", split_root / "test_normal.txt", "test", 0),
        ("test_anomaly", split_root / "test_anomaly.txt", "test", 1),
    ]
    rows = []
    for role, txt, split, label in spec:
        for rel in read_lines(txt):
            rows.append({"role": role, "split": split, "label": label, "source_rel": rel, "path": source_class_root / rel})
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU ADP-only requires CUDA.")
    source_class_root = resolve_source_class_root(args.source_parent)
    add_adpretrain_to_path(str(args.adpretrain_root))
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    encoder.eval(); projector.eval()
    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    for stage in args.stages:
        metrics_dir = args.output_root / "stages" / stage / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ref_rels = read_lines(args.split_root / stage / "train_normal.txt")[: args.num_ref]
        ref_paths = [source_class_root / rel for rel in ref_rels]
        refs = build_refs_gpu(encoder, ref_paths, device, mean_gpu, std_gpu)
        ref_norm = [F.normalize(r, p=2, dim=1) for r in refs]

        rows = stage_eval_rows(args.split_root, stage, source_class_root)
        out_rows = []
        with torch.no_grad():
            for r in rows:
                pil = load_pil_image(r["path"])
                tensor, _, _ = prepare_backend_from_pil(GPU_BACKEND, pil, device, mean_gpu, std_gpu)
                features = encode_multiscale(encoder, tensor)
                matched = []
                for feature, ref, ref_n in zip(features, refs, ref_norm):
                    b, c, h, w = feature.shape
                    flat = feature.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
                    flat_n = F.normalize(flat, p=2, dim=1)
                    idx = torch.argmax(flat_n @ ref_n.T, dim=1)
                    matched.append(ref[idx].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous())
                residual = residual_features(features, matched)
                projected = projector(*residual)
                score = adp_only_image_score(projected, tensor.shape[-1], args.feature_levels)
                out_rows.append({"role": r["role"], "split": r["split"], "label": r["label"], "score": score, "source_rel": r["source_rel"]})
        with (metrics_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["role", "split", "label", "score", "source_rel"])
            w.writeheader(); w.writerows(out_rows)
        print(json.dumps({"stage": stage, "n": len(out_rows), "scores": str(metrics_dir / "scores.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

