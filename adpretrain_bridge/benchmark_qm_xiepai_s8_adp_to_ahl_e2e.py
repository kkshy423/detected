#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8 end-to-end benchmark from raw image to AHL threshold decision.

This script keeps the whole path in one process:
raw image -> transform -> encoder -> reference matching -> residual ->
projector -> compress -> AHL scoring -> threshold.

It does not write .npy feature caches. Reference banks are preloaded in memory
before the timed test pass. Calibration is used only to verify the fixed
production threshold chosen from the existing S8 calibration output.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (
    add_adpretrain_to_path,
    build_transform,
    compress_four_to_two,
    encode_multiscale,
    list_image_files,
    make_encoder,
    make_projector,
    residual_features,
)


BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
AHL_ROOT = Path("/ghome/huangjd/code/detected/AHL")
DEFAULT_STAGE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/20260529_s8_time_profile_stage_roots_plain_fixed_180_70_49/S8"
)
DEFAULT_ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
DEFAULT_PROJECTOR = Path("/ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth")
DEFAULT_AHL_WEIGHTS = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S8/ahl/models_qiumianxiepai_ctest.pkl"
)
DEFAULT_THRESHOLD_COMPARE = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_time_profile_threshold_compare/policy_compare.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260604_s8_adp_to_ahl_e2e_gpu_preprocess_v1"
)
DEFAULT_ALIAS = "models_qiumianxiepai"
DEFAULT_STAGE = "S8"
DEFAULT_POLICY = "strategy_mild_stage_v2_1_safe"


@dataclass
class SampleResult:
    rel_path: str
    role: str
    label: int
    score: float
    pred: int
    phases_ms: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--adpretrain-root", type=Path, default=DEFAULT_ADPRETRAIN_ROOT)
    parser.add_argument("--projector-checkpoint", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--ahl-weights", type=Path, default=DEFAULT_AHL_WEIGHTS)
    parser.add_argument("--threshold-compare-json", type=Path, default=DEFAULT_THRESHOLD_COMPARE)
    parser.add_argument("--threshold-policy", default=DEFAULT_POLICY)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--n-ref", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-calib", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--preprocess-backend",
        choices=("cpu_pil", "gpu_tensor", "gpu_tensor_uint8"),
        default="cpu_pil",
        help="cpu_pil keeps the legacy PIL/torchvision CPU transform; gpu_tensor keeps decode on CPU and runs tensor resize/crop/normalize on GPU.",
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    sync(device)
    start = time.perf_counter()
    out = fn()
    sync(device)
    return out, (time.perf_counter() - start) * 1000.0


def load_stage_manifest(stage_root: Path, alias: str) -> Dict:
    manifest_path = stage_root / alias / "stage_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_role_paths(manifest: Dict, role: str) -> List[Path]:
    class_root = Path(manifest["stage_class_root"])
    rows = [row for row in manifest["mappings"] if row["role"] == role]
    return [class_root / row["stage_rel"] for row in rows]


def load_threshold(compare_json: Path, policy: str, stage: str) -> float:
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    for row in payload.get("rows", []):
        if row.get("stage") == stage and row.get("policy") == policy and row.get("status") == "ok":
            threshold = row.get("threshold")
            if threshold is None:
                break
            return float(threshold)
    raise ValueError(f"Threshold not found for stage={stage} policy={policy} in {compare_json}")


def load_pil_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def imagenet_norm_tensors(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std


def empty_preprocess_phases() -> Dict[str, float]:
    return {
        "image_load_decode_ms": 0.0,
        "transform_normalize_ms": 0.0,
        "h2d_copy_ms": 0.0,
        "pil_to_tensor_cpu_ms": 0.0,
        "raw_h2d_copy_ms": 0.0,
        "gpu_cast_scale_ms": 0.0,
        "gpu_resize_ms": 0.0,
        "gpu_center_crop_ms": 0.0,
        "gpu_normalize_ms": 0.0,
        "preprocess_total_ms": 0.0,
    }


def pil_to_chw_uint8(pil: Image.Image) -> torch.Tensor:
    array = np.array(pil, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_short_side_tensor(batch: torch.Tensor, size: int = 224) -> torch.Tensor:
    _, _, height, width = batch.shape
    if height == size and width == size:
        return batch
    if height < width:
        new_height = size
        new_width = int(round(width * size / height))
    else:
        new_width = size
        new_height = int(round(height * size / width))
    return F.interpolate(batch, size=(new_height, new_width), mode="bicubic", align_corners=False)


def center_crop_tensor(batch: torch.Tensor, size: int = 224) -> torch.Tensor:
    _, _, height, width = batch.shape
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return batch[:, :, top : top + size, left : left + size].contiguous()


def prepare_input_cpu_pil(path: Path, transform, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    phases = empty_preprocess_phases()
    pil, phases["image_load_decode_ms"] = timed(device, lambda: load_pil_image(path))
    tensor_cpu, phases["transform_normalize_ms"] = timed(device, lambda: transform(pil).unsqueeze(0))
    tensor_gpu, phases["h2d_copy_ms"] = timed(device, lambda: tensor_cpu.to(device))
    phases["preprocess_total_ms"] = (
        phases["image_load_decode_ms"] + phases["transform_normalize_ms"] + phases["h2d_copy_ms"]
    )
    return tensor_gpu, phases


def prepare_input_gpu_tensor(
    path: Path,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    phases = empty_preprocess_phases()
    pil, phases["image_load_decode_ms"] = timed(device, lambda: load_pil_image(path))
    raw_cpu, phases["pil_to_tensor_cpu_ms"] = timed(device, lambda: pil_to_chw_uint8(pil))
    raw_gpu, phases["raw_h2d_copy_ms"] = timed(device, lambda: raw_cpu.to(device))
    phases["h2d_copy_ms"] = phases["raw_h2d_copy_ms"]
    batch, phases["gpu_cast_scale_ms"] = timed(
        device, lambda: raw_gpu.unsqueeze(0).to(dtype=torch.float32).div(255.0)
    )
    batch, phases["gpu_resize_ms"] = timed(device, lambda: resize_short_side_tensor(batch, 224))
    batch, phases["gpu_center_crop_ms"] = timed(device, lambda: center_crop_tensor(batch, 224))
    batch, phases["gpu_normalize_ms"] = timed(device, lambda: (batch - norm_mean) / norm_std)
    phases["transform_normalize_ms"] = (
        phases["pil_to_tensor_cpu_ms"]
        + phases["gpu_cast_scale_ms"]
        + phases["gpu_resize_ms"]
        + phases["gpu_center_crop_ms"]
        + phases["gpu_normalize_ms"]
    )
    phases["preprocess_total_ms"] = (
        phases["image_load_decode_ms"] + phases["transform_normalize_ms"] + phases["h2d_copy_ms"]
    )
    return batch, phases


def prepare_input_gpu_tensor_uint8(
    path: Path,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PIL-equivalent GPU path: resize/crop in UINT8 domain on GPU, cast to float
    AFTER crop. Same coarse timing breakdown as prepare_input_gpu_tensor so the
    decoded_image_to_threshold / preprocess_total comparison stays apples-to-apples.
    Uses torchvision TF.resize bicubic+antialias on a CUDA uint8 tensor.
    """
    phases = empty_preprocess_phases()
    pil, phases["image_load_decode_ms"] = timed(device, lambda: load_pil_image(path))
    raw_cpu, phases["pil_to_tensor_cpu_ms"] = timed(device, lambda: pil_to_chw_uint8(pil))
    raw_gpu, phases["raw_h2d_copy_ms"] = timed(device, lambda: raw_cpu.to(device))
    phases["h2d_copy_ms"] = phases["raw_h2d_copy_ms"]
    u8, phases["gpu_resize_ms"] = timed(
        device, lambda: TF.resize(raw_gpu, [224], interpolation=InterpolationMode.BICUBIC, antialias=True)
    )
    u8, phases["gpu_center_crop_ms"] = timed(device, lambda: TF.center_crop(u8, [224, 224]))
    batch, phases["gpu_cast_scale_ms"] = timed(device, lambda: u8.to(dtype=torch.float32).div(255.0).unsqueeze(0))
    batch, phases["gpu_normalize_ms"] = timed(device, lambda: (batch - norm_mean) / norm_std)
    phases["transform_normalize_ms"] = (
        phases["pil_to_tensor_cpu_ms"]
        + phases["gpu_resize_ms"]
        + phases["gpu_center_crop_ms"]
        + phases["gpu_cast_scale_ms"]
        + phases["gpu_normalize_ms"]
    )
    phases["preprocess_total_ms"] = (
        phases["image_load_decode_ms"] + phases["transform_normalize_ms"] + phases["h2d_copy_ms"]
    )
    return batch, phases


def validate_encoder_input(tensor: torch.Tensor, device: torch.device, path: Path) -> None:
    expected_shape = (1, 3, 224, 224)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Invalid encoder input shape for {path}: got {tuple(tensor.shape)}, expected {expected_shape}")
    if tensor.device != device:
        raise ValueError(f"Invalid encoder input device for {path}: got {tensor.device}, expected {device}")


def prepare_input(
    path: Path,
    preprocess_backend: str,
    transform,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if preprocess_backend == "cpu_pil":
        return prepare_input_cpu_pil(path, transform, device)
    if preprocess_backend == "gpu_tensor_uint8":
        return prepare_input_gpu_tensor_uint8(path, device, norm_mean, norm_std)
    return prepare_input_gpu_tensor(path, device, norm_mean, norm_std)


def build_adp_reference_bank(
    encoder,
    ref_paths: Sequence[Path],
    preprocess_backend: str,
    transform,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    num_ref: int,
):
    selected = list(ref_paths)[:num_ref]
    if not selected:
        raise ValueError("No reference images found for ADP bank")
    refs = None
    with torch.no_grad():
        for path in selected:
            batch, _ = prepare_input(path, preprocess_backend, transform, device, norm_mean, norm_std)
            validate_encoder_input(batch, device, path)
            features = encode_multiscale(encoder, batch)
            flat = [x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).contiguous().detach() for x in features]
            if refs is None:
                refs = [[] for _ in flat]
            for idx, item in enumerate(flat):
                refs[idx].append(item)
    refs = [torch.cat(items, dim=0).to(device) for items in refs]
    refs_norm = [F.normalize(ref, p=2, dim=1) for ref in refs]
    return selected, refs, refs_norm


def match_reference_features_cached(
    features: Sequence[torch.Tensor],
    refs: Sequence[torch.Tensor],
    refs_norm: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    matched = []
    for feature, ref, ref_n in zip(features, refs, refs_norm):
        b, c, h, w = feature.shape
        flat = feature.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
        flat_n = F.normalize(flat, p=2, dim=1)
        index = torch.argmax(flat_n @ ref_n.T, dim=1)
        picked = ref[index].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        matched.append(picked)
    return matched


def build_ahl_reference_bank(
    ref_paths: Sequence[Path],
    preprocess_backend: str,
    transform,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    encoder,
    projector,
    refs: Sequence[torch.Tensor],
    refs_norm: Sequence[torch.Tensor],
    limit: int,
):
    ref_feature_list: List[torch.Tensor] = []
    ref_scale_list: List[torch.Tensor] = []
    with torch.no_grad():
        for path in list(ref_paths)[:limit]:
            tensor, _ = prepare_input(path, preprocess_backend, transform, device, norm_mean, norm_std)
            validate_encoder_input(tensor, device, path)
            features = encode_multiscale(encoder, tensor)
            matched = match_reference_features_cached(features, refs, refs_norm)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            feature, feature_scale = compress_four_to_two(projected)
            ref_feature_list.append(feature.detach())
            ref_scale_list.append(feature_scale.detach())
    return torch.cat(ref_feature_list, dim=0), torch.cat(ref_scale_list, dim=0)


def combine_dra_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    score = -1 * outputs[0]
    for item in outputs[1:]:
        score = score + item
    return score


def phase_stats(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, float]:
    vals = np.array([float(row[key]) for row in rows], dtype=float)
    return {
        "mean_ms": float(np.mean(vals)),
        "median_ms": float(np.percentile(vals, 50)),
        "p90_ms": float(np.percentile(vals, 90)),
        "p95_ms": float(np.percentile(vals, 95)),
        "trimmed_mean_ms": float(np.mean(np.sort(vals)[max(1, int(len(vals) * 0.05)):-max(1, int(len(vals) * 0.05))]))
        if len(vals) >= 20
        else float(np.mean(vals)),
        "max_ms": float(np.max(vals)),
    }


def summarize(rows: Sequence[SampleResult], threshold: float) -> Dict[str, float]:
    scores = np.array([row.score for row in rows], dtype=float)
    labels = np.array([row.label for row in rows], dtype=int)
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.preprocess_backend.startswith("gpu_tensor") and device.type != "cuda":
        raise RuntimeError(f"--preprocess-backend {args.preprocess_backend} requires an available CUDA device")
    add_adpretrain_to_path(str(args.adpretrain_root))
    if str(AHL_ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(AHL_ROOT))

    from modeling.DRA_AHL import DRA

    manifest = load_stage_manifest(args.stage_root, args.alias)
    class_root = Path(manifest["stage_class_root"])
    ref_paths = collect_role_paths(manifest, "train_normal")
    calib_normal_paths = collect_role_paths(manifest, "calib_normal")
    calib_anomaly_paths = collect_role_paths(manifest, "calib_anomaly")
    test_normal_paths = collect_role_paths(manifest, "test_normal")
    test_anomaly_paths = collect_role_paths(manifest, "test_anomaly")

    threshold = load_threshold(args.threshold_compare_json, args.threshold_policy, args.stage)
    transform = build_transform("dinov2-large")
    norm_mean, norm_std = imagenet_norm_tensors(device)
    encoder = make_encoder("dinov2-large", str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)

    cfg = type("Cfg", (), {"total_heads": 4, "n_scales": 2, "nRef": args.n_ref})()
    ahl_model = DRA(cfg, backbone="resnet18").to(device)
    state_dict = torch.load(args.ahl_weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    ahl_model.load_state_dict(state_dict)
    ahl_model.eval()

    setup_start = time.perf_counter()
    _, adp_refs, adp_refs_norm = build_adp_reference_bank(
        encoder,
        ref_paths,
        args.preprocess_backend,
        transform,
        device,
        norm_mean,
        norm_std,
        args.n_ref,
    )
    ahl_ref_feature, ahl_ref_scale = build_ahl_reference_bank(
        ref_paths,
        args.preprocess_backend,
        transform,
        device,
        norm_mean,
        norm_std,
        encoder,
        projector,
        adp_refs,
        adp_refs_norm,
        args.n_ref,
    )
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def run_sample(path: Path, role: str, label: int, timed_pass: bool) -> SampleResult:
        total_start = time.perf_counter()

        tensor_gpu, phase_ms = prepare_input(
            path,
            args.preprocess_backend,
            transform,
            device,
            norm_mean,
            norm_std,
        )
        validate_encoder_input(tensor_gpu, device, path)

        features, phase_ms["encoder_ms"] = timed(device, lambda: encode_multiscale(encoder, tensor_gpu))
        matched, phase_ms["adp_reference_match_ms"] = timed(
            device, lambda: match_reference_features_cached(features, adp_refs, adp_refs_norm)
        )
        residual, phase_ms["residual_ms"] = timed(device, lambda: residual_features(features, matched))
        projected, phase_ms["projector_ms"] = timed(device, lambda: projector(*residual))
        (feature, feature_scale), phase_ms["compress_ms"] = timed(
            device, lambda: compress_four_to_two(projected)
        )

        (image, image_scale), phase_ms["ahl_reference_attach_ms"] = timed(
            device,
            lambda: (
                torch.cat([ahl_ref_feature, feature], dim=0),
                torch.cat([ahl_ref_scale, feature_scale], dim=0),
            ),
        )
        outputs, phase_ms["ahl_forward_ms"] = timed(
            device,
            lambda: ahl_model(image=image, image_scale=image_scale, label=None, var=ahl_model.parameters()),
        )
        score, phase_ms["score_postprocess_ms"] = timed(device, lambda: combine_dra_outputs(outputs))
        _, phase_ms["threshold_ms"] = timed(device, lambda: bool(float(score.item()) >= threshold))
        phase_ms["gpu_inference_only_ms"] = sum(
            phase_ms[key]
            for key in [
                "encoder_ms",
                "adp_reference_match_ms",
                "residual_ms",
                "projector_ms",
                "compress_ms",
                "ahl_reference_attach_ms",
                "ahl_forward_ms",
                "score_postprocess_ms",
                "threshold_ms",
            ]
        )
        phase_ms["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        pred = int(float(score.item()) >= threshold)
        return SampleResult(
            rel_path=str(path.relative_to(class_root)),
            role=role,
            label=label,
            score=float(score.item()),
            pred=pred,
            phases_ms=phase_ms,
        )

    def iter_paths_with_labels(paths: Sequence[Path], role: str, label: int):
        for path in paths:
            yield path, role, label

    calib_items = list(iter_paths_with_labels(calib_normal_paths, "calib_normal", 0)) + list(
        iter_paths_with_labels(calib_anomaly_paths, "calib_anomaly", 1)
    )
    test_items = list(iter_paths_with_labels(test_normal_paths, "test_normal", 0)) + list(
        iter_paths_with_labels(test_anomaly_paths, "test_anomaly", 1)
    )

    calib_rows: List[SampleResult] = []
    for path, role, label in calib_items:
        calib_rows.append(run_sample(path, role, label, timed_pass=False))

    test_rows: List[SampleResult] = []
    timed_rows: List[SampleResult] = []
    for idx, (path, role, label) in enumerate(test_items):
        row = run_sample(path, role, label, timed_pass=True)
        test_rows.append(row)
        if idx >= args.warmup:
            timed_rows.append(row)

    if args.max_calib is not None:
        calib_rows = calib_rows[: args.max_calib]
    if args.max_test is not None:
        test_rows = test_rows[: args.max_test]
        timed_rows = [row for i, row in enumerate(timed_rows) if i < args.max_test]

    calib_metrics = summarize(calib_rows, threshold)
    test_metrics = summarize(test_rows, threshold)
    timed_metrics = summarize(timed_rows, threshold)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = summary_dir / "s8_adp_to_ahl_e2e_rows.csv"
    with rows_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rel_path",
                "role",
                "label",
                "score",
                "pred",
                "preprocess_backend",
                "image_load_decode_ms",
                "transform_normalize_ms",
                "h2d_copy_ms",
                "pil_to_tensor_cpu_ms",
                "raw_h2d_copy_ms",
                "gpu_cast_scale_ms",
                "gpu_resize_ms",
                "gpu_center_crop_ms",
                "gpu_normalize_ms",
                "preprocess_total_ms",
                "encoder_ms",
                "adp_reference_match_ms",
                "residual_ms",
                "projector_ms",
                "compress_ms",
                "ahl_reference_attach_ms",
                "ahl_forward_ms",
                "score_postprocess_ms",
                "threshold_ms",
                "gpu_inference_only_ms",
                "total_ms",
            ],
        )
        writer.writeheader()
        for row in test_rows:
            payload = {
                "rel_path": row.rel_path,
                "role": row.role,
                "label": row.label,
                "score": row.score,
                "pred": row.pred,
                "preprocess_backend": args.preprocess_backend,
            }
            payload.update(row.phases_ms)
            writer.writerow(payload)

    def rows_to_dicts(rows: Sequence[SampleResult]) -> List[Dict]:
        out = []
        for row in rows:
            payload = {
                "rel_path": row.rel_path,
                "role": row.role,
                "label": row.label,
                "score": row.score,
                "pred": row.pred,
                "preprocess_backend": args.preprocess_backend,
            }
            payload.update(row.phases_ms)
            out.append(payload)
        return out

    phase_keys = [
        "image_load_decode_ms",
        "transform_normalize_ms",
        "h2d_copy_ms",
        "pil_to_tensor_cpu_ms",
        "raw_h2d_copy_ms",
        "gpu_cast_scale_ms",
        "gpu_resize_ms",
        "gpu_center_crop_ms",
        "gpu_normalize_ms",
        "preprocess_total_ms",
        "encoder_ms",
        "adp_reference_match_ms",
        "residual_ms",
        "projector_ms",
        "compress_ms",
        "ahl_reference_attach_ms",
        "ahl_forward_ms",
        "score_postprocess_ms",
        "threshold_ms",
        "gpu_inference_only_ms",
        "total_ms",
    ]

    phase_summary = {key: phase_stats(rows_to_dicts(timed_rows), key) for key in phase_keys}
    steady_count = len(timed_rows)

    summary_payload = {
        "stage": args.stage,
        "alias": args.alias,
        "stage_root": str(args.stage_root),
        "class_root": str(class_root),
        "threshold_policy": args.threshold_policy,
        "threshold_source": str(args.threshold_compare_json),
        "threshold": threshold,
        "preprocess_backend": args.preprocess_backend,
        "ref_count": len(ref_paths[: args.n_ref]),
        "setup_ms": setup_ms,
        "calib_count": len(calib_rows),
        "test_count": len(test_rows),
        "steady_count": steady_count,
        "calib_metrics": calib_metrics,
        "test_metrics": test_metrics,
        "timed_metrics": timed_metrics,
        "phase_summary": phase_summary,
        "notes": [
            "Reference banks are preloaded before timed measurement.",
            "Calibration is used only to verify the fixed threshold from the existing S8 calibration output.",
            "Timed statistics are collected on test samples after warmup.",
            "cpu_pil keeps the legacy PIL/torchvision CPU transform path.",
            "gpu_tensor keeps image decode on CPU, then runs tensor cast/resize/crop/normalize on GPU.",
            "preprocess_total_ms includes image_load_decode_ms + transform_normalize_ms + h2d_copy_ms.",
            "gpu_inference_only_ms includes ADP/AHL model-side phases after preprocessing.",
        ],
    }
    (summary_dir / "s8_adp_to_ahl_e2e_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# S8 ADP -> AHL end-to-end benchmark",
        "",
        "## Key Results",
        "",
        f"- threshold policy: `{args.threshold_policy}`",
        f"- threshold: `{threshold:.12f}`",
        f"- preprocess backend: `{args.preprocess_backend}`",
        f"- setup time: `{setup_ms:.2f} ms`",
        f"- calib count: `{len(calib_rows)}`",
        f"- test count: `{len(test_rows)}`",
        f"- steady timed count: `{steady_count}`",
        "",
        "### Calibration metrics",
        "",
        f"- accuracy: `{calib_metrics['accuracy']:.4f}`",
        f"- precision: `{calib_metrics['precision']:.4f}`",
        f"- recall: `{calib_metrics['recall']:.4f}`",
        f"- f1: `{calib_metrics['f1']:.4f}`",
        f"- auc_roc: `{calib_metrics['auc_roc']:.4f}`",
        f"- auc_pr: `{calib_metrics['auc_pr']:.4f}`",
        "",
        "### Test metrics",
        "",
        f"- accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- precision: `{test_metrics['precision']:.4f}`",
        f"- recall: `{test_metrics['recall']:.4f}`",
        f"- f1: `{test_metrics['f1']:.4f}`",
        f"- auc_roc: `{test_metrics['auc_roc']:.4f}`",
        f"- auc_pr: `{test_metrics['auc_pr']:.4f}`",
        "",
        "## Steady latency summary",
        "",
        "| phase | mean | median | P90 | P95 | trimmed mean | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in phase_keys:
        s = phase_summary[key]
        md_lines.append(
            f"| {key} | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['p90_ms']:.2f} | {s['p95_ms']:.2f} | {s['trimmed_mean_ms']:.2f} | {s['max_ms']:.2f} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This benchmark keeps raw image decode, transform, ADP encoder, ADP reference matching, residual, projector, compress, AHL reference attach, AHL forward, score postprocess, and threshold in one process.",
            "- `cpu_pil` is the legacy path: PIL decode plus CPU torchvision resize/crop/tensor/normalize, then H2D copy.",
            "- `gpu_tensor` keeps PIL decode on CPU, then moves the raw uint8 tensor to GPU and runs cast/scale, resize, center crop, and normalize on GPU.",
            "- `preprocess_total_ms` is image decode plus transform/normalize plus H2D; `gpu_inference_only_ms` is the ADP/AHL model-side path after preprocessing.",
            "- `.npy` feature writing is not part of the timed path.",
            "- Reference banks are preloaded once before timing.",
            "- Timed statistics are taken on test samples after warmup, so the report is a steady-state end-to-end latency, not a cold-start number.",
            "",
            "## Files",
            "",
            f"- rows csv: `{rows_csv}`",
            f"- summary json: `{summary_dir / 's8_adp_to_ahl_e2e_summary.json'}`",
        ]
    )
    (summary_dir / "s8_adp_to_ahl_e2e_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"WROTE {summary_dir / 's8_adp_to_ahl_e2e_summary.md'}")
    print(f"WROTE {summary_dir / 's8_adp_to_ahl_e2e_summary.json'}")
    print(f"WROTE {rows_csv}")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
