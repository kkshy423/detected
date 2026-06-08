#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU preprocessing equivalence-fix benchmark for S8 ADP->AHL."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from common import add_adpretrain_to_path, compress_four_to_two, encode_multiscale, make_encoder, make_projector, residual_features
from threshold_policies import choose_thresholds


BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
AHL_ROOT = Path("/ghome/huangjd/code/detected/AHL")
DEFAULT_STAGE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_plain_dino_large_norm_val49/S8"
)
DEFAULT_ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
DEFAULT_PROJECTOR = Path("/ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth")
DEFAULT_AHL_WEIGHTS = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages/S8/ahl/models_qiumianxiepai_ctest.pkl"
)
DEFAULT_EXTERNAL_THRESHOLD_COMPARE = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_time_profile_threshold_compare/policy_compare.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_gpu_preprocess_equivalence_fix_v1"
)
DEFAULT_ALIAS = "models_qiumianxiepai"
DEFAULT_STAGE = "S8"
DEFAULT_POLICY = "strategy_mild_stage_v2_1_safe"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
TASK_NAME = "20260607_gpu_preprocess_equivalence_fix_v1"
CPU_BACKEND = "cpu_pil"
BACKENDS = [
    "cpu_pil",
    "gpu_tensor_current",
    "gpu_tensor_torchvision_aa_true",
    "gpu_tensor_torchvision_aa_false",
    "gpu_tensor_uint8_aa_true",
]
GPU_BACKENDS = [x for x in BACKENDS if x != CPU_BACKEND]
SCENARIOS = ["query_only_diff", "ref_and_query_diff"]


@dataclass
class Item:
    path: Path
    rel_path: str
    role: str
    split: str
    label: int


@dataclass
class BackendBank:
    refs: Sequence[torch.Tensor]
    refs_norm: Sequence[torch.Tensor]
    ahl_ref_feature: torch.Tensor
    ahl_ref_scale: torch.Tensor
    setup_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--adpretrain-root", type=Path, default=DEFAULT_ADPRETRAIN_ROOT)
    parser.add_argument("--projector-checkpoint", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--ahl-weights", type=Path, default=DEFAULT_AHL_WEIGHTS)
    parser.add_argument("--external-threshold-compare-json", type=Path, default=DEFAULT_EXTERNAL_THRESHOLD_COMPARE)
    parser.add_argument("--threshold-policy", default=DEFAULT_POLICY)
    parser.add_argument("--fixed-threshold-source", choices=("cpu_calib", "external_compare_json"), default="cpu_calib")
    parser.add_argument("--n-ref", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-calib", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
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


def collect_items(manifest: Dict, role: str, split: str, label: int) -> List[Item]:
    class_root = Path(manifest["stage_class_root"])
    out = []
    for row in manifest["mappings"]:
        if row["role"] != role:
            continue
        path = class_root / row["stage_rel"]
        out.append(Item(path=path, rel_path=str(path.relative_to(class_root)), role=role, split=split, label=label))
    return out


def load_pil_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_external_threshold(compare_json: Path, policy: str, stage: str) -> Optional[float]:
    if not compare_json.exists():
        return None
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    for row in payload.get("rows", []):
        if row.get("stage") == stage and row.get("policy") == policy and row.get("status") == "ok":
            threshold = row.get("threshold")
            return None if threshold is None else float(threshold)
    return None


def resize_supports_antialias() -> bool:
    return "antialias" in inspect.signature(TF.resize).parameters


def resize_tf(image_or_tensor, size, interpolation, antialias: bool):
    if resize_supports_antialias():
        return TF.resize(image_or_tensor, size, interpolation=interpolation, antialias=antialias)
    return TF.resize(image_or_tensor, size, interpolation=interpolation)


def pil_to_chw_uint8(pil: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(pil, copy=True)).permute(2, 0, 1).contiguous()


def prepare_cpu_from_pil(pil: Image.Image, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    phases = zero_phase_dict()
    resized, phases["cpu_resize_ms"] = timed(
        device, lambda: resize_tf(pil, [IMAGE_SIZE], InterpolationMode.BICUBIC, True)
    )
    cropped, phases["cpu_crop_ms"] = timed(device, lambda: TF.center_crop(resized, [IMAGE_SIZE, IMAGE_SIZE]))
    tensor_cpu, phases["cpu_to_tensor_ms"] = timed(device, lambda: TF.to_tensor(cropped).unsqueeze(0))
    norm_cpu, phases["cpu_normalize_ms"] = timed(
        device, lambda: TF.normalize(tensor_cpu.squeeze(0), mean=MEAN, std=STD).unsqueeze(0)
    )
    tensor_gpu, phases["h2d_copy_ms"] = timed(device, lambda: norm_cpu.to(device))
    phases["preprocess_total_ms"] = (
        phases["cpu_resize_ms"]
        + phases["cpu_crop_ms"]
        + phases["cpu_to_tensor_ms"]
        + phases["cpu_normalize_ms"]
        + phases["h2d_copy_ms"]
    )
    return tensor_gpu, phases, norm_cpu.detach().cpu()


def prepare_gpu_current_from_pil(
    pil: Image.Image,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Legacy GPU tensor path from the previous benchmark."""
    phases = zero_phase_dict()
    raw_cpu, phases["pil_to_tensor_cpu_ms"] = timed(device, lambda: pil_to_chw_uint8(pil))
    raw_gpu, phases["raw_h2d_copy_ms"] = timed(device, lambda: raw_cpu.to(device))
    batch, phases["gpu_cast_scale_ms"] = timed(
        device, lambda: raw_gpu.unsqueeze(0).to(dtype=torch.float32).div(255.0)
    )
    batch, phases["gpu_resize_ms"] = timed(
        device, lambda: resize_tf(batch, [IMAGE_SIZE], InterpolationMode.BICUBIC, True)
    )
    batch, phases["gpu_center_crop_ms"] = timed(device, lambda: TF.center_crop(batch, [IMAGE_SIZE, IMAGE_SIZE]))
    batch, phases["gpu_normalize_ms"] = timed(device, lambda: (batch - mean_gpu) / std_gpu)
    phases["preprocess_total_ms"] = (
        phases["pil_to_tensor_cpu_ms"]
        + phases["raw_h2d_copy_ms"]
        + phases["gpu_cast_scale_ms"]
        + phases["gpu_resize_ms"]
        + phases["gpu_center_crop_ms"]
        + phases["gpu_normalize_ms"]
    )
    return batch, phases, batch.detach().cpu()


def prepare_gpu_torchvision_from_pil(
    pil: Image.Image,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    antialias: bool,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Torchvision tensor resize/crop candidate using a single CHW tensor."""
    phases = zero_phase_dict()
    raw_cpu, phases["pil_to_tensor_cpu_ms"] = timed(device, lambda: pil_to_chw_uint8(pil))
    raw_gpu, phases["raw_h2d_copy_ms"] = timed(device, lambda: raw_cpu.to(device))
    chw, phases["gpu_cast_scale_ms"] = timed(device, lambda: raw_gpu.to(dtype=torch.float32).div(255.0))
    chw, phases["gpu_resize_ms"] = timed(
        device, lambda: resize_tf(chw, [IMAGE_SIZE], InterpolationMode.BICUBIC, antialias)
    )
    chw, phases["gpu_center_crop_ms"] = timed(device, lambda: TF.center_crop(chw, [IMAGE_SIZE, IMAGE_SIZE]))
    batch, phases["gpu_normalize_ms"] = timed(device, lambda: (chw.unsqueeze(0) - mean_gpu) / std_gpu)
    phases["preprocess_total_ms"] = (
        phases["pil_to_tensor_cpu_ms"]
        + phases["raw_h2d_copy_ms"]
        + phases["gpu_cast_scale_ms"]
        + phases["gpu_resize_ms"]
        + phases["gpu_center_crop_ms"]
        + phases["gpu_normalize_ms"]
    )
    return batch, phases, batch.detach().cpu()


def prepare_gpu_uint8_from_pil(
    pil: Image.Image,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    antialias: bool,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """PIL-equivalent GPU path: resize/crop in the UINT8 domain on GPU, cast to
    float AFTER resize.

    PIL bicubic runs the interpolation on uint8 pixels and quantises the result
    back to uint8. The previous GPU path divided by 255 to float BEFORE resizing,
    so bicubic overshoot survived as continuous values instead of being clipped
    and rounded. Keeping the resize in uint8 (torchvision computes bicubic on
    uint8 tensors in higher precision then rounds/clamps back to uint8, matching
    PIL) reproduces the CPU-PIL tensor far more closely. cast_scale is therefore
    timed AFTER crop here, with gpu_resize/gpu_center_crop measuring the uint8 ops.
    """
    phases = zero_phase_dict()
    raw_cpu, phases["pil_to_tensor_cpu_ms"] = timed(device, lambda: pil_to_chw_uint8(pil))
    raw_gpu, phases["raw_h2d_copy_ms"] = timed(device, lambda: raw_cpu.to(device))
    chw_u8, phases["gpu_resize_ms"] = timed(
        device, lambda: resize_tf(raw_gpu, [IMAGE_SIZE], InterpolationMode.BICUBIC, antialias)
    )
    chw_u8, phases["gpu_center_crop_ms"] = timed(device, lambda: TF.center_crop(chw_u8, [IMAGE_SIZE, IMAGE_SIZE]))
    chw, phases["gpu_cast_scale_ms"] = timed(device, lambda: chw_u8.to(dtype=torch.float32).div(255.0))
    batch, phases["gpu_normalize_ms"] = timed(device, lambda: (chw.unsqueeze(0) - mean_gpu) / std_gpu)
    phases["preprocess_total_ms"] = (
        phases["pil_to_tensor_cpu_ms"]
        + phases["raw_h2d_copy_ms"]
        + phases["gpu_resize_ms"]
        + phases["gpu_center_crop_ms"]
        + phases["gpu_cast_scale_ms"]
        + phases["gpu_normalize_ms"]
    )
    return batch, phases, batch.detach().cpu()


def zero_phase_dict() -> Dict[str, float]:
    return {
        "image_load_decode_ms": 0.0,
        "cpu_resize_ms": 0.0,
        "cpu_crop_ms": 0.0,
        "cpu_to_tensor_ms": 0.0,
        "cpu_normalize_ms": 0.0,
        "h2d_copy_ms": 0.0,
        "pil_to_tensor_cpu_ms": 0.0,
        "raw_h2d_copy_ms": 0.0,
        "gpu_cast_scale_ms": 0.0,
        "gpu_resize_ms": 0.0,
        "gpu_center_crop_ms": 0.0,
        "gpu_normalize_ms": 0.0,
        "preprocess_total_ms": 0.0,
        "encoder_ms": 0.0,
        "adp_reference_match_ms": 0.0,
        "residual_ms": 0.0,
        "projector_ms": 0.0,
        "compress_ms": 0.0,
        "ahl_reference_attach_ms": 0.0,
        "ahl_forward_ms": 0.0,
        "score_postprocess_ms": 0.0,
        "threshold_ms": 0.0,
        "tensor_to_threshold_ms": 0.0,
        "decoded_image_to_threshold_ms": 0.0,
        "file_end_to_end_ms": 0.0,
    }


PREPROCESS_PHASE_KEYS = [
    "cpu_resize_ms",
    "cpu_crop_ms",
    "cpu_to_tensor_ms",
    "cpu_normalize_ms",
    "h2d_copy_ms",
    "pil_to_tensor_cpu_ms",
    "raw_h2d_copy_ms",
    "gpu_cast_scale_ms",
    "gpu_resize_ms",
    "gpu_center_crop_ms",
    "gpu_normalize_ms",
    "preprocess_total_ms",
]


def validate_encoder_input(tensor: torch.Tensor, device: torch.device, rel_path: str) -> None:
    expected_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Invalid encoder input shape for {rel_path}: {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"Invalid encoder input device for {rel_path}: {tensor.device} != {device}")


def flatten_feature_map(feature: torch.Tensor) -> torch.Tensor:
    _, c, _, _ = feature.shape
    return feature.permute(0, 2, 3, 1).reshape(-1, c).contiguous()


def match_reference_features_cached(
    features: Sequence[torch.Tensor],
    refs: Sequence[torch.Tensor],
    refs_norm: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    matched = []
    for feature, ref, ref_n in zip(features, refs, refs_norm):
        b, c, h, w = feature.shape
        flat = flatten_feature_map(feature)
        flat_n = F.normalize(flat, p=2, dim=1)
        index = torch.argmax(flat_n @ ref_n.T, dim=1)
        picked = ref[index].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        matched.append(picked)
    return matched


def combine_dra_outputs(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    score = -1 * outputs[0]
    for item in outputs[1:]:
        score = score + item
    return score


def prepare_backend_from_pil(
    backend: str,
    pil: Image.Image,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    if backend == CPU_BACKEND:
        return prepare_cpu_from_pil(pil, device)
    if backend == "gpu_tensor_current":
        return prepare_gpu_current_from_pil(pil, device, mean_gpu, std_gpu)
    if backend == "gpu_tensor_torchvision_aa_true":
        return prepare_gpu_torchvision_from_pil(pil, device, mean_gpu, std_gpu, antialias=True)
    if backend == "gpu_tensor_torchvision_aa_false":
        return prepare_gpu_torchvision_from_pil(pil, device, mean_gpu, std_gpu, antialias=False)
    if backend == "gpu_tensor_uint8_aa_true":
        return prepare_gpu_uint8_from_pil(pil, device, mean_gpu, std_gpu, antialias=True)
    raise ValueError(backend)


def build_backend_bank(
    backend: str,
    ref_items: Sequence[Item],
    encoder,
    projector,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    n_ref: int,
) -> BackendBank:
    selected = list(ref_items)[:n_ref]
    if not selected:
        raise ValueError("No reference items")
    start = time.perf_counter()
    refs = None
    with torch.no_grad():
        for item in selected:
            pil = load_pil_image(item.path)
            tensor, _, _ = prepare_backend_from_pil(backend, pil, device, mean_gpu, std_gpu)
            validate_encoder_input(tensor, device, item.rel_path)
            features = encode_multiscale(encoder, tensor)
            flat = [flatten_feature_map(x).detach() for x in features]
            if refs is None:
                refs = [[] for _ in flat]
            for idx, fmap in enumerate(flat):
                refs[idx].append(fmap)
    refs = [torch.cat(items, dim=0).to(device) for items in refs]
    refs_norm = [F.normalize(ref, p=2, dim=1) for ref in refs]
    ref_feature_list = []
    ref_scale_list = []
    with torch.no_grad():
        for item in selected:
            pil = load_pil_image(item.path)
            tensor, _, _ = prepare_backend_from_pil(backend, pil, device, mean_gpu, std_gpu)
            features = encode_multiscale(encoder, tensor)
            matched = match_reference_features_cached(features, refs, refs_norm)
            residual = residual_features(features, matched)
            projected = projector(*residual)
            feature, feature_scale = compress_four_to_two(projected)
            ref_feature_list.append(feature.detach())
            ref_scale_list.append(feature_scale.detach())
    setup_ms = (time.perf_counter() - start) * 1000.0
    return BackendBank(
        refs=refs,
        refs_norm=refs_norm,
        ahl_ref_feature=torch.cat(ref_feature_list, dim=0),
        ahl_ref_scale=torch.cat(ref_scale_list, dim=0),
        setup_ms=setup_ms,
    )


def score_tensor(
    tensor: torch.Tensor,
    bank: BackendBank,
    encoder,
    projector,
    ahl_model,
    device: torch.device,
    threshold: float,
) -> Tuple[float, int, Dict[str, float]]:
    phases = zero_phase_dict()
    validate_encoder_input(tensor, device, "tensor")
    with torch.no_grad():
        features, phases["encoder_ms"] = timed(device, lambda: encode_multiscale(encoder, tensor))
        matched, phases["adp_reference_match_ms"] = timed(
            device, lambda: match_reference_features_cached(features, bank.refs, bank.refs_norm)
        )
        residual, phases["residual_ms"] = timed(device, lambda: residual_features(features, matched))
        projected, phases["projector_ms"] = timed(device, lambda: projector(*residual))
        (feature, feature_scale), phases["compress_ms"] = timed(device, lambda: compress_four_to_two(projected))
        (image, image_scale), phases["ahl_reference_attach_ms"] = timed(
            device,
            lambda: (
                torch.cat([bank.ahl_ref_feature, feature], dim=0),
                torch.cat([bank.ahl_ref_scale, feature_scale], dim=0),
            ),
        )
        outputs, phases["ahl_forward_ms"] = timed(
            device,
            lambda: ahl_model(image=image, image_scale=image_scale, label=None, var=ahl_model.parameters()),
        )
        score_tensor_value, phases["score_postprocess_ms"] = timed(device, lambda: combine_dra_outputs(outputs))
        _, phases["threshold_ms"] = timed(device, lambda: bool(float(score_tensor_value.item()) >= threshold))
    score = float(score_tensor_value.item())
    pred = int(score >= threshold)
    phases["tensor_to_threshold_ms"] = sum(
        phases[k]
        for k in [
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
    return score, pred, phases


def tensor_diff_stats(cpu_tensor: torch.Tensor, gpu_tensor: torch.Tensor) -> Dict[str, float]:
    cpu_flat = cpu_tensor.reshape(-1).float()
    gpu_flat = gpu_tensor.reshape(-1).float()
    diff = torch.abs(cpu_flat - gpu_flat)
    return {
        "cpu_min": float(cpu_flat.min().item()),
        "cpu_max": float(cpu_flat.max().item()),
        "cpu_mean": float(cpu_flat.mean().item()),
        "cpu_std": float(cpu_flat.std(unbiased=False).item()),
        "gpu_min": float(gpu_flat.min().item()),
        "gpu_max": float(gpu_flat.max().item()),
        "gpu_mean": float(gpu_flat.mean().item()),
        "gpu_std": float(gpu_flat.std(unbiased=False).item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "p95_abs_diff": float(torch.quantile(diff, 0.95).item()),
        "p99_abs_diff": float(torch.quantile(diff, 0.99).item()),
        "cosine_similarity": float(F.cosine_similarity(cpu_flat, gpu_flat, dim=0).item()),
    }


def metrics_at(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    labels_np = np.array(labels, dtype=int)
    scores_np = np.array(scores, dtype=float)
    preds = (scores_np >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels_np == 1)))
    fp = int(np.sum((preds == 1) & (labels_np == 0)))
    tn = int(np.sum((preds == 0) & (labels_np == 0)))
    fn = int(np.sum((preds == 0) & (labels_np == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels_np) if len(labels_np) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(labels_np, scores_np)) if len(set(labels_np.tolist())) > 1 else 0.0,
        "auc_pr": float(average_precision_score(labels_np, scores_np)) if len(set(labels_np.tolist())) > 1 else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def phase_stats(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, float]:
    vals = np.array([float(r[key]) for r in rows], dtype=float)
    if vals.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def numeric_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = np.array(list(values), dtype=float)
    if vals.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_backend_threshold(policy: str, stage: str, labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    chosen = choose_thresholds(np.array(labels, dtype=int), np.array(scores, dtype=float), stage=stage)
    if policy not in chosen:
        raise KeyError(f"threshold policy {policy!r} not available; choices={sorted(chosen)}")
    return chosen[policy]


def fmt_metric(row: Dict[str, float]) -> str:
    return (
        f"| {row['backend']} | {row['threshold']:.6f} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | {row['auc_roc']:.4f} | {row['auc_pr']:.4f} | "
        f"{int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This equivalence benchmark requires CUDA because GPU tensor backends are evaluated.")

    add_adpretrain_to_path(str(args.adpretrain_root))
    if str(AHL_ROOT) not in sys.path:
        sys.path.insert(0, str(AHL_ROOT))
    from modeling.DRA_AHL import DRA

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_stage_manifest(args.stage_root, args.alias)
    train_items = collect_items(manifest, "train_normal", "train", 0)
    calib_items = collect_items(manifest, "calib_normal", "calib", 0) + collect_items(manifest, "calib_anomaly", "calib", 1)
    test_items = collect_items(manifest, "test_normal", "test", 0) + collect_items(manifest, "test_anomaly", "test", 1)
    if args.max_calib is not None:
        calib_items = calib_items[: args.max_calib]
    if args.max_test is not None:
        test_items = test_items[: args.max_test]
    all_items = calib_items + test_items
    backend_labels = {"calib": [x.label for x in calib_items], "test": [x.label for x in test_items]}

    mean_gpu = torch.tensor(MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    encoder = make_encoder("dinov2-large", str(args.adpretrain_root), device)
    projector = make_projector(encoder, str(args.adpretrain_root), str(args.projector_checkpoint), device)
    cfg = type("Cfg", (), {"total_heads": 4, "n_scales": 2, "nRef": args.n_ref})()
    ahl_model = DRA(cfg, backbone="resnet18").to(device)
    state_dict = torch.load(args.ahl_weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    ahl_model.load_state_dict(state_dict)
    ahl_model.eval()

    banks = {
        backend: build_backend_bank(backend, train_items, encoder, projector, device, mean_gpu, std_gpu, args.n_ref)
        for backend in BACKENDS
    }

    external_threshold = load_external_threshold(args.external_threshold_compare_json, args.threshold_policy, args.stage)
    placeholder_threshold = external_threshold if external_threshold is not None else 0.0

    backend_scores = {
        scenario: {backend: {"calib": [], "test": []} for backend in BACKENDS}
        for scenario in SCENARIOS
    }
    score_records = {
        scenario: {backend: [] for backend in BACKENDS}
        for scenario in SCENARIOS
    }
    tensor_diff_rows = {backend: [] for backend in GPU_BACKENDS}
    latency_records: List[Dict] = []

    score_tasks = [(scenario, backend) for scenario in SCENARIOS for backend in BACKENDS]

    for item_index, item in enumerate(all_items):
        pil, decode_ms = timed(device, lambda p=item.path: load_pil_image(p))

        prepare_order = BACKENDS[item_index % len(BACKENDS) :] + BACKENDS[: item_index % len(BACKENDS)]
        prepared = {}
        for prepare_position, backend in enumerate(prepare_order, start=1):
            tensor, pre_phases, tensor_cpu = prepare_backend_from_pil(backend, pil, device, mean_gpu, std_gpu)
            validate_encoder_input(tensor, device, item.rel_path)
            prepared[backend] = {
                "tensor": tensor,
                "pre_phases": pre_phases,
                "tensor_cpu": tensor_cpu,
                "prepare_position": prepare_position,
            }

        cpu_tensor_cpu = prepared[CPU_BACKEND]["tensor_cpu"]
        for backend in GPU_BACKENDS:
            diff_row = {
                "backend": backend,
                "rel_path": item.rel_path,
                "split": item.split,
                "role": item.role,
                "label": item.label,
            }
            diff_row.update(tensor_diff_stats(cpu_tensor_cpu, prepared[backend]["tensor_cpu"]))
            tensor_diff_rows[backend].append(diff_row)

        task_order = score_tasks[item_index % len(score_tasks) :] + score_tasks[: item_index % len(score_tasks)]
        for score_position, (scenario, backend) in enumerate(task_order, start=1):
            bank_backend = CPU_BACKEND if scenario == "query_only_diff" else backend
            score, _, score_phases = score_tensor(
                prepared[backend]["tensor"],
                banks[bank_backend],
                encoder,
                projector,
                ahl_model,
                device,
                placeholder_threshold,
            )
            backend_scores[scenario][backend][item.split].append(score)
            score_records[scenario][backend].append(
                {
                    "rel_path": item.rel_path,
                    "split": item.split,
                    "role": item.role,
                    "label": item.label,
                    "score": score,
                }
            )

            phases = zero_phase_dict()
            phases.update(score_phases)
            for key in PREPROCESS_PHASE_KEYS:
                phases[key] = prepared[backend]["pre_phases"].get(key, 0.0)
            phases["image_load_decode_ms"] = decode_ms
            phases["decoded_image_to_threshold_ms"] = phases["preprocess_total_ms"] + phases["tensor_to_threshold_ms"]
            phases["file_end_to_end_ms"] = phases["image_load_decode_ms"] + phases["decoded_image_to_threshold_ms"]
            latency_records.append(
                {
                    "scenario": scenario,
                    "backend": backend,
                    "ref_backend": bank_backend,
                    "rel_path": item.rel_path,
                    "split": item.split,
                    "role": item.role,
                    "label": item.label,
                    "prepare_position": prepared[backend]["prepare_position"],
                    "score_position": score_position,
                    "timed": int(item.split == "test" and item_index - len(calib_items) >= args.warmup),
                    **phases,
                }
            )

    cpu_threshold_row = choose_backend_threshold(
        args.threshold_policy,
        args.stage,
        backend_labels["calib"],
        backend_scores["query_only_diff"][CPU_BACKEND]["calib"],
    )
    fixed_threshold = (
        float(cpu_threshold_row["threshold"])
        if args.fixed_threshold_source == "cpu_calib"
        else external_threshold
    )
    if fixed_threshold is None:
        raise ValueError("--fixed-threshold-source external_compare_json requested, but external threshold was not found")

    tensor_summary = {
        backend: {
            key: numeric_stats([float(row[key]) for row in rows])
            for key in ["max_abs_diff", "mean_abs_diff", "p95_abs_diff", "p99_abs_diff", "cosine_similarity"]
        }
        for backend, rows in tensor_diff_rows.items()
    }

    def diff_stats_for(scenario: str, backend: str, split: Optional[str] = None) -> Dict[str, float]:
        cpu_rows = score_records[scenario][CPU_BACKEND]
        backend_rows = score_records[scenario][backend]
        vals = []
        changed = 0
        count = 0
        for cpu_row, backend_row in zip(cpu_rows, backend_rows):
            if split is not None and cpu_row["split"] != split:
                continue
            cpu_score = float(cpu_row["score"])
            backend_score = float(backend_row["score"])
            vals.append(abs(cpu_score - backend_score))
            changed += int((cpu_score >= fixed_threshold) != (backend_score >= fixed_threshold))
            count += 1
        stats = numeric_stats(vals)
        stats["pred_changed_count"] = int(changed)
        stats["n"] = int(count)
        return stats

    cpu_fixed_metrics_by_scenario = {
        scenario: metrics_at(backend_labels["test"], backend_scores[scenario][CPU_BACKEND]["test"], fixed_threshold)
        for scenario in SCENARIOS
    }

    fixed_metric_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    metrics_rows: List[Dict] = []
    calibrated_thresholds = {}
    for scenario in SCENARIOS:
        cpu_pre_stats = phase_stats(
            [r for r in latency_records if r["scenario"] == scenario and r["backend"] == CPU_BACKEND and int(r["timed"]) == 1],
            "preprocess_total_ms",
        )
        cpu_fixed = cpu_fixed_metrics_by_scenario[scenario]
        for backend in BACKENDS:
            test_diff = diff_stats_for(scenario, backend, "test")
            all_diff = diff_stats_for(scenario, backend, None)
            pre_stats = phase_stats(
                [r for r in latency_records if r["scenario"] == scenario and r["backend"] == backend and int(r["timed"]) == 1],
                "preprocess_total_ms",
            )
            tensor_stats = {
                "tensor_max_abs_diff_median": 0.0,
                "tensor_mean_abs_diff_median": 0.0,
                "tensor_p95_abs_diff_median": 0.0,
                "tensor_p99_abs_diff_median": 0.0,
                "tensor_max_abs_diff_p95": 0.0,
                "tensor_mean_abs_diff_p95": 0.0,
                "tensor_p95_abs_diff_p95": 0.0,
                "tensor_p99_abs_diff_p95": 0.0,
            }
            if backend != CPU_BACKEND:
                tensor_stats = {
                    "tensor_max_abs_diff_median": tensor_summary[backend]["max_abs_diff"]["median"],
                    "tensor_mean_abs_diff_median": tensor_summary[backend]["mean_abs_diff"]["median"],
                    "tensor_p95_abs_diff_median": tensor_summary[backend]["p95_abs_diff"]["median"],
                    "tensor_p99_abs_diff_median": tensor_summary[backend]["p99_abs_diff"]["median"],
                    "tensor_max_abs_diff_p95": tensor_summary[backend]["max_abs_diff"]["p95"],
                    "tensor_mean_abs_diff_p95": tensor_summary[backend]["mean_abs_diff"]["p95"],
                    "tensor_p95_abs_diff_p95": tensor_summary[backend]["p95_abs_diff"]["p95"],
                    "tensor_p99_abs_diff_p95": tensor_summary[backend]["p99_abs_diff"]["p95"],
                }

            fixed_metrics = metrics_at(backend_labels["test"], backend_scores[scenario][backend]["test"], fixed_threshold)
            fixed_metric_lookup[(scenario, backend)] = fixed_metrics
            f1_gap = abs(fixed_metrics["f1"] - cpu_fixed["f1"])
            recall_gap = abs(fixed_metrics["recall"] - cpu_fixed["recall"])
            preprocess_faster = pre_stats["median"] < cpu_pre_stats["median"]
            meets_acceptance = (
                backend != CPU_BACKEND
                and test_diff["pred_changed_count"] <= 2
                and test_diff["p95"] <= 0.02
                and f1_gap <= 0.01
                and recall_gap <= 0.02
                and preprocess_faster
            )
            fixed_row = {
                "scenario": scenario,
                "backend": backend,
                "threshold_mode": "fixed_cpu_threshold",
                "threshold": fixed_threshold,
                "accuracy": fixed_metrics["accuracy"],
                "precision": fixed_metrics["precision"],
                "recall": fixed_metrics["recall"],
                "f1": fixed_metrics["f1"],
                "auc_roc": fixed_metrics["auc_roc"],
                "auc_pr": fixed_metrics["auc_pr"],
                "tp": fixed_metrics["tp"],
                "fp": fixed_metrics["fp"],
                "tn": fixed_metrics["tn"],
                "fn": fixed_metrics["fn"],
                "pred_changed_all_count": all_diff["pred_changed_count"],
                "pred_changed_test_count": test_diff["pred_changed_count"],
                "score_abs_diff_median": test_diff["median"],
                "score_abs_diff_p95": test_diff["p95"],
                "score_abs_diff_max": test_diff["max"],
                "fixed_f1_gap_vs_cpu": f1_gap,
                "fixed_recall_gap_vs_cpu": recall_gap,
                "preprocess_total_median_ms": pre_stats["median"],
                "cpu_preprocess_total_median_ms": cpu_pre_stats["median"],
                "preprocess_faster_than_cpu": int(preprocess_faster),
                "meets_acceptance": int(meets_acceptance),
                **tensor_stats,
            }
            metrics_rows.append(fixed_row)

            threshold_row = choose_backend_threshold(
                args.threshold_policy,
                args.stage,
                backend_labels["calib"],
                backend_scores[scenario][backend]["calib"],
            )
            backend_threshold = float(threshold_row["threshold"])
            calibrated_thresholds[f"{scenario}:{backend}"] = backend_threshold
            cal_metrics = metrics_at(backend_labels["test"], backend_scores[scenario][backend]["test"], backend_threshold)
            metrics_rows.append(
                {
                    **fixed_row,
                    "threshold_mode": "backend_calibrated",
                    "threshold": backend_threshold,
                    "accuracy": cal_metrics["accuracy"],
                    "precision": cal_metrics["precision"],
                    "recall": cal_metrics["recall"],
                    "f1": cal_metrics["f1"],
                    "auc_roc": cal_metrics["auc_roc"],
                    "auc_pr": cal_metrics["auc_pr"],
                    "tp": cal_metrics["tp"],
                    "fp": cal_metrics["fp"],
                    "tn": cal_metrics["tn"],
                    "fn": cal_metrics["fn"],
                    "meets_acceptance": "",
                }
            )

    changed_rows: List[Dict] = []
    for scenario in SCENARIOS:
        cpu_rows = score_records[scenario][CPU_BACKEND]
        for backend in GPU_BACKENDS:
            for cpu_row, backend_row in zip(cpu_rows, score_records[scenario][backend]):
                cpu_score = float(cpu_row["score"])
                backend_score = float(backend_row["score"])
                cpu_pred = int(cpu_score >= fixed_threshold)
                backend_pred = int(backend_score >= fixed_threshold)
                if cpu_pred == backend_pred:
                    continue
                changed_rows.append(
                    {
                        "scenario": scenario,
                        "backend": backend,
                        "split": cpu_row["split"],
                        "role": cpu_row["role"],
                        "rel_path": cpu_row["rel_path"],
                        "label": cpu_row["label"],
                        "fixed_cpu_threshold": fixed_threshold,
                        "cpu_score": cpu_score,
                        "backend_score": backend_score,
                        "score_abs_diff": abs(cpu_score - backend_score),
                        "cpu_pred_fixed_threshold": cpu_pred,
                        "backend_pred_fixed_threshold": backend_pred,
                    }
                )

    latency_rows: List[Dict] = []
    for scenario in SCENARIOS:
        for backend in BACKENDS:
            timed_rows = [
                r for r in latency_records
                if r["scenario"] == scenario and r["backend"] == backend and int(r["timed"]) == 1
            ]
            for phase in ["preprocess_total_ms", "tensor_to_threshold_ms", "decoded_image_to_threshold_ms"]:
                stats = phase_stats(timed_rows, phase)
                latency_rows.append(
                    {
                        "scenario": scenario,
                        "backend": backend,
                        "phase": phase,
                        "mean_ms": stats["mean"],
                        "median_ms": stats["median"],
                        "p95_ms": stats["p95"],
                        "max_ms": stats["max"],
                    }
                )

    write_csv(output_root / "equivalence_fix_metrics.csv", metrics_rows)
    write_csv(output_root / "equivalence_fix_latency.csv", latency_rows)
    write_csv(output_root / "changed_samples_by_backend.csv", changed_rows)

    config = {
        "task": TASK_NAME,
        "stage_root": str(args.stage_root),
        "alias": args.alias,
        "stage": args.stage,
        "ahl_weights": str(args.ahl_weights),
        "projector_checkpoint": str(args.projector_checkpoint),
        "external_threshold_source": str(args.external_threshold_compare_json),
        "external_threshold_value": external_threshold,
        "fixed_threshold_source": args.fixed_threshold_source,
        "fixed_cpu_threshold_value": fixed_threshold,
        "calibrated_thresholds": calibrated_thresholds,
        "threshold_policy": args.threshold_policy,
        "n_ref": args.n_ref,
        "warmup": args.warmup,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "interpolation_mode": "BICUBIC",
        "antialias_supported_by_torchvision_resize": resize_supports_antialias(),
        "decode_policy": "PIL decode on CPU; primary latency uses decoded_image_to_threshold_ms and ignores image_load_decode_ms.",
        "backends": {
            "cpu_pil": "CPU PIL resize/crop/to_tensor/normalize baseline, unchanged.",
            "gpu_tensor_current": "Previous GPU tensor path: PIL to CHW uint8 CPU tensor, batched NCHW GPU resize/crop, antialias=True.",
            "gpu_tensor_torchvision_aa_true": "Torchvision tensor path on single CHW GPU tensor, BICUBIC, antialias=True.",
            "gpu_tensor_torchvision_aa_false": "Torchvision tensor path on single CHW GPU tensor, BICUBIC, antialias=False.",
            "gpu_tensor_uint8_aa_true": "PIL-equivalent GPU path: resize/center_crop in UINT8 domain on GPU (round+clamp like PIL), cast to float/255 and normalize AFTER crop. BICUBIC, antialias=True.",
        },
        "scenarios": {
            "query_only_diff": "Reference bank fixed to CPU-PIL; query uses each backend.",
            "ref_and_query_diff": "Reference bank and query both use the corresponding backend.",
        },
        "acceptance": {
            "prediction_changed_test_max": 2,
            "score_abs_diff_p95_max": 0.02,
            "fixed_threshold_f1_gap_max": 0.01,
            "fixed_threshold_recall_gap_max": 0.02,
            "preprocess_total_median_must_be_lower_than_cpu": True,
        },
        "backend_reference_setup_ms": {backend: banks[backend].setup_ms for backend in banks},
        "counts": {
            "train_ref_pool": len(train_items),
            "calib": len(calib_items),
            "test": len(test_items),
            "steady_test": max(0, len(test_items) - args.warmup),
        },
    }
    (output_root / "config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fixed_rows = [r for r in metrics_rows if r["threshold_mode"] == "fixed_cpu_threshold"]
    ref_fixed_gpu_rows = [r for r in fixed_rows if r["scenario"] == "ref_and_query_diff" and r["backend"] in GPU_BACKENDS]
    best_row = min(
        ref_fixed_gpu_rows,
        key=lambda r: (
            int(r["pred_changed_test_count"]),
            float(r["score_abs_diff_p95"]),
            float(r["fixed_f1_gap_vs_cpu"]),
            float(r["fixed_recall_gap_vs_cpu"]),
        ),
    )
    accepted_rows = [r for r in ref_fixed_gpu_rows if int(r["meets_acceptance"]) == 1]

    def metric_line(row: Dict) -> str:
        return (
            f"| {row['scenario']} | {row['backend']} | {int(row['pred_changed_test_count'])} | "
            f"{float(row['score_abs_diff_median']):.6f} | {float(row['score_abs_diff_p95']):.6f} | "
            f"{float(row['recall']):.4f} | {float(row['f1']):.4f} | "
            f"{float(row['preprocess_total_median_ms']):.2f} | {row['meets_acceptance']} |"
        )

    def latency_lookup(scenario: str, backend: str, phase: str) -> Dict:
        for row in latency_rows:
            if row["scenario"] == scenario and row["backend"] == backend and row["phase"] == phase:
                return row
        raise KeyError((scenario, backend, phase))

    diff_notes = []
    for backend in GPU_BACKENDS:
        qo = next(r for r in fixed_rows if r["scenario"] == "query_only_diff" and r["backend"] == backend)
        rq = next(r for r in fixed_rows if r["scenario"] == "ref_and_query_diff" and r["backend"] == backend)
        if int(rq["pred_changed_test_count"]) > int(qo["pred_changed_test_count"]) or float(rq["score_abs_diff_p95"]) > float(qo["score_abs_diff_p95"]) * 1.2:
            cause = "reference bank preprocess also increases the gap"
        elif int(qo["pred_changed_test_count"]) >= int(rq["pred_changed_test_count"]) and float(qo["score_abs_diff_p95"]) >= float(rq["score_abs_diff_p95"]) * 0.9:
            cause = "query preprocess is the dominant source"
        else:
            cause = "query and reference effects are close"
        diff_notes.append(f"- `{backend}`: {cause}.")

    substitute_answer = (
        f"Yes. Accepted backend(s): {', '.join(r['backend'] for r in accepted_rows)}."
        if accepted_rows
        else "No. No GPU backend met all acceptance criteria, so CPU-PIL should remain the production metric backend."
    )

    md = [
        f"# {TASK_NAME}",
        "",
        "## Scope",
        "",
        f"- Stage: `{args.stage}`",
        f"- Output root: `{output_root}`",
        f"- Fixed CPU threshold source: `{args.fixed_threshold_source}`",
        f"- Fixed CPU threshold: `{fixed_threshold:.12f}`",
        f"- Threshold policy: `{args.threshold_policy}`",
        f"- n_ref: `{args.n_ref}`",
        f"- warmup: `{args.warmup}`",
        "- Primary latency: `decoded_image_to_threshold_ms`; `image_load_decode_ms` is not part of the production-preloaded-image estimate.",
        "",
        "## Fixed-Threshold Equivalence",
        "",
        "| scenario | backend | test pred changed | score diff median | score diff P95 | Recall | F1 | preprocess median ms | accepted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    md.extend(metric_line(r) for r in fixed_rows)
    md.extend(
        [
            "",
            "## Latency",
            "",
            "| scenario | backend | preprocess median | tensor-to-threshold median | decoded-image-to-threshold median |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for scenario in SCENARIOS:
        for backend in BACKENDS:
            pre = latency_lookup(scenario, backend, "preprocess_total_ms")
            tensor = latency_lookup(scenario, backend, "tensor_to_threshold_ms")
            decoded = latency_lookup(scenario, backend, "decoded_image_to_threshold_ms")
            md.append(
                f"| {scenario} | {backend} | {pre['median_ms']:.2f} | {tensor['median_ms']:.2f} | {decoded['median_ms']:.2f} |"
            )
    md.extend(
        [
            "",
            "## Tensor Difference",
            "",
            "| backend | mean_abs_diff median | p95_abs_diff median | p99_abs_diff median | max_abs_diff P95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend in GPU_BACKENDS:
        s = tensor_summary[backend]
        md.append(
            f"| {backend} | {s['mean_abs_diff']['median']:.8f} | {s['p95_abs_diff']['median']:.8f} | "
            f"{s['p99_abs_diff']['median']:.8f} | {s['max_abs_diff']['p95']:.8f} |"
        )
    md.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Closest GPU backend: `{best_row['backend']}` under `ref_and_query_diff` "
            f"(test pred changed={int(best_row['pred_changed_test_count'])}, "
            f"score_abs_diff_P95={float(best_row['score_abs_diff_p95']):.6f}, "
            f"F1={float(best_row['f1']):.4f}).",
            "2. Query/reference source diagnosis:",
            *diff_notes,
            f"3. CPU-PIL replacement decision: {substitute_answer}",
            "4. Production metric recommendation: keep `cpu_pil` as the production metric backend unless an accepted GPU backend appears in this summary.",
        ]
    )
    (output_root / "equivalence_fix_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "task": TASK_NAME,
                "fixed_threshold": fixed_threshold,
                "best_ref_and_query_backend": best_row["backend"],
                "accepted_backends": [r["backend"] for r in accepted_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
