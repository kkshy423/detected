#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FP32 precision audit for TensorRT DINOv2-large encoder only.

This diagnostic stays encoder-only and does not modify projector/full-stage
logic. Heavy artifacts are written to runs/. Small reports are written to
summary/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from common import add_adpretrain_to_path, build_transform, encode_multiscale, load_image_tensor, make_encoder


TASK_NAME = "20260624_tensorrt_encoder_fp32_precision_audit_v1"
ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
DATASET_ROOT = Path("/gdata1/huangjd/xidun_mvtec_format_6_1/models__\u7403\u9762\u659c\u62cd")
SPLIT_ROOT = BRIDGE_ROOT / "splits" / "20260529_qm_xiepai_6_1_fixed_180_70_val49"
DEFAULT_SUMMARY_ROOT = BRIDGE_ROOT / "summary" / TASK_NAME
DEFAULT_RUNS_ROOT = BRIDGE_ROOT / "runs" / TASK_NAME


def run_cmd(cmd: Sequence[str], timeout: int = 120) -> Dict[str, object]:
    try:
        proc = subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": list(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": list(cmd), "error": repr(exc)}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_version(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"import_error: {exc!r}"


def collect_env(summary_root: Path) -> Dict:
    import torchvision
    from PIL import Image

    env = {
        "task_name": TASK_NAME,
        "python": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "torchvision": torchvision.__version__,
        "pillow": Image.__version__,
        "tensorrt_python": safe_version("tensorrt"),
        "onnx_python": safe_version("onnx"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "nvidia_smi": run_cmd(["nvidia-smi"], timeout=60),
        "gpu": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            env["gpu"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "total_memory_bytes": props.total_memory,
                }
            )
    write_json(summary_root / "env_inventory.json", env)
    return env


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        feats = encode_multiscale(self.encoder, x)
        return tuple(feats)


def feature_shapes(features: Sequence[torch.Tensor]) -> List[List[int]]:
    return [list(x.shape) for x in features]


def export_onnx(model: torch.nn.Module, onnx_path: Path, device: torch.device) -> Dict:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(dummy)
    output_names = [f"feat_{idx}" for idx in range(len(outputs))]
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=None,
    )
    return {"onnx_path": str(onnx_path), "output_names": output_names, "output_shapes": feature_shapes(outputs)}


def stats_ms(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def time_pytorch(model: torch.nn.Module, x: torch.Tensor, warmup: int, timed: int) -> Dict:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        vals: List[float] = []
        for _ in range(timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            vals.append((time.perf_counter() - t0) * 1000.0)
    row = {"backend": "pytorch_tf32_off", "warmup": warmup, "timed": timed}
    row.update(stats_ms(vals))
    return row


def load_trt_engine(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    return trt, engine, context


def tensor_name_iter(engine) -> Iterable[str]:
    if hasattr(engine, "num_io_tensors"):
        for idx in range(engine.num_io_tensors):
            yield engine.get_tensor_name(idx)
    else:
        for idx in range(engine.num_bindings):
            yield engine.get_binding_name(idx)


def is_input_tensor(trt, engine, name: str) -> bool:
    if hasattr(engine, "get_tensor_mode"):
        return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    return engine.binding_is_input(name)


def get_tensor_dtype(trt, engine, name: str):
    if hasattr(engine, "get_tensor_dtype"):
        dtype = engine.get_tensor_dtype(name)
    else:
        dtype = engine.get_binding_dtype(name)
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if hasattr(trt.DataType, "INT8") and dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise RuntimeError(f"unsupported TensorRT output dtype for {name}: {dtype}")


class TrtRunner:
    def __init__(self, engine_path: Path, output_shapes: Sequence[Sequence[int]], device: torch.device):
        self.trt, self.engine, self.context = load_trt_engine(engine_path)
        self.input_name = None
        self.output_names: List[str] = []
        for name in tensor_name_iter(self.engine):
            if is_input_tensor(self.trt, self.engine, name):
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None:
            raise RuntimeError("TensorRT engine has no input tensor")
        self.output_names.sort()
        self.output_tensors = []
        for name, shape in zip(self.output_names, output_shapes):
            dtype = get_tensor_dtype(self.trt, self.engine, name)
            self.output_tensors.append(torch.empty(tuple(shape), device=device, dtype=dtype))

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        if hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
            for name, tensor in zip(self.output_names, self.output_tensors):
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        else:
            bindings = [0] * self.engine.num_bindings
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                if self.engine.binding_is_input(idx):
                    bindings[idx] = int(x.data_ptr())
                else:
                    out_idx = self.output_names.index(name)
                    bindings[idx] = int(self.output_tensors[out_idx].data_ptr())
            ok = self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return list(self.output_tensors)


def time_trt(runner: TrtRunner, x: torch.Tensor, name: str, warmup: int, timed: int) -> Dict:
    for _ in range(warmup):
        _ = runner(x)
    torch.cuda.synchronize()
    vals: List[float] = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = runner(x)
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    row = {"backend": name, "warmup": warmup, "timed": timed}
    row.update(stats_ms(vals))
    return row


def feature_diff(pt: Sequence[torch.Tensor], trt_out: Sequence[torch.Tensor], backend: str, sample_idx: int, sample_rel: str) -> List[Dict]:
    rows: List[Dict] = []
    for idx, (a, b) in enumerate(zip(pt, trt_out)):
        af = a.detach().float().reshape(-1)
        bf = b.detach().float().reshape(-1)
        diff = (af - bf).abs()
        denom = torch.linalg.norm(af).clamp_min(1e-12)
        rows.append(
            {
                "backend": backend,
                "sample_idx": sample_idx,
                "sample_rel": sample_rel,
                "feature_idx": idx,
                "shape": "x".join(str(x) for x in a.shape),
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "p95_abs": float(torch.quantile(diff, 0.95).item()),
                "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
                "relative_l2": float((torch.linalg.norm(af - bf) / denom).item()),
            }
        )
    return rows


def summarize_diff(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for backend in sorted({r["backend"] for r in rows}):
        for feature_idx in sorted({r["feature_idx"] for r in rows if r["backend"] == backend}):
            sub = [r for r in rows if r["backend"] == backend and r["feature_idx"] == feature_idx]
            out.append(
                {
                    "backend": backend,
                    "feature_idx": feature_idx,
                    "max_abs_max": max(r["max_abs"] for r in sub),
                    "mean_abs_mean": statistics.mean(r["mean_abs"] for r in sub),
                    "p95_abs_max": max(r["p95_abs"] for r in sub),
                    "cosine_min": min(r["cosine"] for r in sub),
                    "relative_l2_max": max(r["relative_l2"] for r in sub),
                    "n_samples": len(sub),
                }
            )
    return out


def select_sample_paths(dataset_root: Path, split_root: Path, sample_count: int) -> List[Path]:
    normal = (split_root / "test_normal.txt").read_text(encoding="utf-8").splitlines()
    anomaly = (split_root / "test_anomaly.txt").read_text(encoding="utf-8").splitlines()
    picks = normal[: sample_count // 2] + anomaly[: sample_count - sample_count // 2]
    return [dataset_root / rel for rel in picks]


def build_config_flags(config, trt) -> List[str]:
    flags: List[str] = []
    for name in dir(trt.BuilderFlag):
        if not name.isupper():
            continue
        flag = getattr(trt.BuilderFlag, name)
        try:
            if config.get_flag(flag):
                flags.append(name)
        except Exception:
            continue
    return sorted(flags)


def build_engine_with_config(
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    variant_name: str,
    disable_tf32: bool,
    strong_fp32_constraints: bool,
) -> Dict:
    import tensorrt as trt

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    class Recorder(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            log_lines.append(f"[{severity}] {msg}")

    logger = Recorder()
    t0 = time.perf_counter()
    manifest = {
        "variant": variant_name,
        "disable_tf32_requested": disable_tf32,
        "strong_fp32_constraints_requested": strong_fp32_constraints,
        "builder": "tensorrt_python",
    }
    try:
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        ok = parser.parse(onnx_path.read_bytes())
        for idx in range(parser.num_errors):
            log_lines.append(str(parser.get_error(idx)))
        if not ok:
            raise RuntimeError("TensorRT ONNX parser failed")

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
        else:
            config.max_workspace_size = 4 * 1024 * 1024 * 1024

        default_tf32 = bool(config.get_flag(trt.BuilderFlag.TF32)) if hasattr(config, "get_flag") else None
        manifest["tf32_default_before_edit"] = default_tf32

        if disable_tf32 and hasattr(config, "clear_flag") and hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)

        strong_mode_applied = False
        strong_mode_flags: List[str] = []
        if strong_fp32_constraints:
            for name in ("OBEY_PRECISION_CONSTRAINTS", "STRICT_TYPES"):
                if hasattr(trt.BuilderFlag, name):
                    config.set_flag(getattr(trt.BuilderFlag, name))
                    strong_mode_flags.append(name)
                    strong_mode_applied = True
            manifest["strong_fp32_constraints_applied"] = strong_mode_applied
            manifest["strong_fp32_constraint_flags"] = strong_mode_flags
        else:
            manifest["strong_fp32_constraints_applied"] = False
            manifest["strong_fp32_constraint_flags"] = []

        manifest["flags_after_edit"] = build_config_flags(config, trt)
        manifest["tf32_enabled_after_edit"] = "TF32" in manifest["flags_after_edit"]

        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT build_serialized_network returned None")
            engine_path.write_bytes(bytes(serialized))
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("TensorRT build_engine returned None")
            engine_path.write_bytes(engine.serialize())
        returncode = 0
    except Exception as exc:
        log_lines.append(f"ERROR: {exc!r}")
        manifest["build_error"] = repr(exc)
        returncode = 1
    elapsed = time.perf_counter() - t0
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8", errors="replace")
    manifest.update(
        {
            "returncode": returncode,
            "elapsed_s": elapsed,
            "engine_path": str(engine_path),
            "log_path": str(log_path),
            "engine_exists": engine_path.exists(),
            "engine_size_bytes": engine_path.stat().st_size if engine_path.exists() else 0,
        }
    )
    return manifest


def write_final_report(summary_root: Path, report: Dict) -> None:
    lines = [
        f"# {TASK_NAME}",
        "",
        "## Scope",
        "",
        "- Encoder only.",
        "- No projector, no full-stage replacement, no FP16 drop-in, no main-env change.",
        "- PyTorch baseline uses TF32 off.",
        "",
        "## Engine Config",
        "",
    ]
    for item in report["build_manifest"]["engines"]:
        lines.append(
            f"- {item['variant']}: tf32_enabled={item.get('tf32_enabled_after_edit')}, "
            f"flags={item.get('flags_after_edit', [])}, returncode={item.get('returncode')}"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| backend | median_ms | mean_ms | p95_ms | min_ms | max_ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["latency_rows"]:
        lines.append(
            f"| {row['backend']} | {row['median_ms']:.4f} | {row['mean_ms']:.4f} | {row['p95_ms']:.4f} | {row['min_ms']:.4f} | {row['max_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Diff Summary",
            "",
            "| backend | feature_idx | max_abs_max | mean_abs_mean | p95_abs_max | cosine_min | relative_l2_max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["feature_diff_summary"]:
        lines.append(
            f"| {row['backend']} | {row['feature_idx']} | {row['max_abs_max']:.6g} | {row['mean_abs_mean']:.6g} | {row['p95_abs_max']:.6g} | {row['cosine_min']:.9f} | {row['relative_l2_max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["conclusions"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `build_config_manifest.json`",
            "- `fp32_precision_audit_latency.csv`",
            "- `fp32_precision_audit_feature_diff.csv`",
            "",
        ]
    )
    (summary_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--backbone", default="dinov2-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timed", type=int, default=100)
    p.add_argument("--sample-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--rewrite-report-only", action="store_true")
    return p.parse_args()


def refresh_conclusions(report: Dict) -> Dict:
    by_backend = {row["backend"]: row for row in report["feature_diff_summary"] if row["feature_idx"] == 3}
    latency_by_backend = {row["backend"]: row for row in report["latency_rows"]}
    conclusions: List[str] = []
    default_row = by_backend.get("trt_fp32_default")
    disabled_row = by_backend.get("trt_fp32_tf32_disabled")
    strict_row = by_backend.get("trt_fp32_tf32_disabled_strict")
    if default_row and disabled_row:
        diff_ratio = disabled_row["relative_l2_max"] / max(default_row["relative_l2_max"], 1e-12)
        conclusions.append(
            f"disable_tf32 deepest-feature relative_l2_max changed from {default_row['relative_l2_max']:.6g} to {disabled_row['relative_l2_max']:.6g}."
        )
        conclusions.append(
            f"disable_tf32 deepest-feature cosine_min changed from {default_row['cosine_min']:.9f} to {disabled_row['cosine_min']:.9f}."
        )
        default_latency = latency_by_backend.get("trt_fp32_default")
        disabled_latency = latency_by_backend.get("trt_fp32_tf32_disabled")
        if default_latency and disabled_latency:
            conclusions.append(
                f"disable_tf32 median latency changed from {default_latency['median_ms']:.4f} ms to {disabled_latency['median_ms']:.4f} ms."
            )
        if diff_ratio < 0.7:
            conclusions.append("TF32 root-cause hypothesis supported in this audit.")
        else:
            conclusions.append("TF32 root-cause hypothesis not supported in this audit.")
    else:
        conclusions.append("Required FP32 engine variants were not all built successfully.")
    if strict_row:
        conclusions.append(
            f"stronger FP32 constraints deepest-feature relative_l2_max = {strict_row['relative_l2_max']:.6g}, cosine_min = {strict_row['cosine_min']:.9f}."
        )
    conclusions.append("Stop here. Do not enter full-stage drop-in from this audit.")
    report["conclusions"] = conclusions
    return report


def main() -> int:
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    if args.rewrite_report_only:
        report_path = args.summary_root / "diagnostic_result.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report = refresh_conclusions(report)
        write_json(report_path, report)
        write_final_report(args.summary_root, report)
        return 0

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    env = collect_env(args.summary_root)
    missing: List[str] = []
    if env.get("tensorrt_python") is None:
        missing.append("TensorRT Python package not importable")
    if env.get("onnx_python") is None:
        missing.append("onnx Python package not importable")
    if not torch.cuda.is_available():
        missing.append("torch.cuda.is_available() is False")
    if missing:
        write_json(args.summary_root / "missing_dependency.json", {"missing": missing, "env": env})
        (args.summary_root / "final_report.md").write_text(
            "# missing dependency\n\n" + "\n".join(f"- {x}" for x in missing) + "\n",
            encoding="utf-8",
        )
        return 0

    device = torch.device(args.device)
    add_adpretrain_to_path(str(args.adpretrain_root))
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    wrapper = EncoderWrapper(encoder).to(device).eval()

    onnx_path = args.runs_root / "onnx" / "dinov2_large_encoder_multiscale_b1_224.onnx"
    onnx_meta = export_onnx(wrapper, onnx_path, device)

    sample_paths = select_sample_paths(args.dataset_root, args.split_root, args.sample_count)
    transform = build_transform(args.backbone)
    sample_inputs = [load_image_tensor(path, transform, device) for path in sample_paths]
    sample_rels = [str(path.relative_to(args.dataset_root)).replace("\\", "/") for path in sample_paths]

    with torch.no_grad():
        pytorch_outputs = [wrapper(inp) for inp in sample_inputs]

    x = sample_inputs[0]
    latency_rows = [time_pytorch(wrapper, x, args.warmup, args.timed)]

    engine_specs = [
        ("trt_fp32_default", False, False),
        ("trt_fp32_tf32_disabled", True, False),
    ]

    build_manifest = {
        "task_name": TASK_NAME,
        "sample_count": len(sample_paths),
        "sample_paths": sample_rels,
        "onnx_meta": onnx_meta,
        "engines": [],
    }

    import tensorrt as trt

    if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS") and hasattr(trt.BuilderFlag, "STRICT_TYPES"):
        engine_specs.append(("trt_fp32_tf32_disabled_strict", True, True))

    diff_rows: List[Dict] = []
    for variant_name, disable_tf32, strong_constraints in engine_specs:
        build_info = build_engine_with_config(
            onnx_path=onnx_path,
            engine_path=args.runs_root / "engines" / f"{variant_name}.engine",
            log_path=args.runs_root / "logs" / f"{variant_name}.log",
            variant_name=variant_name,
            disable_tf32=disable_tf32,
            strong_fp32_constraints=strong_constraints,
        )
        build_manifest["engines"].append(build_info)
        if build_info.get("returncode") != 0 or not build_info.get("engine_exists"):
            continue
        runner = TrtRunner(Path(build_info["engine_path"]), onnx_meta["output_shapes"], device)
        latency_rows.append(time_trt(runner, x, variant_name, args.warmup, args.timed))
        for idx, (inp, sample_rel) in enumerate(zip(sample_inputs, sample_rels)):
            out = runner(inp)
            torch.cuda.synchronize()
            diff_rows.extend(feature_diff(pytorch_outputs[idx], out, variant_name, idx, sample_rel))

    feature_diff_summary = summarize_diff(diff_rows)
    write_json(args.summary_root / "build_config_manifest.json", build_manifest)
    write_csv(args.summary_root / "fp32_precision_audit_latency.csv", latency_rows)
    write_csv(args.summary_root / "fp32_precision_audit_feature_diff.csv", diff_rows)

    report = {
        "build_manifest": build_manifest,
        "latency_rows": latency_rows,
        "feature_diff_summary": feature_diff_summary,
        "conclusions": [],
    }
    report = refresh_conclusions(report)
    write_json(args.summary_root / "diagnostic_result.json", report)
    write_final_report(args.summary_root, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
