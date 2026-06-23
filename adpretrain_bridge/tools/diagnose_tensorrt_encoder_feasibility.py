#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TensorRT feasibility diagnostic for DINOv2-large encoder only.

This script intentionally does not export projector/full-stage pipelines, does
not train, and does not modify threshold/bridge logic. Heavy artifacts are
written to runs/. Small reports are written to summary/.
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

from common import add_adpretrain_to_path, encode_multiscale, make_encoder


TASK_NAME = "20260623_tensorrt_encoder_feasibility_v1"
ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
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
    except Exception as exc:  # pragma: no cover - diagnostic robustness
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
    except Exception as exc:  # pragma: no cover - diagnostic robustness
        return f"import_error: {exc!r}"


def collect_env(summary_root: Path) -> Dict:
    import torchvision
    from PIL import Image

    trtexec = shutil.which("trtexec")
    nvcc = shutil.which("nvcc")
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
        "onnxruntime_python": safe_version("onnxruntime"),
        "trtexec_path": trtexec,
        "nvcc_path": nvcc,
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": run_cmd(["nvidia-smi"], timeout=60),
        "nvcc_version": run_cmd([nvcc, "--version"], timeout=60) if nvcc else None,
        "trtexec_version": run_cmd([trtexec, "--version"], timeout=60) if trtexec else None,
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


def trtexec_build(trtexec: str, onnx_path: Path, engine_path: Path, log_path: Path, fp16: bool) -> Dict:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--shapes=input:1x3x224x224",
        "--memPoolSize=workspace:4096",
        "--verbose",
    ]
    if fp16:
        cmd.append("--fp16")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return {
        "precision": "FP16" if fp16 else "FP32",
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "engine_path": str(engine_path),
        "log_path": str(log_path),
        "engine_exists": engine_path.exists(),
        "engine_size_bytes": engine_path.stat().st_size if engine_path.exists() else 0,
    }


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
    row = {"backend": "pytorch", "warmup": warmup, "timed": timed}
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


def get_tensor_shape(engine, name: str):
    if hasattr(engine, "get_tensor_shape"):
        return tuple(int(x) for x in engine.get_tensor_shape(name))
    return tuple(int(x) for x in engine.get_binding_shape(name))


class TrtRunner:
    def __init__(self, engine_path: Path, output_shapes: Sequence[Sequence[int]], device: torch.device):
        self.trt, self.engine, self.context = load_trt_engine(engine_path)
        self.device = device
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
        self.output_tensors = [
            torch.empty(tuple(shape), device=device, dtype=torch.float32) for shape in output_shapes
        ]

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


def feature_diff(pt: Sequence[torch.Tensor], trt_out: Sequence[torch.Tensor], backend: str, sample_idx: int) -> List[Dict]:
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


def write_missing_report(summary_root: Path, missing: List[str], env: Dict) -> None:
    text = [
        f"# {TASK_NAME} missing dependency report",
        "",
        "TensorRT feasibility diagnostic stopped before ONNX export/build because required dependencies are unavailable.",
        "",
        "## Missing",
        "",
    ]
    text.extend(f"- {item}" for item in missing)
    text.extend(
        [
            "",
            "## Recorded Environment",
            "",
            f"- torch: {env.get('torch')}",
            f"- torch_cuda: {env.get('torch_cuda')}",
            f"- torchvision: {env.get('torchvision')}",
            f"- TensorRT python: {env.get('tensorrt_python')}",
            f"- trtexec: {env.get('trtexec_path')}",
            "",
            "No dependencies were installed or modified.",
            "",
        ]
    )
    (summary_root / "missing_dependency_report.md").write_text("\n".join(text), encoding="utf-8")
    (summary_root / "final_report.md").write_text("\n".join(text), encoding="utf-8")


def write_final_report(summary_root: Path, report: Dict) -> None:
    build = report.get("build", {})
    rows = report.get("latency_rows", [])
    diff_summary = report.get("diff_summary", [])
    lines = [
        f"# {TASK_NAME}",
        "",
        "## Scope",
        "",
        "- Encoder only: DINOv2-large wrapper, fixed input [1,3,224,224].",
        "- No projector, no full-stage replacement, no INT8, no retraining, no threshold/alpha change.",
        "",
        "## Status",
        "",
        f"- ONNX export: {'success' if report.get('onnx_export_success') else 'failed'}",
        f"- TensorRT FP32 build: {'success' if build.get('FP32', {}).get('returncode') == 0 and build.get('FP32', {}).get('engine_exists') else 'failed'}",
        f"- TensorRT FP16 build: {'success' if build.get('FP16', {}).get('returncode') == 0 and build.get('FP16', {}).get('engine_exists') else 'failed'}",
        "",
        "## Latency",
        "",
        "| backend | median_ms | mean_ms | p95_ms | min_ms | max_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['median_ms']:.4f} | {row['mean_ms']:.4f} | {row['p95_ms']:.4f} | {row['min_ms']:.4f} | {row['max_ms']:.4f} |"
        )
    lines.extend(["", "## Feature Diff Summary", ""])
    lines.extend(
        [
            "| backend | feature_idx | max_abs_max | mean_abs_mean | p95_abs_max | cosine_min | relative_l2_max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diff_summary:
        lines.append(
            f"| {row['backend']} | {row['feature_idx']} | {row['max_abs_max']:.6g} | {row['mean_abs_mean']:.6g} | {row['p95_abs_max']:.6g} | {row['cosine_min']:.9f} | {row['relative_l2_max']:.6g} |"
        )
    conclusions = report.get("conclusions", [])
    lines.extend(["", "## Conclusion", ""])
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- runs: `{report.get('runs_root')}`",
            f"- env inventory: `env_inventory.json`",
            f"- build results: `trt_build_results.json`",
            f"- latency: `encoder_latency.csv`",
            f"- feature diff detail: `feature_diff_detail.csv`",
            f"- feature diff summary: `feature_diff_summary.csv`",
            "",
        ]
    )
    (summary_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--adpretrain-root", type=Path, default=ADPRETRAIN_ROOT)
    p.add_argument("--backbone", default="dinov2-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timed", type=int, default=100)
    p.add_argument("--diff-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260623)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    env = collect_env(args.summary_root)
    missing: List[str] = []
    if not env.get("trtexec_path"):
        missing.append("trtexec executable not found on PATH")
    if env.get("tensorrt_python") is None:
        missing.append("TensorRT Python package not importable")
    if env.get("onnx_python") is None:
        missing.append("onnx Python package not importable")
    if not torch.cuda.is_available():
        missing.append("torch.cuda.is_available() is False")
    if missing:
        write_missing_report(args.summary_root, missing, env)
        return 0

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    add_adpretrain_to_path(str(args.adpretrain_root))
    encoder = make_encoder(args.backbone, str(args.adpretrain_root), device)
    wrapper = EncoderWrapper(encoder).to(device).eval()

    onnx_path = args.runs_root / "onnx" / "dinov2_large_encoder_multiscale_b1_224.onnx"
    export_meta = export_onnx(wrapper, onnx_path, device)
    write_json(args.summary_root / "onnx_export_meta.json", export_meta)

    trtexec = str(env["trtexec_path"])
    build_results: Dict[str, Dict] = {}
    build_results["FP32"] = trtexec_build(
        trtexec,
        onnx_path,
        args.runs_root / "engines" / "dinov2_large_encoder_fp32.engine",
        args.runs_root / "logs" / "trtexec_fp32_build.log",
        fp16=False,
    )
    build_results["FP16"] = trtexec_build(
        trtexec,
        onnx_path,
        args.runs_root / "engines" / "dinov2_large_encoder_fp16.engine",
        args.runs_root / "logs" / "trtexec_fp16_build.log",
        fp16=True,
    )
    write_json(args.summary_root / "trt_build_results.json", build_results)

    latency_rows: List[Dict] = []
    diff_rows: List[Dict] = []
    x = torch.randn(1, 3, 224, 224, device=device)
    latency_rows.append(time_pytorch(wrapper, x, args.warmup, args.timed))

    sample_inputs = [torch.randn(1, 3, 224, 224, device=device) for _ in range(args.diff_samples)]
    with torch.no_grad():
        pt_outputs = [wrapper(inp) for inp in sample_inputs]

    for precision in ("FP32", "FP16"):
        br = build_results[precision]
        if br.get("returncode") != 0 or not br.get("engine_exists"):
            continue
        runner = TrtRunner(Path(br["engine_path"]), export_meta["output_shapes"], device)
        latency_rows.append(time_trt(runner, x, f"trt_{precision.lower()}", args.warmup, args.timed))
        for idx, inp in enumerate(sample_inputs):
            out = runner(inp)
            torch.cuda.synchronize()
            diff_rows.extend(feature_diff(pt_outputs[idx], out, f"trt_{precision.lower()}", idx))

    diff_summary = summarize_diff(diff_rows)
    write_csv(args.summary_root / "encoder_latency.csv", latency_rows)
    write_csv(args.summary_root / "feature_diff_detail.csv", diff_rows)
    write_csv(args.summary_root / "feature_diff_summary.csv", diff_summary)

    conclusions = []
    fp32_ok = build_results["FP32"].get("returncode") == 0 and build_results["FP32"].get("engine_exists")
    fp16_ok = build_results["FP16"].get("returncode") == 0 and build_results["FP16"].get("engine_exists")
    conclusions.append("ONNX export succeeded." if onnx_path.exists() else "ONNX export failed.")
    conclusions.append("FP32 engine build succeeded." if fp32_ok else "FP32 engine build failed; do not proceed.")
    conclusions.append("FP16 engine build succeeded." if fp16_ok else "FP16 engine build failed; do not use FP16.")
    if fp32_ok:
        fp32_rows = [r for r in diff_summary if r["backend"] == "trt_fp32"]
        fp32_equiv = bool(fp32_rows) and max(r["relative_l2_max"] for r in fp32_rows) < 1e-3 and min(r["cosine_min"] for r in fp32_rows) > 0.999
        conclusions.append(
            "FP32 feature equivalence is within the pre-check tolerance used in this diagnostic."
            if fp32_equiv
            else "FP32 feature diff is not clearly equivalent; inspect feature_diff_summary.csv before any drop-in work."
        )
    if fp16_ok:
        lat = {r["backend"]: r for r in latency_rows}
        if "pytorch" in lat and "trt_fp16" in lat:
            speedup = lat["pytorch"]["median_ms"] / max(lat["trt_fp16"]["median_ms"], 1e-12)
            conclusions.append(f"FP16 median latency speedup vs PyTorch encoder: {speedup:.3f}x.")
            conclusions.append(
                "Recommend next-round encoder drop-in full-stage diagnostic."
                if fp32_ok and speedup > 1.05
                else "Do not proceed to full-stage drop-in yet unless build/equivalence/latency issues are resolved."
            )
    report = {
        "runs_root": str(args.runs_root),
        "onnx_export_success": onnx_path.exists(),
        "build": build_results,
        "latency_rows": latency_rows,
        "diff_summary": diff_summary,
        "conclusions": conclusions,
    }
    write_json(args.summary_root / "diagnostic_result.json", report)
    write_final_report(args.summary_root, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
