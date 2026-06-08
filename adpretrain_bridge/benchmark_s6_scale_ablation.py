#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6 AHL bridge ablation: dual scale vs 14x14 feature only.

This is an eval-only bridge experiment. It keeps ADPretrain features fixed and
only changes whether AHL consumes the 7x7 feature_scale branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
AHL_ROOT = Path("/ghome/huangjd/code/detected/AHL")
DEFAULT_STAGE_ROOT = Path(
    "/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/"
    "20260519_stage_roots_plain_fixed_180_79/S6"
)
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "output/20260528_s6_scale_ablation_v1"
DEFAULT_AHL_WEIGHTS = (
    BRIDGE_ROOT
    / "output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S6/ahl/models_qiumianxiepai_ctest.pkl"
)
DEFAULT_BASELINE_METRICS = (
    BRIDGE_ROOT / "output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S6/metrics/metrics.json"
)
DEFAULT_ALIAS = "models_qiumianxiepai"
DEFAULT_STAGE = "S6"
DEFAULT_POLICY = "strategy_mild_stage_v2_1_safe"


@dataclass
class Item:
    role: str
    label: int
    stage_rel: str
    source_rel: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--ahl-weights", type=Path, default=DEFAULT_AHL_WEIGHTS)
    p.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    p.add_argument("--alias", default=DEFAULT_ALIAS)
    p.add_argument("--stage", default=DEFAULT_STAGE)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--n-ref", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-calib", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    return p.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_feature_path(class_root: Path, scale: str, stage_rel: str) -> Path:
    rel = Path(stage_rel)
    return class_root / scale / rel.with_suffix(".npy")


def load_feature(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.load(path).astype(np.float32, copy=False)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def load_manifest(stage_root: Path, alias: str) -> Tuple[Path, Dict]:
    class_root = stage_root / alias
    manifest_path = class_root / "stage_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return class_root, read_json(manifest_path)


def role_items(manifest: Dict, roles: Iterable[str]) -> List[Item]:
    wanted = set(roles)
    out: List[Item] = []
    for row in manifest["mappings"]:
        role = str(row["role"])
        if role not in wanted:
            continue
        label = 0 if role.endswith("_normal") else 1
        out.append(Item(role=role, label=label, stage_rel=row["stage_rel"], source_rel=row["source_rel"]))
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def combine_dra_scores(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    score = -1.0 * outputs[0]
    for item in outputs[1:]:
        score = score + item
    return score.reshape(-1)


def run_variant(
    model: torch.nn.Module,
    class_root: Path,
    ref_feature: torch.Tensor,
    ref_scale: torch.Tensor,
    items: Sequence[Item],
    use_scale: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict[str, float]]:
    scores: List[float] = []
    labels: List[int] = []
    rows: List[Dict] = []
    elapsed: List[float] = []
    for item in items:
        feature = load_feature(stage_feature_path(class_root, "feature", item.stage_rel), device)
        image = torch.cat([ref_feature, feature], dim=0)
        if use_scale:
            feature_scale = load_feature(stage_feature_path(class_root, "feature_scale", item.stage_rel), device)
            image_scale = torch.cat([ref_scale, feature_scale], dim=0)
        else:
            image_scale = None

        sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(image=image, image_scale=image_scale, label=None, var=model.parameters())
            score_tensor = combine_dra_scores(outputs)
        sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        score = float(score_tensor.detach().cpu().numpy().reshape(-1)[0])
        scores.append(score)
        labels.append(item.label)
        elapsed.append(elapsed_ms)
        rows.append(
            {
                "role": item.role,
                "label": item.label,
                "score": score,
                "stage_rel": item.stage_rel,
                "source_rel": item.source_rel,
                "ahl_forward_ms": elapsed_ms,
            }
        )
    timing = {
        "mean_ms": float(np.mean(elapsed)) if elapsed else None,
        "median_ms": float(np.percentile(elapsed, 50)) if elapsed else None,
        "p90_ms": float(np.percentile(elapsed, 90)) if elapsed else None,
        "p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        "max_ms": float(np.max(elapsed)) if elapsed else None,
    }
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), rows, timing


def aucs(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(set(labels.tolist())) < 2:
        return {"auc_roc": None, "auc_pr": None}
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
    }


def metric_delta(a: Dict, b: Dict) -> Dict:
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "tp", "fp", "tn", "fn"]
    out = {}
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        out[key] = None if av is None or bv is None else bv - av
    return out


def write_scores_csv(path: Path, rows_by_variant: Dict[str, Sequence[Dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "role", "label", "score", "pred", "stage_rel", "source_rel", "ahl_forward_ms"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, rows in rows_by_variant.items():
            for row in rows:
                payload = {"variant": variant}
                payload.update(row)
                writer.writerow(payload)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(AHL_ROOT))
    sys.path.insert(0, str(BRIDGE_ROOT))
    from modeling.DRA_AHL import DRA
    from threshold_policies import apply_thresholds

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    class_root, manifest = load_manifest(args.stage_root, args.alias)
    ref_items = role_items(manifest, ["train_normal"])[: args.n_ref]
    if len(ref_items) < args.n_ref:
        raise ValueError(f"Need {args.n_ref} train_normal reference samples, got {len(ref_items)}")
    calib_items = role_items(manifest, ["calib_normal", "calib_anomaly"])
    test_items = role_items(manifest, ["test_normal", "test_anomaly"])
    if args.max_calib is not None:
        calib_items = calib_items[: args.max_calib]
    if args.max_test is not None:
        test_items = test_items[: args.max_test]

    ref_feature = torch.cat(
        [load_feature(stage_feature_path(class_root, "feature", item.stage_rel), device) for item in ref_items],
        dim=0,
    )
    ref_scale = torch.cat(
        [load_feature(stage_feature_path(class_root, "feature_scale", item.stage_rel), device) for item in ref_items],
        dim=0,
    )

    state_dict = torch.load(args.ahl_weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    variants = {
        "dual_scale": {"n_scales": 2, "use_scale": True},
        "feature14_only": {"n_scales": 1, "use_scale": False},
    }
    output_root = args.output_root
    summary_dir = output_root / "summary"
    rows_by_variant: Dict[str, List[Dict]] = {}
    summary: Dict[str, Dict] = {}

    for name, cfg_row in variants.items():
        cfg = type("Cfg", (), {"total_heads": 4, "n_scales": cfg_row["n_scales"], "nRef": args.n_ref})()
        model = DRA(cfg, backbone="resnet18").to(device)
        model.load_state_dict(state_dict)
        model.eval()

        calib_labels, calib_scores, calib_rows, calib_timing = run_variant(
            model, class_root, ref_feature, ref_scale, calib_items, cfg_row["use_scale"], device
        )
        test_labels, test_scores, test_rows, test_timing = run_variant(
            model, class_root, ref_feature, ref_scale, test_items, cfg_row["use_scale"], device
        )
        policy_results = apply_thresholds(calib_labels, calib_scores, test_labels, test_scores, stage=args.stage)
        primary = policy_results[args.policy]
        threshold = float(primary["calibration"]["threshold"])
        for row in calib_rows + test_rows:
            row["pred"] = int(float(row["score"]) >= threshold)
        rows_by_variant[name] = calib_rows + test_rows
        test_metric = dict(primary["test"])
        test_metric.update(aucs(test_labels, test_scores))
        calib_metric = dict(primary["calibration"])
        calib_metric.update(aucs(calib_labels, calib_scores))
        summary[name] = {
            "description": "AHL n_scales=2, uses 14x14 feature and 7x7 feature_scale"
            if cfg_row["use_scale"]
            else "AHL n_scales=1, uses only 14x14 feature; 7x7 feature_scale is not passed",
            "n_scales": cfg_row["n_scales"],
            "uses_feature_scale": bool(cfg_row["use_scale"]),
            "policy": args.policy,
            "threshold": threshold,
            "calibration": calib_metric,
            "test": test_metric,
            "calib_timing": calib_timing,
            "test_timing": test_timing,
        }

    diff = metric_delta(summary["dual_scale"]["test"], summary["feature14_only"]["test"])
    baseline_metrics = read_json(args.baseline_metrics) if args.baseline_metrics.exists() else {}
    payload = {
        "stage": args.stage,
        "alias": args.alias,
        "stage_root": str(args.stage_root),
        "class_root": str(class_root),
        "weights": str(args.ahl_weights),
        "policy": args.policy,
        "n_ref": args.n_ref,
        "reference_items": [item.__dict__ for item in ref_items],
        "calib_count": len(calib_items),
        "test_count": len(test_items),
        "variants": summary,
        "feature14_only_minus_dual_scale": diff,
        "historical_dual_scale_metrics": {
            "source": str(args.baseline_metrics),
            "primary_policy": baseline_metrics.get("primary_policy"),
            "primary": baseline_metrics.get("primary"),
            "auc_roc": baseline_metrics.get("auc_roc"),
            "auc_pr": baseline_metrics.get("auc_pr"),
        },
        "notes": [
            "Eval-only bridge ablation; ADPretrain feature caches and AHL weights are fixed.",
            "Both variants use the same fixed reference bank selected from the S6 train_normal manifest order.",
            "The production threshold strategy is fixed, and each variant calibrates its threshold on its own calibration scores.",
            "No test-best-F1 threshold is used for the primary comparison.",
        ],
    }
    write_json(summary_dir / "s6_scale_ablation_summary.json", payload)
    write_scores_csv(summary_dir / "s6_scale_ablation_scores.csv", rows_by_variant)

    lines = [
        "# S6 AHL Feature Scale Ablation",
        "",
        "Eval-only comparison with ADPretrain features and AHL weights fixed.",
        "",
        f"- stage: `{args.stage}`",
        f"- policy: `{args.policy}`",
        f"- reference count: `{args.n_ref}`",
        f"- calib/test count: `{len(calib_items)}` / `{len(test_items)}`",
        "",
        "| Variant | feature_scale | threshold | Acc | P | R | F1 | AUC-ROC | AUC-PR | TP | FP | TN | FN | median ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["dual_scale", "feature14_only"]:
        row = summary[name]
        test = row["test"]
        timing = row["test_timing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    "yes" if row["uses_feature_scale"] else "no",
                    fmt(row["threshold"]),
                    fmt(test.get("accuracy")),
                    fmt(test.get("precision")),
                    fmt(test.get("recall")),
                    fmt(test.get("f1")),
                    fmt(test.get("auc_roc")),
                    fmt(test.get("auc_pr")),
                    fmt(test.get("tp")),
                    fmt(test.get("fp")),
                    fmt(test.get("tn")),
                    fmt(test.get("fn")),
                    fmt(timing.get("median_ms")),
                    fmt(timing.get("p95_ms")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "",
            "feature14_only minus dual_scale on test:",
            "",
            "| Acc | P | R | F1 | AUC-ROC | AUC-PR | TP | FP | TN | FN |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join(
                fmt(diff.get(k))
                for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "tp", "fp", "tn", "fn"]
            )
            + " |",
            "",
            "## Interpretation",
            "",
        ]
    )
    f1_delta = diff.get("f1")
    recall_delta = diff.get("recall")
    precision_delta = diff.get("precision")
    if f1_delta is not None and f1_delta > 0.01:
        verdict = "14x14-only improves the fixed-policy test F1; feature_scale is likely diluting this S6 signal under the current bridge."
    elif f1_delta is not None and f1_delta < -0.01:
        verdict = "dual-scale remains better on fixed-policy test F1; feature_scale still contributes useful signal at S6."
    else:
        verdict = "F1 difference is small; feature_scale has no clear decision-level benefit under this S6 setting."
    lines.append(f"- {verdict}")
    lines.append(
        f"- Precision delta `{fmt(precision_delta)}`, recall delta `{fmt(recall_delta)}`; use these to judge whether the change trades missed defects for false alarms."
    )
    lines.append("- This is an AHL-side bridge ablation only; ADPretrain residual/projector/compress are unchanged.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- summary json: `{summary_dir / 's6_scale_ablation_summary.json'}`")
    lines.append(f"- scores csv: `{summary_dir / 's6_scale_ablation_scores.csv'}`")
    (summary_dir / "s6_scale_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {summary_dir / 's6_scale_ablation_summary.md'}")
    print(f"WROTE {summary_dir / 's6_scale_ablation_summary.json'}")
    print(f"WROTE {summary_dir / 's6_scale_ablation_scores.csv'}")


if __name__ == "__main__":
    main()
