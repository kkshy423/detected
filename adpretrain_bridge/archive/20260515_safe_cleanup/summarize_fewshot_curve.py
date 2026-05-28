#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize first-round qm_xiepai few-shot AHL+CHMM curve."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fewshot_qm_xiepai_common import FIRST_ROUND_STAGES, MAIN_DOC, metrics_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize qm_xiepai few-shot curve metrics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=FIRST_ROUND_STAGES)
    parser.add_argument("--main-doc", type=Path, default=MAIN_DOC)
    parser.add_argument("--usable-f1", type=float, default=0.80)
    parser.add_argument("--usable-precision", type=float, default=0.80)
    parser.add_argument("--usable-recall", type=float, default=0.80)
    parser.add_argument("--supervised-switch-f1", type=float, default=0.90)
    parser.add_argument("--supervised-switch-precision", type=float, default=0.90)
    parser.add_argument("--supervised-switch-recall", type=float, default=0.90)
    parser.add_argument("--diminish-min-f1-gain", type=float, default=0.02)
    return parser.parse_args()


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_rows(output_root: Path, stages: List[str]) -> List[Dict[str, object]]:
    rows = []
    for stage in stages:
        path = metrics_dir(output_root, stage) / "metrics.json"
        payload = read_json(path)
        primary = payload["primary"]
        rows.append({
            "stage": stage,
            "primary_policy": payload["primary_policy"],
            "precision": float(primary["precision"]),
            "recall": float(primary["recall"]),
            "f1": float(primary["f1"]),
            "accuracy": float(primary["accuracy"]),
            "auc_roc": float(payload["auc_roc"]),
            "auc_pr": float(payload["auc_pr"]),
            "inference_time_ms": payload.get("inference_time_ms"),
            "calib_normal": payload["calibration_counts"]["normal"],
            "calib_anomaly": payload["calibration_counts"]["anomaly"],
            "test_normal": payload["test_counts"]["normal"],
            "test_anomaly": payload["test_counts"]["anomaly"],
        })
    return rows


def first_stage(rows, predicate):
    for row in rows:
        if predicate(row):
            return row["stage"]
    return None


def diminishing_stage(rows: List[dict], min_gain: float):
    previous = None
    for row in rows:
        if previous is not None:
            gain = row["f1"] - previous["f1"]
            if gain < min_gain:
                return row["stage"], gain
        previous = row
    return None, None


def main() -> None:
    args = parse_args()
    rows = load_rows(args.output_root, args.stages)
    if not rows:
        raise ValueError("No metrics rows found")
    p_delta = rows[-1]["precision"] - rows[0]["precision"]
    r_delta = rows[-1]["recall"] - rows[0]["recall"]
    f_delta = rows[-1]["f1"] - rows[0]["f1"]
    usable = first_stage(rows, lambda r: r["precision"] >= args.usable_precision and r["recall"] >= args.usable_recall and r["f1"] >= args.usable_f1)
    switch = first_stage(rows, lambda r: r["precision"] >= args.supervised_switch_precision and r["recall"] >= args.supervised_switch_recall and r["f1"] >= args.supervised_switch_f1)
    diminish, diminish_gain = diminishing_stage(rows, args.diminish_min_f1_gain)

    conclusion = {
        "precision_improved_from_S0_to_last": p_delta > 0,
        "recall_stable_or_improved_from_S0_to_last": r_delta >= -0.01,
        "f1_improved_from_S0_to_last": f_delta > 0,
        "first_usable_stage_under_assumption": usable,
        "first_full_supervised_switch_stage_under_assumption": switch,
        "first_diminishing_return_stage_by_f1_gain": diminish,
        "diminishing_return_gain": diminish_gain,
        "assumptions": {
            "usable": {"precision": args.usable_precision, "recall": args.usable_recall, "f1": args.usable_f1},
            "full_supervised_switch": {"precision": args.supervised_switch_precision, "recall": args.supervised_switch_recall, "f1": args.supervised_switch_f1},
            "diminishing_return_min_f1_gain": args.diminish_min_f1_gain,
        },
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(args.output_root),
        "stages": args.stages,
        "rows": rows,
        "deltas": {"precision": p_delta, "recall": r_delta, "f1": f_delta},
        "conclusion": conclusion,
    }

    mdir = args.output_root / "metrics" / "fewshot_curve"
    mdir.mkdir(parents=True, exist_ok=True)
    write_json(mdir / "fewshot_curve_summary.json", summary)
    with (mdir / "fewshot_curve_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| Stage | Policy | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms | Calib N/A | Test N/A |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['stage']} | {row['primary_policy']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['accuracy'])} | {fmt(row['auc_roc'])} | {fmt(row['auc_pr'])} | {fmt(row['inference_time_ms'])} | {row['calib_normal']}/{row['calib_anomaly']} | {row['test_normal']}/{row['test_anomaly']} |"
        )

    lines = [
        "# qm_xiepai Few-Shot CLIP-B16 + AHL + CHMM",
        "",
        "## First-Round Curve",
        "",
        *table,
        "",
        "## Trend Judgement",
        "",
        f"- Precision S0 -> {rows[-1]['stage']}: {fmt(rows[0]['precision'])} -> {fmt(rows[-1]['precision'])} (delta {fmt(p_delta)}).",
        f"- Recall S0 -> {rows[-1]['stage']}: {fmt(rows[0]['recall'])} -> {fmt(rows[-1]['recall'])} (delta {fmt(r_delta)}).",
        f"- F1 S0 -> {rows[-1]['stage']}: {fmt(rows[0]['f1'])} -> {fmt(rows[-1]['f1'])} (delta {fmt(f_delta)}).",
        f"- First usable stage under P/R/F1 >= {args.usable_precision:.2f}/{args.usable_recall:.2f}/{args.usable_f1:.2f}: {usable or 'not reached'}.",
        f"- First full-supervised switch stage under P/R/F1 >= {args.supervised_switch_precision:.2f}/{args.supervised_switch_recall:.2f}/{args.supervised_switch_f1:.2f}: {switch or 'not reached'}.",
        f"- First diminishing-return stage by F1 gain < {args.diminish_min_f1_gain:.2f}: {diminish or 'not observed'}.",
        "",
        "## Files",
        "",
        f"- Summary JSON: `{mdir / 'fewshot_curve_summary.json'}`",
        f"- Summary CSV: `{mdir / 'fewshot_curve_summary.csv'}`",
        f"- Output root: `{args.output_root}`",
    ]
    md_text = "\n".join(lines) + "\n"
    (mdir / "fewshot_curve_summary.md").write_text(md_text, encoding="utf-8")
    (args.output_root / "result.md").write_text(md_text, encoding="utf-8")

    doc_lines = [
        "# qm_xiepai Few-Shot CLIP-B16 + AHL + CHMM",
        "",
        "## Experiment Contract",
        "",
        "- Dataset alias: `models_qiumianxiepai`.",
        "- Original data: `/gdata1/huangjd/data/xidun_mvtec_format/models__球面斜拍`.",
        "- Split root: `/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot`.",
        "- Feature cache base: `/gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache`.",
        f"- Output root: `{args.output_root}`.",
        "- First round stages: `S0`, `S2`, `S4`, `S6`, `S8`.",
        "- Threshold rule: S0-S2 use calibration-normal percentiles; S3-S8 use calibration best-F1 as primary when calibration anomalies exist.",
        "",
        "## Results",
        "",
        *table,
        "",
        "## Conclusion",
        "",
        f"1. Precision improved from S0 to {rows[-1]['stage']}: `{conclusion['precision_improved_from_S0_to_last']}`.",
        f"2. Recall stable or improved from S0 to {rows[-1]['stage']}: `{conclusion['recall_stable_or_improved_from_S0_to_last']}`.",
        f"3. First usable stage under the stated threshold: `{usable or 'not reached'}`.",
        f"4. Diminishing-return stage by F1 gain: `{diminish or 'not observed'}`.",
        f"5. Full-supervised switch condition: `{switch or 'not reached'}`.",
        "6. CLIP-B16 + AHL + CHMM suitability is judged from the table above, not assumed a priori.",
        "7. Next-step candidates: DRA+CLIP score fusion, threshold calibration, and full-supervised baseline.",
        "",
        "## Reproducibility Commands",
        "",
        "```bash",
        "cd /ghome/huangjd/code/detected/adpretrain_bridge",
        "python3 prepare_qm_xiepai_fewshot_splits.py --seed 20260511",
        "python3 check_fewshot_splits.py",
        "python3 export_qm_xiepai_clip_features.py --output-base /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache --device cuda:0",
        f"python3 summarize_fewshot_curve.py --output-root {args.output_root}",
        "```",
    ]
    args.main_doc.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()