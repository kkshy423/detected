#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive superseded qm_xiepai experiment artifacts without deleting results."""

import fnmatch
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
ARCHIVE_ROOT = PROJECT_ROOT / "archive" / "20260513_yolo_recovery_cleanup"

OUTPUT_DIRS = {
    "output/qm_xiepai_yolo26s_cls_train_test_20260512": "superseded YOLO run before dependency fixes",
    "output/qm_xiepai_yolo26s_cls_train_test_20260512_retry1": "superseded YOLO retry before dependency fixes",
    "output/qm_xiepai_yolo26s_cls_train_test_20260512_retry2": "superseded YOLO retry before retry3 weights",
    "output/qm_xiepai_adpretrain_only_clip_b16_plain_train_test_20260512": "older non-official ADPretrain-only summary; official output is kept active",
    "output/qm_xiepai_fewshot_clip_b16_chmm_20260511": "legacy calibration-split CHMM run; current split is train/test-only",
}

DOC_FILES = {
    "qm_xiepai_fewshot_clip_b16_chmm.md": "legacy calibration-split report; current train/test-only docs remain active",
}

PBS_PATTERNS = {
    "pbs/generated_run": [
        "run_qm_xiepai_yolo26s_cls_train_test_S*_20260512.sh",
        "run_qm_xiepai_yolo26s_cls_train_test_S*_20260512_retry1.sh",
        "run_qm_xiepai_yolo26s_cls_train_test_S*_20260512_retry2.sh",
    ],
    "pbs/generated_submit": [
        "qm_xiepai_yolo26s_cls_train_test_S*_20260512.pbs",
        "qm_xiepai_yolo26s_cls_train_test_S*_20260512_retry1.pbs",
        "qm_xiepai_yolo26s_cls_train_test_S*_20260512_retry2.pbs",
    ],
}


def safe_move(rel_path, reason):
    src = PROJECT_ROOT / rel_path
    if not src.exists():
        return {"path": rel_path, "status": "missing", "reason": reason}
    dst = ARCHIVE_ROOT / rel_path
    if dst.exists():
        return {"path": rel_path, "status": "archive_target_exists", "reason": reason}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"path": rel_path, "archived_to": str(dst), "status": "archived", "reason": reason}


def iter_pbs_matches(root_rel, patterns):
    root = PROJECT_ROOT / root_rel
    if not root.exists():
        return
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            yield path.relative_to(PROJECT_ROOT).as_posix()


def write_current_index(manifest_path):
    lines = [
        "# qm_xiepai Current Result Index",
        "",
        "Updated: 2026-05-13",
        "",
        "## Active Outputs",
        "",
        "- ADPretrain-only official fixed-test: `output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512`",
        "- ADPretrain-only official full-original: `output/qm_xiepai_adpretrain_only_clip_b16_official_full_original_20260513`",
        "- AHL plain no-CHMM train/test: `output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512`",
        "- AHL + CHMM train/test ablation: `output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512`",
        "- YOLO retry3 source weights: `output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3`",
        "- YOLO retry3 recovered eval: `output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval`",
        "",
        "## Active Docs",
        "",
        "- Train/test CHMM curve: `qm_xiepai_fewshot_clip_b16_chmm_train_test.md`",
        "- Plain vs CHMM and split/full comparison: `qm_xiepai_plain_vs_chmm_split_vs_full_20260512.md`",
        "- Current index: `qm_xiepai_current_results_20260513.md`",
        "",
        "## Archived",
        "",
        f"- Archive root: `{ARCHIVE_ROOT}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "No experiment output was deleted by this cleanup; superseded artifacts were moved under `archive/`.",
        "",
    ]
    (PROJECT_ROOT / "qm_xiepai_current_results_20260513.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    actions = []
    for rel_path, reason in OUTPUT_DIRS.items():
        actions.append(safe_move(rel_path, reason))
    for rel_path, reason in DOC_FILES.items():
        actions.append(safe_move(rel_path, reason))
    for root_rel, patterns in PBS_PATTERNS.items():
        for rel_path in iter_pbs_matches(root_rel, patterns):
            actions.append(safe_move(rel_path, "superseded YOLO PBS launcher/submit file from failed pre-retry3 attempts"))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archive_root": str(ARCHIVE_ROOT),
        "policy": "archive only; no deletion",
        "actions": actions,
    }
    manifest_path = ARCHIVE_ROOT / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Archive Manifest",
        "",
        f"Archive root: `{ARCHIVE_ROOT}`",
        "",
        "| Status | Path | Reason |",
        "|---|---|---|",
    ]
    for item in actions:
        md_lines.append(f"| {item['status']} | `{item['path']}` | {item['reason']} |")
    (ARCHIVE_ROOT / "archive_manifest.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_current_index(manifest_path)
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
