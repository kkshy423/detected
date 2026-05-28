#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/20260518_policy_compare_eval_6_1_val_threshold/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python - <<'PY' 2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/20260518_policy_compare_eval_6_1_val_threshold/logs/policy_compare_eval.log
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from threshold_policies import apply_thresholds

ROOT = Path('/ghome/huangjd/code/detected/adpretrain_bridge')
OUT = ROOT / 'output/20260518_policy_compare_eval_6_1_val_threshold'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ('AHL_plain_ADPretrain', ROOT / 'output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518'),
    ('YOLO26s_cls', ROOT / 'output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517'),
    ('YOLO26s_cls', ROOT / 'output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260518_retry_s4_ampoff'),
]
POLICIES = ['production_normal_p95_0', 'strategy_stage_adaptive']
STAGES = [f'S{i}' for i in range(9)]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def auc_or_none(labels, scores, kind):
    try:
        if len(set(labels.tolist())) < 2:
            return None
        if kind == 'roc':
            return float(roc_auc_score(labels, scores))
        return float(average_precision_score(labels, scores))
    except Exception:
        return None


def load_scores(path: Path):
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f'empty scores file: {path}')
    fields = set(rows[0].keys())
    if 'split' in fields:
        calib = [r for r in rows if r.get('split') == 'val']
        test = [r for r in rows if r.get('split') == 'test']
    elif 'role' in fields:
        calib = [r for r in rows if str(r.get('role', '')).startswith('calib_')]
        test = [r for r in rows if str(r.get('role', '')).startswith('test_')]
    else:
        raise ValueError(f'unknown scores schema: {path}')
    def arr(part, col):
        return np.asarray([float(r[col]) for r in part], dtype=float)
    return arr(calib, 'label').astype(int), arr(calib, 'score'), arr(test, 'label').astype(int), arr(test, 'score')


def find_stage_metrics(root: Path, stage: str) -> Optional[Path]:
    candidates = [root / 'stages' / stage / 'metrics' / 'metrics.json', root / stage / 'metrics' / 'metrics.json']
    for p in candidates:
        if p.exists():
            return p
    found = list(root.glob(f'**/{stage}/metrics/metrics.json'))
    return found[0] if found else None


def normalize_policy_name(name: str) -> str:
    if name == 'production_normal_p95_0':
        return 'production_normal_p95'
    return name

rows: List[Dict] = []
detailed: List[Dict] = []
seen = set()
for method, src in SOURCES:
    for stage in STAGES:
        metrics_path = find_stage_metrics(src, stage)
        key = (method, stage)
        # Prefer retry/newer source if duplicate stage appears later and has a completed metrics file.
        if metrics_path is None:
            if key not in seen:
                rows.append({'method': method, 'stage': stage, 'policy': 'ALL', 'status': 'missing', 'reason': f'metrics not found under {src}'})
            continue
        try:
            metrics = read_json(metrics_path)
            status = metrics.get('status', 'ok')
            if status != 'ok':
                if key not in seen:
                    rows.append({'method': method, 'stage': stage, 'policy': 'ALL', 'status': status, 'reason': metrics.get('reason', '')})
                continue
            score_file = Path(metrics.get('score_file') or metrics_path.parent / 'scores.csv')
            if not score_file.exists():
                threshold_results = metrics.get('threshold_results') or {}
                missing = [p for p in POLICIES if p not in threshold_results]
                if missing:
                    raise FileNotFoundError(f'missing score file and policies {missing}: {score_file}')
                recalculated = threshold_results
                calib_labels = calib_scores = test_labels = test_scores = None
            else:
                calib_labels, calib_scores, test_labels, test_scores = load_scores(score_file)
                recalculated = apply_thresholds(calib_labels, calib_scores, test_labels, test_scores, stage=stage)
            # If duplicate, remove earlier rows for this method-stage so retry S4 supersedes old missing/failed.
            rows = [r for r in rows if not (r.get('method') == method and r.get('stage') == stage)]
            for policy in POLICIES:
                item = recalculated.get(policy) or recalculated.get('val_normal_p95_0' if policy == 'production_normal_p95_0' else policy)
                if not item:
                    rows.append({'method': method, 'stage': stage, 'policy': normalize_policy_name(policy), 'status': 'missing_policy', 'reason': f'{policy} not available'})
                    continue
                test = item.get('test', {})
                calib = item.get('calibration', {})
                row = {
                    'method': method,
                    'stage': stage,
                    'policy': normalize_policy_name(policy),
                    'status': 'ok',
                    'reason': '',
                    'threshold': test.get('threshold', calib.get('threshold')),
                    'accuracy': test.get('accuracy'),
                    'precision': test.get('precision'),
                    'recall': test.get('recall'),
                    'time_ms': metrics.get('time_ms') or metrics.get('inference_time_ms') or metrics.get('test_time_ms'),
                    'f1': test.get('f1'),
                    'tp': test.get('tp'),
                    'fp': test.get('fp'),
                    'tn': test.get('tn'),
                    'fn': test.get('fn'),
                    'auc_roc': metrics.get('auc_roc') if metrics.get('auc_roc') is not None else (auc_or_none(test_labels, test_scores, 'roc') if score_file.exists() else None),
                    'auc_pr': metrics.get('auc_pr') if metrics.get('auc_pr') is not None else (auc_or_none(test_labels, test_scores, 'pr') if score_file.exists() else None),
                    'calib_threshold': calib.get('threshold'),
                    'source_policy': calib.get('source_policy'),
                    'policy_family': calib.get('policy_family'),
                    'metrics_json': str(metrics_path),
                    'score_file': str(score_file),
                }
                rows.append(row)
            detailed.append({'method': method, 'stage': stage, 'metrics_json': str(metrics_path), 'recalculated_policies': list(recalculated.keys())})
            seen.add(key)
        except Exception as e:
            rows = [r for r in rows if not (r.get('method') == method and r.get('stage') == stage)]
            rows.append({'method': method, 'stage': stage, 'policy': 'ALL', 'status': 'failed', 'reason': repr(e), 'metrics_json': str(metrics_path)})

fields = ['method','stage','policy','status','reason','accuracy','precision','recall','time_ms','f1','tp','fp','tn','fn','auc_roc','auc_pr','threshold','calib_threshold','source_policy','policy_family','metrics_json','score_file']
with (OUT / 'policy_compare.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)

def fmt(v):
    if v is None or v == '':
        return ''
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)

lines = []
lines.append('# 20260518 Policy Compare Eval 6_1 Val Threshold')
lines.append('')
lines.append('???????? scores/metrics ???????????production_normal_p95 ????????strategy_stage_adaptive ?????????')
lines.append('')
lines.append('method stage policy status acc prec recall time_ms f1 tp fp tn fn auc_roc auc_pr')
for r in rows:
    lines.append(' '.join(fmt(r.get(k)) for k in ['method','stage','policy','status','accuracy','precision','recall','time_ms','f1','tp','fp','tn','fn','auc_roc','auc_pr']))
(OUT / 'policy_compare.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'policy_compare.json').write_text(json.dumps({'status':'ok','rows':rows,'details':detailed}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status':'ok','out':str(OUT),'rows':len(rows)}, ensure_ascii=False), flush=True)
PY
