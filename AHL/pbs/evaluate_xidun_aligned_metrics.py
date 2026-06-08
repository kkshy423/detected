#!/usr/bin/env python3
"""Evaluate AHL xidun runs against aligned project metrics.

This script is train-free. By default it reads existing AHL result.txt files and
computes image-level classification metrics from the last evaluation block.
Optionally, --benchmark-model-inference loads trained AHL checkpoints and times
feature-level model forward passes. It does not retrain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

DEFAULT_DATASET_ROOT = Path('/gdata1/huangjd/data/xidun_mvtec_format')
DEFAULT_RESULT_ROOT = Path('/ghome/huangjd/code/detected/AHL/trained_models_xidun')
DEFAULT_SUMMARY_DIR = Path('/ghome/huangjd/code/detected/summary')
DEFAULT_AHL_ROOT = Path('/ghome/huangjd/code/detected/AHL')

CLASSIFICATION_ACC_TARGET = 0.98
INFERENCE_TIME_TARGET_MS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train-free AHL aligned metric evaluator for xidun experiments.')
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument('--summary-dir', type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument('--ahl-root', type=Path, default=DEFAULT_AHL_ROOT)
    parser.add_argument('--n-anomaly', type=int, default=10)
    parser.add_argument('--threshold-strategy', choices=['best-f1', 'best-accuracy'], default='best-f1')
    parser.add_argument('--benchmark-model-inference', action='store_true', help='Load checkpoints and time AHL model forward passes. This can be slow.')
    parser.add_argument('--device', default='cuda', help='Used only with --benchmark-model-inference. Examples: cuda, cpu')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images; used only for model timing mode.')
    return parser.parse_args()


def supported_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() or path.is_symlink())


def eval_size_for_dataset(dataset_dir: Path, n_anomaly: int) -> Tuple[int, int, int]:
    test_dir = dataset_dir / 'test'
    good_count = supported_count(test_dir / 'good')
    defect_count = 0
    if test_dir.exists():
        for child in test_dir.iterdir():
            if child.is_dir() and child.name != 'good':
                defect_count += supported_count(child)
    eval_size = good_count + max(defect_count - n_anomaly, 0)
    return good_count, defect_count, eval_size


def read_last_eval_block(result_file: Path, eval_size: int) -> Tuple[np.ndarray, np.ndarray, int]:
    rows = []
    with result_file.open('r', encoding='utf-8', errors='replace') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                rows.append((int(float(parts[0])), float(parts[1])))
            except ValueError:
                continue
    if eval_size <= 0:
        raise ValueError(f'invalid eval_size={eval_size} for {result_file}')
    if len(rows) < eval_size:
        raise ValueError(f'not enough result rows in {result_file}: {len(rows)} < {eval_size}')
    last_rows = rows[-eval_size:]
    labels = np.asarray([label for label, _ in last_rows], dtype=np.int64)
    scores = np.asarray([score for _, score in last_rows], dtype=np.float64)
    return labels, scores, len(rows)


def choose_threshold(labels: np.ndarray, scores: np.ndarray, strategy: str) -> Tuple[float, Dict[str, float]]:
    _, _, thresholds = precision_recall_curve(labels, scores)
    best_threshold = float(thresholds[0]) if len(thresholds) else float(scores.max())
    best_score = -1.0
    best_metrics: Dict[str, float] = {}
    search_thresholds = thresholds if len(thresholds) else np.asarray([best_threshold])
    for threshold in search_thresholds:
        preds = (scores >= threshold).astype(np.int64)
        accuracy = float(accuracy_score(labels, preds))
        precision = float(precision_score(labels, preds, zero_division=0))
        recall = float(recall_score(labels, preds, zero_division=0))
        f1 = float(f1_score(labels, preds, zero_division=0))
        selector = f1 if strategy == 'best-f1' else accuracy
        if selector > best_score:
            best_score = selector
            best_threshold = float(threshold)
            best_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    if not best_metrics:
        preds = np.zeros_like(labels)
        best_metrics = {
            'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
        }
    return best_threshold, best_metrics


def evaluate_from_result(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_dir in sorted(path for path in args.dataset_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        result_file = args.result_root / f'AHL_MVTec_{dataset}' / 'result.txt'
        if not result_file.exists():
            rows.append({'dataset': dataset, 'status': 'missing_result_txt'})
            continue
        good_count, defect_count, eval_size = eval_size_for_dataset(dataset_dir, args.n_anomaly)
        labels, scores, result_rows = read_last_eval_block(result_file, eval_size)
        threshold, threshold_metrics = choose_threshold(labels, scores, args.threshold_strategy)
        auc_roc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None
        auc_pr = float(average_precision_score(labels, scores)) if len(set(labels.tolist())) == 2 else None
        rows.append({
            'dataset': dataset,
            'status': 'ok',
            'eval_mode': 'from_existing_result_txt_last_block',
            'threshold_strategy': args.threshold_strategy,
            'threshold': threshold,
            'classification_accuracy': threshold_metrics['accuracy'],
            'precision': threshold_metrics['precision'],
            'recall': threshold_metrics['recall'],
            'f1': threshold_metrics['f1'],
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'good_count': good_count,
            'defect_count_total': defect_count,
            'eval_size': eval_size,
            'result_rows': result_rows,
            'epochs_like': result_rows / eval_size if eval_size else None,
            'classification_accuracy_target': CLASSIFICATION_ACC_TARGET,
            'classification_accuracy_pass': threshold_metrics['accuracy'] >= CLASSIFICATION_ACC_TARGET,
            'detection_map': None,
            'detection_map_target': 0.95,
            'detection_map_pass': None,
            'segmentation_iou': None,
            'segmentation_iou_target': 0.7,
            'segmentation_iou_pass': None,
            'inference_time_ms': None,
            'inference_time_target_ms': INFERENCE_TIME_TARGET_MS,
            'inference_time_pass': None,
            'note': 'AHL is image-level anomaly scoring; detection mAP and segmentation IoU are not applicable. result.txt cannot measure real model inference time.',
        })
    return rows


def parse_setting_file(setting_file: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}
    with setting_file.open('r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if ' : ' not in line:
                continue
            key, value = line.rstrip('\n').split(' : ', 1)
            if value == 'None':
                values[key] = None
            elif value in {'True', 'False'}:
                values[key] = value == 'True'
            else:
                try:
                    values[key] = int(value)
                except ValueError:
                    try:
                        values[key] = float(value)
                    except ValueError:
                        values[key] = value
    return values


def benchmark_model_inference(args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    import torch
    sys.path.insert(0, str(args.ahl_root))
    from main import Trainer, setup_seed  # type: ignore
    for row in rows:
        if row.get('status') != 'ok':
            continue
        dataset = str(row['dataset'])
        run_dir = args.result_root / f'AHL_MVTec_{dataset}'
        setting_file = run_dir / 'setting.txt'
        checkpoint = run_dir / f'{dataset}_ctest.pkl'
        if not setting_file.exists() or not checkpoint.exists():
            row['inference_note'] = 'missing setting.txt or checkpoint'
            continue
        settings = parse_setting_file(setting_file)
        settings['dataset_root'] = str(args.dataset_root)
        settings['experiment_dir'] = str(run_dir)
        settings['classname'] = dataset
        settings['feat_classname'] = dataset
        settings['AHL'] = False
        settings['auxiliary'] = False
        settings['episode_num'] = max(int(settings.get('episode_num', 1) or 1), 1)
        settings['cluster_num'] = max(int(settings.get('cluster_num', 1) or 1), 1)
        settings['no_cuda'] = args.device == 'cpu'
        settings['cuda'] = args.device != 'cpu' and torch.cuda.is_available()
        settings['workers'] = 0
        setup_seed(int(settings.get('ramdn_seed', 42)))
        trainer_args = SimpleNamespace(**settings)
        trainer = Trainer(trainer_args)
        trainer.model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
        if trainer_args.cuda:
            trainer.model = trainer.model.cuda()
            trainer.criterion = trainer.criterion.cuda()
            trainer.mse_loss = trainer.mse_loss.cuda()
        trainer.model.eval()
        times_ms: List[float] = []
        processed = 0
        with torch.no_grad():
            for sample in trainer.test_loader:
                image, image_scale, target = sample['image'], sample['image_scale'], sample['label']
                if trainer_args.cuda:
                    image, image_scale, target = image.cuda(), image_scale.cuda(), target.cuda()
                if trainer_args.total_heads == 4:
                    try:
                        ref_image = next(trainer.ref)['image']
                        ref_image_scale = next(trainer.ref)['image_scale']
                    except StopIteration:
                        trainer.ref = iter(trainer.ref_loader)
                        ref_image = next(trainer.ref)['image']
                        ref_image_scale = next(trainer.ref)['image_scale']
                    if trainer_args.cuda:
                        ref_image = ref_image.cuda()
                        ref_image_scale = ref_image_scale.cuda()
                    image = torch.cat([ref_image, image], dim=0)
                    image_scale = torch.cat([ref_image_scale, image_scale], dim=0)
                if trainer_args.cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = trainer.model.forward(image=image, image_scale=image_scale, label=target, var=trainer.model.parameters())
                if trainer_args.cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if processed >= args.warmup:
                    times_ms.append(elapsed_ms / max(int(target.numel()), 1))
                processed += int(target.numel())
                if args.max_images and processed >= args.max_images:
                    break
        if times_ms:
            avg_ms = float(mean(times_ms))
            row['inference_time_ms'] = avg_ms
            row['inference_time_pass'] = avg_ms <= INFERENCE_TIME_TARGET_MS
            row['inference_note'] = 'feature-level AHL forward time; excludes DRA feature extraction and disk I/O'


def fmt(value: object) -> str:
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'PASS' if value else 'FAIL'
    if isinstance(value, float):
        if math.isnan(value):
            return 'N/A'
        return f'{value:.4f}'
    return str(value)


def write_outputs(rows: List[Dict[str, object]], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / 'ahl_aligned_metrics_full.json'
    md_path = summary_dir / 'ahl_aligned_metrics_full.md'
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    valid_rows = [row for row in rows if row.get('status') == 'ok']

    def mean_of(key: str) -> Optional[float]:
        values = [row.get(key) for row in valid_rows if isinstance(row.get(key), (int, float)) and row.get(key) is not None]
        return float(mean(values)) if values else None

    lines = [
        '# AHL Aligned Metrics',
        '',
        '## Mean',
        '',
        '| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        '| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            len(valid_rows),
            fmt(mean_of('classification_accuracy')),
            fmt(mean_of('precision')),
            fmt(mean_of('recall')),
            fmt(mean_of('f1')),
            fmt(mean_of('auc_roc')),
            fmt(mean_of('auc_pr')),
            fmt(mean_of('inference_time_ms')),
        ),
        '',
        '## Per Class',
        '',
        '| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in rows:
        lines.append('| {dataset} | {acc} | {acc_pass} | {precision} | {recall} | {f1} | {auc_roc} | {auc_pr} | {time_ms} | {time_pass} |'.format(
            dataset=row.get('dataset'),
            acc=fmt(row.get('classification_accuracy')),
            acc_pass=fmt(row.get('classification_accuracy_pass')) if row.get('classification_accuracy_pass') is not None else 'N/A',
            precision=fmt(row.get('precision')),
            recall=fmt(row.get('recall')),
            f1=fmt(row.get('f1')),
            auc_roc=fmt(row.get('auc_roc')),
            auc_pr=fmt(row.get('auc_pr')),
            time_ms=fmt(row.get('inference_time_ms')),
            time_pass=fmt(row.get('inference_time_pass')) if row.get('inference_time_pass') is not None else 'N/A',
        ))
    lines.extend([
        '',
        '## Scope',
        '',
        '- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time',
        '- Not alignable: detection mAP, segmentation IoU',
        '',
        '## Files',
        '',
        f'- JSON: `{json_path}`',
        f'- Markdown: `{md_path}`',
    ])
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'WROTE {json_path}')
    print(f'WROTE {md_path}')

def main() -> None:
    args = parse_args()
    rows = evaluate_from_result(args)
    if args.benchmark_model_inference:
        benchmark_model_inference(args, rows)
    write_outputs(rows, args.summary_dir)


if __name__ == '__main__':
    main()
