#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
from statistics import mean
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute train-free precision/recall for AHL xidun runs from saved result.txt files.'
    )
    parser.add_argument('--dataset-root', default='/gdata1/huangjd/data/xidun_mvtec_format')
    parser.add_argument('--result-root', default='/ghome/huangjd/code/detected/AHL/trained_models_xidun')
    parser.add_argument('--summary-dir', default='/ghome/huangjd/code/detected/summary')
    parser.add_argument(
        '--n-anomaly',
        type=int,
        default=10,
        help='Used to reconstruct final test-set size: good + (all_defect - n_anomaly).',
    )
    return parser.parse_args()


def supported_count(folder: Path):
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() or path.is_symlink())


def eval_size_for_dataset(dataset_dir: Path, n_anomaly: int):
    test_dir = dataset_dir / 'test'
    good_count = supported_count(test_dir / 'good')
    defect_count = 0
    if test_dir.exists():
        for child in test_dir.iterdir():
            if child.is_dir() and child.name != 'good':
                defect_count += supported_count(child)
    eval_size = good_count + max(defect_count - n_anomaly, 0)
    return good_count, defect_count, eval_size


def read_last_epoch_scores(result_file: Path, eval_size: int):
    rows = []
    with result_file.open('r', encoding='utf-8', errors='replace') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                label = int(float(parts[0]))
                score = float(parts[1])
            except Exception:
                continue
            rows.append((label, score))

    if eval_size <= 0:
        raise ValueError(f'invalid eval_size={eval_size} for {result_file}')
    if len(rows) < eval_size:
        raise ValueError(f'not enough rows in {result_file}: {len(rows)} < {eval_size}')

    last_rows = rows[-eval_size:]
    labels = [label for label, _ in last_rows]
    scores = [score for _, score in last_rows]
    return labels, scores, len(rows)


def best_f1_metrics(labels, scores):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    best = {
        'threshold': None,
        'precision': 0.0,
        'recall': 0.0,
        'f1': -1.0,
    }

    for index in range(len(thresholds)):
        precision = float(precisions[index + 1])
        recall = float(recalls[index + 1])
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best['f1']:
            best = {
                'threshold': float(thresholds[index]),
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

    preds = [1 if score >= best['threshold'] else 0 for score in scores]
    best['precision_check'] = float(precision_score(labels, preds, zero_division=0))
    best['recall_check'] = float(recall_score(labels, preds, zero_division=0))
    return best


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    result_root = Path(args.result_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_csv = summary_dir / 'ahl_xidun_trainfree_pr_metrics.csv'
    out_md = summary_dir / 'ahl_xidun_trainfree_pr_metrics.md'
    out_json = summary_dir / 'ahl_xidun_trainfree_pr_metrics.json'

    rows = []
    for dataset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        result_file = result_root / ('AHL_MVTec_' + dataset_dir.name) / 'result.txt'
        if not result_file.exists():
            continue

        good_count, defect_count, eval_size = eval_size_for_dataset(dataset_dir, args.n_anomaly)
        labels, scores, total_rows = read_last_epoch_scores(result_file, eval_size)
        auc_roc = float(roc_auc_score(labels, scores))
        auc_pr = float(average_precision_score(labels, scores))
        best = best_f1_metrics(labels, scores)

        rows.append({
            'dataset': dataset_dir.name,
            'good_count': good_count,
            'defect_count': defect_count,
            'eval_size': eval_size,
            'result_rows': total_rows,
            'epochs_like': total_rows / eval_size,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'best_f1_threshold': best['threshold'],
            'precision_best_f1': best['precision'],
            'recall_best_f1': best['recall'],
            'f1_best_f1': best['f1'],
        })

    rows.sort(key=lambda item: item['dataset'])

    fieldnames = [
        'dataset', 'good_count', 'defect_count', 'eval_size', 'result_rows', 'epochs_like',
        'auc_roc', 'auc_pr', 'best_f1_threshold', 'precision_best_f1', 'recall_best_f1', 'f1_best_f1',
    ]
    with out_csv.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    mean_auc_roc = mean(row['auc_roc'] for row in rows) if rows else None
    mean_auc_pr = mean(row['auc_pr'] for row in rows) if rows else None
    mean_precision = mean(row['precision_best_f1'] for row in rows) if rows else None
    mean_recall = mean(row['recall_best_f1'] for row in rows) if rows else None
    mean_f1 = mean(row['f1_best_f1'] for row in rows) if rows else None

    lines = []
    lines.append('# AHL xidun train-free Precision / Recall 汇总')
    lines.append('')
    lines.append('- 说明：本报告完全基于各子类现有 `result.txt` 的最后一轮测试分数生成，不需要重训、不需要显卡。')
    lines.append('- 阈值策略：对每个子类在测试集上搜索 **best-F1 threshold**，据此给出 `precision_best_f1` 和 `recall_best_f1`。')
    lines.append('- 注意：这是带标签的后验最佳阈值结果，适合离线分析，不等同于部署时固定阈值表现。')
    lines.append('')
    lines.append('## 均值')
    lines.append('')
    lines.append('| 子类数 | 平均 AUC-ROC | 平均 AUC-PR | 平均 Precision | 平均 Recall | 平均 F1 |')
    lines.append('| ---: | ---: | ---: | ---: | ---: | ---: |')
    lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
        len(rows), mean_auc_roc, mean_auc_pr, mean_precision, mean_recall, mean_f1
    ))
    lines.append('')
    lines.append('## 各子类')
    lines.append('')
    lines.append('| 子类 | AUC-ROC | AUC-PR | Precision | Recall | F1 | Threshold | Eval Size |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        lines.append(
            '| {dataset} | {auc_roc:.4f} | {auc_pr:.4f} | {precision_best_f1:.4f} | {recall_best_f1:.4f} | {f1_best_f1:.4f} | {best_f1_threshold:.6f} | {eval_size} |'.format(**row)
        )
    lines.append('')
    lines.append('## 输出文件')
    lines.append('')
    lines.append('- JSON: `{}`'.format(out_json))
    lines.append('- Markdown: `{}`'.format(out_md))
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('WROTE', out_json)
    print('WROTE', out_md)


if __name__ == '__main__':
    main()
