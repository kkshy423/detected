import argparse
import os
from pathlib import Path

import numpy as np

IMAGE_EXTS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(description='Check ADPretrain cache compatibility with AHL.')
    p.add_argument('--dataset-root', default='/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_cache')
    p.add_argument('--classes', nargs='*', default=None)
    p.add_argument('--allow-partial', action='store_true', help='Allow missing cache files for smoke tests.')
    return p.parse_args()


def image_stems(root: Path, split: str):
    out = []
    for defect in sorted((root / split).iterdir(), key=lambda p: p.name):
        if not defect.is_dir():
            continue
        for p in sorted(defect.iterdir(), key=lambda x: x.name):
            if p.suffix.lower() in IMAGE_EXTS:
                out.append(str(Path(split) / defect.name / p.stem))
    return out


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    classes = args.classes or sorted([p.name for p in root.iterdir() if p.is_dir()])
    failures = []
    for cls in classes:
        cr = root / cls
        print(f'CLASS {cls}')
        for split in ['train', 'test']:
            stems = image_stems(cr, split)
            miss = []
            shapes = {}
            scale_shapes = {}
            for stem in stems:
                fp = cr / 'feature' / f'{stem}.npy'
                sp = cr / 'feature_scale' / f'{stem}.npy'
                if not fp.exists() or not sp.exists():
                    miss.append(stem)
                    continue
                a = np.load(fp, mmap_mode='r')
                b = np.load(sp, mmap_mode='r')
                shapes[a.shape] = shapes.get(a.shape, 0) + 1
                scale_shapes[b.shape] = scale_shapes.get(b.shape, 0) + 1
                if a.shape != (512, 14, 14) or b.shape != (512, 7, 7):
                    failures.append((cls, split, stem, a.shape, b.shape))
            print(f'  {split}: images={len(stems)} missing={len(miss)} feature_shapes={shapes} scale_shapes={scale_shapes}')
            if miss[:5]:
                print('  missing examples:', miss[:5])
            if miss and not args.allow_partial:
                failures.append((cls, split, 'missing', len(miss), 'files'))
    if failures:
        print('FAILED shape checks:')
        for item in failures[:20]:
            print(item)
        raise SystemExit(1)
    print('Cache check OK')


if __name__ == '__main__':
    main()