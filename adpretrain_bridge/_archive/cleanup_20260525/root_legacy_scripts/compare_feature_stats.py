import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Compare original AHL cache and ADPretrain cache stats.')
    p.add_argument('--original-root', default='/gdata1/huangjd/data/xidun_mvtec_format')
    p.add_argument('--candidate-root', default='/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_cache')
    p.add_argument('--classes', nargs='*', default=None)
    p.add_argument('--original-class', default=None)
    p.add_argument('--candidate-class', default=None)
    p.add_argument('--original-discover-prefix', default='models__')
    p.add_argument('--max-files', type=int, default=20)
    return p.parse_args()


def discover_class(root: Path, prefix: str):
    matches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f'No class directory starts with {prefix!r} under {root}')
    if len(matches) > 1:
        print(f'WARNING: multiple matches for prefix {prefix!r}, using {matches[0]!r}: {matches}')
    return matches[0]


def collect(root: Path, cls: str, sub: str, max_files: int):
    files = sorted((root / cls / sub).glob('**/*.npy'))[:max_files]
    vals = []
    for p in files:
        a = np.load(p)
        vals.append((a.shape, float(a.mean()), float(a.std()), float(np.linalg.norm(a.reshape(-1)))))
    return vals


def summarize(vals):
    if not vals:
        return 'no files'
    shapes = {}
    arr = np.array([[v[1], v[2], v[3]] for v in vals], dtype=np.float64)
    for shape, *_ in vals:
        shapes[shape] = shapes.get(shape, 0) + 1
    return {'shapes': shapes, 'mean': arr[:,0].mean(), 'std': arr[:,1].mean(), 'norm': arr[:,2].mean()}


def main():
    args = parse_args()
    orig = Path(args.original_root)
    cand = Path(args.candidate_root)
    if args.candidate_class:
        pairs = [(args.original_class or discover_class(orig, args.original_discover_prefix), args.candidate_class)]
    else:
        pairs = [(cls, cls) for cls in (args.classes or sorted([p.name for p in cand.iterdir() if p.is_dir()]))]
    for orig_cls, cand_cls in pairs:
        print(f'CLASS original={orig_cls} candidate={cand_cls}')
        for sub in ['feature/train/good', 'feature_scale/train/good', 'feature/test/good', 'feature_scale/test/good']:
            print(' ', sub)
            print('   original ', summarize(collect(orig, orig_cls, sub, args.max_files)))
            print('   candidate', summarize(collect(cand, cand_cls, sub, args.max_files)))


if __name__ == '__main__':
    main()
