import argparse
import time
from pathlib import Path

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(desc, flush=True)
        return iterable

from common import (
    build_reference_features,
    build_transform,
    compress_four_to_two,
    encode_multiscale,
    ensure_dataset_links,
    iter_mvtec_images,
    list_image_files,
    load_image_tensor,
    make_encoder,
    make_projector,
    match_reference_features,
    residual_features,
    save_feature_pair,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description='Export ADPretrain projected residual features in AHL cache format.')
    p.add_argument('--adpretrain-root', default='/ghome/huangjd/code/detected/ADPretrain')
    p.add_argument('--dataset-root', default='/gdata1/huangjd/data/xidun_mvtec_format')
    p.add_argument('--output-root', default='/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_cache')
    p.add_argument('--backbone', default='dinov2-base', choices=['dinov2-base', 'dinov2-large', 'clip-base', 'clip-large', 'imagebind'])
    p.add_argument('--checkpoint', default='', help='ADPretrain projector checkpoint, e.g. checkpoints/dino-base/checkpoints_pro_angle.pth')
    p.add_argument('--classes', nargs='*', default=None)
    p.add_argument('--class-name', default=None, help='Single real source class name. If omitted, use --discover-prefix when --alias-class is set.')
    p.add_argument('--discover-prefix', default='models__', help='Prefix used to discover one source class without non-ASCII shell args.')
    p.add_argument('--alias-class', default=None, help='ASCII class name written under output root for a single-class export.')
    p.add_argument('--num-ref', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-images', type=int, default=0, help='Smoke-test limit per split/class; 0 means all images.')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--splits', nargs='+', default=['train', 'test'], help='Dataset splits to export, e.g. train val test.')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    if args.alias_class:
        if args.classes:
            raise ValueError('--alias-class is only supported for a single class; use --class-name or --discover-prefix.')
        if args.class_name:
            source_class = args.class_name
        else:
            matches = sorted(p.name for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith(args.discover_prefix))
            if not matches:
                raise FileNotFoundError(f'No class directory starts with {args.discover_prefix!r} under {dataset_root}')
            if len(matches) > 1:
                print(f'WARNING: multiple matches for prefix {args.discover_prefix!r}, using {matches[0]!r}: {matches}')
            source_class = matches[0]
        class_pairs = [(source_class, args.alias_class)]
    else:
        class_pairs = [(name, name) for name in (args.classes or sorted([p.name for p in dataset_root.iterdir() if p.is_dir()]))]

    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, args.adpretrain_root, device)
    projector = make_projector(encoder, args.adpretrain_root, args.checkpoint, device)

    meta = {
        'source_dataset_root': str(dataset_root),
        'output_root': str(output_root),
        'backbone': args.backbone,
        'checkpoint': args.checkpoint,
        'num_ref': args.num_ref,
        'device_requested': args.device,
        'device_used': str(device),
        'class_pairs': class_pairs,
        'feature_rule': 'projected residual; layer3/layer4 resized to 14x14 then averaged as feature; layer1/layer2 resized to 7x7 then averaged as feature_scale; first 512 channels retained as deterministic linear projection',
    }
    write_json(output_root / 'adpretrain_bridge_meta.json', meta)

    for class_name, output_class_name in class_pairs:
        src_class_root = dataset_root / class_name
        dst_class_root = output_root / output_class_name
        ensure_dataset_links(src_class_root, dst_class_root)
        ref_paths = list_image_files(src_class_root / 'train' / 'good')
        refs = build_reference_features(encoder, ref_paths, transform, device, args.num_ref)
        runtime_rows = []
        for split in args.splits:
            if not (src_class_root / split).exists():
                continue
            items = list(iter_mvtec_images(src_class_root, split))
            if args.limit_images > 0:
                items = items[:args.limit_images]
            for rel_no_suffix, img_path in tqdm(items, desc=f'{class_name}/{split}'):
                out_file = dst_class_root / 'feature' / f'{rel_no_suffix}.npy'
                out_scale = dst_class_root / 'feature_scale' / f'{rel_no_suffix}.npy'
                if args.skip_existing and out_file.exists() and out_scale.exists():
                    continue
                with torch.no_grad():
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    image_start = time.perf_counter()
                    image = load_image_tensor(img_path, transform, device)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    image_ms = (time.perf_counter() - image_start) * 1000.0

                    encoder_start = time.perf_counter()
                    features = encode_multiscale(encoder, image)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    encoder_ms = (time.perf_counter() - encoder_start) * 1000.0

                    match_start = time.perf_counter()
                    matched = match_reference_features(features, refs)
                    residual = residual_features(features, matched)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    match_residual_ms = (time.perf_counter() - match_start) * 1000.0

                    projector_start = time.perf_counter()
                    projected = projector(*residual)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    projector_ms = (time.perf_counter() - projector_start) * 1000.0

                    compress_start = time.perf_counter()
                    feature, feature_scale = compress_four_to_two(projected)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    compress_ms = (time.perf_counter() - compress_start) * 1000.0

                    save_start = time.perf_counter()
                    save_feature_pair(dst_class_root, rel_no_suffix, feature, feature_scale)
                    save_ms = (time.perf_counter() - save_start) * 1000.0
                    runtime_rows.append({
                        'image_load_transform_ms': image_ms,
                        'encoder_ms': encoder_ms,
                        'match_residual_ms': match_residual_ms,
                        'projector_ms': projector_ms,
                        'compress_ms': compress_ms,
                        'feature_full_ms': image_ms + encoder_ms + match_residual_ms + projector_ms + compress_ms,
                        'feature_save_ms': save_ms,
                    })
        def mean_field(name):
            return float(np.mean([row[name] for row in runtime_rows])) if runtime_rows else None
        write_json(dst_class_root / 'feature_runtime.json', {
            'time_kind': 'single_image_adpretrain_projected_feature_mean_ms',
            'time_adpretrain_feature_ms': mean_field('feature_full_ms'),
            'time_adpretrain_projected_feature_ms': mean_field('feature_full_ms'),
            'time_image_load_transform_ms': mean_field('image_load_transform_ms'),
            'time_adpretrain_encoder_ms': mean_field('encoder_ms'),
            'time_adpretrain_match_residual_ms': mean_field('match_residual_ms'),
            'time_adpretrain_projector_ms': mean_field('projector_ms'),
            'time_adpretrain_compress_ms': mean_field('compress_ms'),
            'time_feature_save_ms': mean_field('feature_save_ms'),
            'sample_count': len(runtime_rows),
            'note': 'Production timing covers image load/transform, ADPretrain encoder, reference matching, residual, projector, and compression to AHL feature tensors. Saving .npy files is recorded separately and excluded from production inference total.',
        })
    print(f'Export finished: {output_root}')


if __name__ == '__main__':
    main()
