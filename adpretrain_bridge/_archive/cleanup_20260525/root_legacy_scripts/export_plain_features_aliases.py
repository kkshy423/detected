import argparse
from pathlib import Path

import torch
from tqdm import tqdm

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
from xidun_aliases import suffixed_aliases


def parse_args():
    p = argparse.ArgumentParser(description="Export ADPretrain features for Xidun classes with ASCII aliases.")
    p.add_argument("--adpretrain-root", default="/ghome/huangjd/code/detected/ADPretrain")
    p.add_argument("--dataset-root", default="/gdata1/huangjd/data/xidun_mvtec_format")
    p.add_argument("--output-root", required=True)
    p.add_argument("--backbone", default="clip-base", choices=["clip-base", "dinov2-base", "dinov2-large"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--alias-suffix", required=True)
    p.add_argument("--alias-profile", default="xidun6", choices=["xidun6", "xidun3"])
    p.add_argument("--num-ref", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    class_pairs = suffixed_aliases(args.alias_suffix, args.alias_profile)

    transform = build_transform(args.backbone)
    encoder = make_encoder(args.backbone, args.adpretrain_root, device)
    projector = make_projector(encoder, args.adpretrain_root, args.checkpoint, device)

    write_json(
        output_root / "adpretrain_bridge_meta.json",
        {
            "source_dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "backbone": args.backbone,
            "checkpoint": args.checkpoint,
            "alias_profile": args.alias_profile,
            "num_ref": args.num_ref,
            "class_pairs": class_pairs,
            "feature_rule": "projected residual; layer3/layer4 -> feature (512,14,14); layer1/layer2 -> feature_scale (512,7,7)",
        },
    )

    for real_class, alias_class in class_pairs:
        src_class_root = dataset_root / real_class
        dst_class_root = output_root / alias_class
        ensure_dataset_links(src_class_root, dst_class_root)
        ref_paths = list_image_files(src_class_root / "train" / "good")
        refs = build_reference_features(encoder, ref_paths, transform, device, args.num_ref)
        for split in ["train", "test"]:
            for rel_no_suffix, img_path in tqdm(list(iter_mvtec_images(src_class_root, split)), desc=f"{alias_class}/{split}"):
                out_file = dst_class_root / "feature" / f"{rel_no_suffix}.npy"
                out_scale = dst_class_root / "feature_scale" / f"{rel_no_suffix}.npy"
                if args.skip_existing and out_file.exists() and out_scale.exists():
                    continue
                with torch.no_grad():
                    image = load_image_tensor(img_path, transform, device)
                    features = encode_multiscale(encoder, image)
                    matched = match_reference_features(features, refs)
                    residual = residual_features(features, matched)
                    projected = projector(*residual)
                    feature, feature_scale = compress_four_to_two(projected)
                    save_feature_pair(dst_class_root, rel_no_suffix, feature, feature_scale)
    print(f"Export finished: {output_root}")


if __name__ == "__main__":
    main()
