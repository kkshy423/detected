import argparse
import os
from pathlib import Path


ALIAS_PAIRS = {
    "noaug": [
        ("aug", "feature"),
        ("aug_scale", "feature_scale"),
    ],
    "task_aug": [
        ("aug_dream", "feature"),
        ("aug_dream_scale", "feature_scale"),
        ("aug_paste", "feature"),
        ("aug_paste_scale", "feature_scale"),
        ("aug_mix", "feature"),
        ("aug_mix_scale", "feature_scale"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create plain-feature alias directories required by AHL augmentation entry points."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=["noaug", "task_aug", "all"],
        default="noaug",
        help="noaug creates aug/aug_scale aliases; task_aug creates aug_dream/paste/mix aliases.",
    )
    return parser.parse_args()


def desired_pairs(mode):
    if mode == "all":
        return ALIAS_PAIRS["noaug"] + ALIAS_PAIRS["task_aug"]
    return ALIAS_PAIRS[mode]


def create_alias(class_root, alias_name, source_name):
    source = class_root / source_name
    alias = class_root / alias_name
    if not source.is_dir():
        raise FileNotFoundError(f"source directory missing: {source}")

    if alias.is_symlink():
        target = os.readlink(alias)
        resolved = (alias.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
        if resolved == source.resolve():
            return "exists"
        raise RuntimeError(f"alias conflict: {alias} -> {target}, expected {source}")

    if alias.exists():
        raise RuntimeError(f"refusing to overwrite existing non-symlink path: {alias}")

    os.symlink(str(source), str(alias), target_is_directory=True)
    return "created"


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    created = 0
    existing = 0

    for class_name in args.classes:
        class_root = root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"class directory missing: {class_root}")
        print(f"CLASS {class_name}")
        for alias_name, source_name in desired_pairs(args.mode):
            status = create_alias(class_root, alias_name, source_name)
            print(f"  {alias_name} -> {source_name}: {status}")
            if status == "created":
                created += 1
            else:
                existing += 1

    print(f"Alias preparation OK: created={created} existing={existing}")


if __name__ == "__main__":
    main()
