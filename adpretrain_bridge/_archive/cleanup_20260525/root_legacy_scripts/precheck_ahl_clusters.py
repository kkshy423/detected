#!/usr/bin/env python3
"""Precheck AHL normal-cluster viability and write per-class cluster_num choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose safe AHL cluster_num values per cached class.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--preferred-cluster-num", type=int, default=3)
    parser.add_argument("--fallback-cluster-num", type=int, default=2)
    parser.add_argument("--min-effective-clusters", type=int, default=2)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def effective_count(root: Path, class_name: str, cluster_num: int, random_state: int, min_cluster_size: int) -> dict:
    files = sorted((root / class_name / "feature" / "train" / "good").glob("*.npy"))
    if len(files) < cluster_num:
        return {"file_count": len(files), "cluster_num": cluster_num, "counts": [], "effective_clusters": 0}
    arrays = [np.load(path, mmap_mode="r").reshape(-1) for path in files]
    x = np.stack(arrays)
    labels = KMeans(n_clusters=cluster_num, random_state=random_state, n_init=10).fit(x).labels_
    counts = np.bincount(labels, minlength=cluster_num).astype(int).tolist()
    return {
        "file_count": len(files),
        "cluster_num": cluster_num,
        "counts": counts,
        "effective_clusters": int(sum(count >= min_cluster_size for count in counts)),
    }


def main() -> None:
    args = parse_args()
    out = {
        "dataset_root": str(args.dataset_root),
        "preferred_cluster_num": args.preferred_cluster_num,
        "fallback_cluster_num": args.fallback_cluster_num,
        "min_effective_clusters": args.min_effective_clusters,
        "min_cluster_size": args.min_cluster_size,
        "classes": {},
    }
    for class_name in args.classes:
        preferred = effective_count(args.dataset_root, class_name, args.preferred_cluster_num, args.random_state, args.min_cluster_size)
        chosen = args.preferred_cluster_num
        fallback = None
        reason = "preferred_ok"
        if preferred["effective_clusters"] < args.min_effective_clusters:
            fallback = effective_count(args.dataset_root, class_name, args.fallback_cluster_num, args.random_state, args.min_cluster_size)
            chosen = args.fallback_cluster_num
            reason = "fallback_due_to_insufficient_effective_clusters"
        out["classes"][class_name] = {
            "chosen_cluster_num": chosen,
            "reason": reason,
            "preferred": preferred,
            "fallback": fallback,
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
