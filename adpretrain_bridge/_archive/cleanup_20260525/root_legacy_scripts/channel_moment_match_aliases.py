import argparse
import json
from pathlib import Path

from channel_moment_match_cache import LINK_DIRS, NOAUG_ALIAS_PAIRS, SCALES, ensure_link, write_scale
from xidun_aliases import suffixed_aliases


def parse_args():
    p = argparse.ArgumentParser(description="Per-channel moment matching for Xidun ASCII alias cache.")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-alias-suffix", required=True)
    p.add_argument("--output-alias-suffix", required=True)
    p.add_argument("--alias-profile", default="xidun6", choices=["xidun6", "xidun3"])
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    source_pairs = dict(suffixed_aliases(args.source_alias_suffix, args.alias_profile))
    output_pairs = dict(suffixed_aliases(args.output_alias_suffix, args.alias_profile))
    meta = {
        "method": "per_channel_moment_match_train_good_to_reference",
        "source_root": str(args.source_root),
        "reference_root": str(args.reference_root),
        "output_root": str(args.output_root),
        "source_alias_suffix": args.source_alias_suffix,
        "output_alias_suffix": args.output_alias_suffix,
        "alias_profile": args.alias_profile,
        "eps": args.eps,
        "classes": {},
    }

    for real_class, source_alias in source_pairs.items():
        output_alias = output_pairs[real_class]
        src_class = args.source_root / source_alias
        ref_class = args.reference_root / real_class
        out_class = args.output_root / output_alias
        out_class.mkdir(parents=True, exist_ok=True)

        linked = {}
        for name in LINK_DIRS:
            src = src_class / name
            if src.exists():
                linked[name] = ensure_link(src, out_class / name)

        stats = {}
        for scale in SCALES:
            stats[scale] = write_scale(src_class, ref_class, out_class, scale, args.eps, args.skip_existing, out_class / "moment_stats")

        noaug_aliases = {}
        for alias_name, source_name in NOAUG_ALIAS_PAIRS:
            noaug_aliases[alias_name] = ensure_link(out_class / source_name, out_class / alias_name)

        meta["classes"][real_class] = {
            "source_alias": source_alias,
            "output_alias": output_alias,
            "linked_dirs": linked,
            "noaug_aliases": noaug_aliases,
            "scales": stats,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "channel_moment_match_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
