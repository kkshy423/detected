# -*- coding: utf-8 -*-
"""Shared helpers for the qm_xiepai few-shot AHL bridge scripts."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}

SOURCE_CLASS_ROOT = Path("/gdata1/huangjd/data/xidun_mvtec_format/models__球面斜拍")
SOURCE_DATASET_ROOT = SOURCE_CLASS_ROOT.parent
REFERENCE_DATASET_ROOT = SOURCE_DATASET_ROOT
ALIAS_CLASS = "models_qiumianxiepai"
SPLIT_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot")
CACHE_BASE = Path("/gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache")
BRIDGE_ROOT = Path("/ghome/huangjd/code/detected/adpretrain_bridge")
AHL_ROOT = Path("/ghome/huangjd/code/detected/AHL")
ADPRETRAIN_ROOT = Path("/ghome/huangjd/code/detected/ADPretrain")
OFFICIAL_CLIP_PROJECTOR = ADPRETRAIN_ROOT / "checkpoints/clip-base/checkpoints_img_angle.pth"
MAIN_DOC = BRIDGE_ROOT / "qm_xiepai_fewshot_clip_b16_chmm.md"


class StageSpec(object):
    def __init__(self, stage, available_normal, available_anomaly, train_normal, train_anomaly, calib_normal, calib_anomaly):
        self.stage = stage
        self.available_normal = available_normal
        self.available_anomaly = available_anomaly
        self.train_normal = train_normal
        self.train_anomaly = train_anomaly
        self.calib_normal = calib_normal
        self.calib_anomaly = calib_anomaly

    def as_dict(self):
        return {
            "stage": self.stage,
            "available_normal": self.available_normal,
            "available_anomaly": self.available_anomaly,
            "train_normal": self.train_normal,
            "train_anomaly": self.train_anomaly,
            "calib_normal": self.calib_normal,
            "calib_anomaly": self.calib_anomaly,
        }


STAGE_SPECS = {
    "S0": StageSpec("S0", 50, 0, 40, 0, 10, 0),
    "S1": StageSpec("S1", 100, 1, 80, 1, 20, 0),
    "S2": StageSpec("S2", 150, 3, 120, 3, 30, 0),
    "S3": StageSpec("S3", 200, 5, 160, 4, 40, 1),
    "S4": StageSpec("S4", 300, 10, 240, 8, 60, 2),
    "S5": StageSpec("S5", 400, 20, 320, 16, 80, 4),
    "S6": StageSpec("S6", 500, 40, 400, 32, 100, 8),
    "S7": StageSpec("S7", 560, 80, 448, 64, 112, 16),
    "S8": StageSpec("S8", 560, 139, 448, 111, 112, 28),
}
FIRST_ROUND_STAGES = ["S0", "S2", "S4", "S6", "S8"]

ROLE_TO_FILE = {
    "train_normal": "train_normal.txt",
    "train_anomaly": "train_anomaly.txt",
    "calib_normal": "calib_normal.txt",
    "calib_anomaly": "calib_anomaly.txt",
}

ROLE_TO_STAGE_DIR = {
    "train_normal": "train/good",
    "train_anomaly": "test/train_defect",
    "calib_normal": "test/good",
    "calib_anomaly": "test/defect",
    "test_normal": "test/good",
    "test_anomaly": "test/defect",
}

AUG_ALIAS_PAIRS = [
    ("aug", "feature"),
    ("aug_scale", "feature_scale"),
    ("aug_dream", "feature"),
    ("aug_dream_scale", "feature_scale"),
    ("aug_paste", "feature"),
    ("aug_paste_scale", "feature_scale"),
    ("aug_mix", "feature"),
    ("aug_mix_scale", "feature_scale"),
]


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def is_supported_path(path):
    return path.suffix.lower() in IMAGE_EXTS


def list_supported_entries(folder):
    if not folder.exists():
        raise FileNotFoundError(folder)
    return sorted(
        [p for p in folder.iterdir() if is_supported_path(p) and (p.is_file() or p.is_symlink())],
        key=lambda p: natural_key(p.name),
    )


def rel_list(folder, base):
    return [str(path.relative_to(base)).replace(os.sep, "/") for path in list_supported_entries(folder)]


def read_lines(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stage_dir(split_root, stage):
    return split_root / stage


def stage_spec_dict():
    return {name: spec.as_dict() for name, spec in STAGE_SPECS.items()}


def ensure_no_duplicates(name, values):
    if len(values) != len(set(values)):
        seen = set()
        dupes = []
        for value in values:
            if value in seen:
                dupes.append(value)
            seen.add(value)
        raise ValueError("{} contains duplicates: {}".format(name, dupes[:10]))


def source_image_path(cache_or_source_class_root, source_rel):
    return cache_or_source_class_root / source_rel


def source_feature_path(cache_class_root, scale, source_rel):
    rel = Path(source_rel)
    return cache_class_root / scale / rel.with_suffix(".npy")


def role_stage_rel(role, source_rel):
    if role not in ROLE_TO_STAGE_DIR:
        raise KeyError(role)
    source = Path(source_rel)
    stem = "__".join(source.with_suffix("").parts)
    safe_name = "{}__{}{}".format(role, stem, source.suffix.lower())
    return str(Path(ROLE_TO_STAGE_DIR[role]) / safe_name).replace(os.sep, "/")


def stage_feature_rel(scale, stage_image_rel):
    rel = Path(stage_image_rel).with_suffix(".npy")
    return str(Path(scale) / rel).replace(os.sep, "/")


def ensure_symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            target = os.readlink(str(dst))
            resolved = (dst.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
            if resolved == src.resolve():
                return "exists"
        raise FileExistsError("Refusing to overwrite existing path: {}".format(dst))
    os.symlink(str(src), str(dst), target_is_directory=src.is_dir())
    return "created"


def ensure_empty_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def count_existing_supported(folder):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if is_supported_path(p) and (p.is_file() or p.is_symlink()))


def load_stage_split(split_root, stage):
    root = stage_dir(split_root, stage)
    data = {role: read_lines(root / filename) for role, filename in ROLE_TO_FILE.items()}
    data["test_normal"] = read_lines(split_root / "test_normal.txt")
    data["test_anomaly"] = read_lines(split_root / "test_anomaly.txt")
    return data


def ahl_eval_order(class_root, know_class="train_defect"):
    test_root = class_root / "test"
    order = []
    good_root = test_root / "good"
    if good_root.exists():
        for name in os.listdir(str(good_root)):
            path = good_root / name
            if is_supported_path(path) and (path.is_file() or path.is_symlink()):
                order.append(str(Path("test/good") / name).replace(os.sep, "/"))
    if test_root.exists():
        for class_name in os.listdir(str(test_root)):
            if class_name in {"good", know_class}:
                continue
            class_dir = test_root / class_name
            if not class_dir.is_dir():
                continue
            for name in os.listdir(str(class_dir)):
                path = class_dir / name
                if is_supported_path(path) and (path.is_file() or path.is_symlink()):
                    order.append(str(Path("test") / class_name / name).replace(os.sep, "/"))
    return order


def read_result_rows(result_file):
    rows = []
    with result_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                rows.append((int(float(parts[0])), float(parts[1])))
            except ValueError:
                continue
    return rows


def stage_output_dir(output_root, stage):
    return output_root / "stages" / stage


def metrics_dir(output_root, stage):
    return stage_output_dir(output_root, stage) / "metrics"


def config_dir(output_root):
    return output_root / "config"