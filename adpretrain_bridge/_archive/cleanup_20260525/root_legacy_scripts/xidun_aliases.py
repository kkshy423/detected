"""Stable Xidun class aliases used by PBS-safe bridge experiments."""

XIDUN_CLASS_ALIASES = [
    ("models__球面斜拍", "models_qiumianxiepai"),
    ("qiusai__models__底面检测__端面检测", "qiusai_models_dimian_duanmian"),
    ("qiusai__models__球面俯拍", "qiusai_models_qiumianfupai"),
    ("yuanzhu__models__内孔中", "yuanzhu_models_neikongzhong"),
    ("yuanzhu__models__孔口", "yuanzhu_models_kongkou"),
    ("yuanzhu__models__端面", "yuanzhu_models_duanmian"),
]

XIDUN_THREECLS_CLASS_ALIASES = [
    ("models", "models"),
    ("qiusai", "qiusai"),
    ("yuanzhu", "yuanzhu"),
]

ALIAS_PROFILES = {
    "xidun6": XIDUN_CLASS_ALIASES,
    "xidun3": XIDUN_THREECLS_CLASS_ALIASES,
}


def suffixed_aliases(suffix: str, profile: str = "xidun6"):
    if profile not in ALIAS_PROFILES:
        raise ValueError(f"unknown alias profile: {profile}; available={sorted(ALIAS_PROFILES)}")
    return [(real, f"{alias}_{suffix}") for real, alias in ALIAS_PROFILES[profile]]
