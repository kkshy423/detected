#!/usr/bin/env python3
"""Run AHL main.py with cuDNN disabled for the auxiliary LSTM path.

This keeps AHL source untouched and only changes the runtime backend switch that
triggered CUDNN_STATUS_EXECUTION_FAILED in previous ADPretrain-cache runs. It
also supplies tiny fallbacks for optional plotting/einops imports when the PBS
runtime environment does not provide them.
"""

import runpy
import sys
import types
from pathlib import Path

import torch

AHL_MAIN = Path('/ghome/huangjd/code/detected/AHL/main.py')


def _install_optional_import_stubs():
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        matplotlib_stub = types.ModuleType('matplotlib')
        pyplot_stub = types.ModuleType('matplotlib.pyplot')
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules.setdefault('matplotlib', matplotlib_stub)
        sys.modules.setdefault('matplotlib.pyplot', pyplot_stub)

    try:
        import einops  # noqa: F401
    except ImportError:
        einops_stub = types.ModuleType('einops')

        def rearrange(x, pattern, **kwargs):
            normalized = ' '.join(pattern.split())
            if normalized == 'e k n-> k n e':
                return x.permute(1, 2, 0)
            raise NotImplementedError('Fallback einops.rearrange only supports: e k n-> k n e')

        einops_stub.rearrange = rearrange
        sys.modules.setdefault('einops', einops_stub)


_install_optional_import_stubs()


def _install_runtime_compatibility_patches():
    """Keep AHL source untouched while handling bridge-only edge cases."""
    import argparse
    import random

    original_add_argument = argparse.ArgumentParser.add_argument

    def parse_bool(value):
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

    def add_argument_bool_aware(self, *args, **kwargs):
        if kwargs.get("type") is bool:
            kwargs = dict(kwargs)
            kwargs["type"] = parse_bool
        return original_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = add_argument_bool_aware

    original_sample = random.sample

    def sample_allow_replacement_when_too_small(population, k, *args, **kwargs):
        if k <= len(population) or len(population) == 0:
            return original_sample(population, k, *args, **kwargs)
        pool = tuple(population)
        values = list(original_sample(population, len(population), *args, **kwargs))
        values.extend(random.choice(pool) for _ in range(k - len(values)))
        return values

    random.sample = sample_allow_replacement_when_too_small


_install_runtime_compatibility_patches()
ahl_root = str(AHL_MAIN.parent)
if ahl_root not in sys.path:
    sys.path.insert(0, ahl_root)
torch.backends.cudnn.enabled = False
runpy.run_path(str(AHL_MAIN), run_name='__main__')
