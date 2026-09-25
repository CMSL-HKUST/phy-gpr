from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 41


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def prepare_runtime(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    cache_root = Path("/tmp/codex-cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_output_dirs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return prepare_runtime(output_dir)


def save_pickle(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
