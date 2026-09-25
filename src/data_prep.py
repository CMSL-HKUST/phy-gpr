"""Data loading for the AlSi10Mg 11-model workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import PROJECT_ROOT

EXCEL_PATH = PROJECT_ROOT / "data" / "AlSi10Mg PSP feature table.xlsx"

COLUMNS = {
    "P": 2,
    "v": 3,
    "d_mean": 14,
    "d_std": 16,
    "phi_mean": 32,
    "phi_std": 33,
    "yield_mean": 40,
    "yield_std": 41,
}


def load_data(path: str | Path = EXCEL_PATH, n_groups: int = 32) -> pd.DataFrame:
    raw = pd.read_excel(Path(path), header=None)
    data = raw.iloc[4:].copy()
    frame = pd.DataFrame({name: pd.to_numeric(data.iloc[:, col], errors="coerce") for name, col in COLUMNS.items()})
    frame = frame.dropna().reset_index(drop=True).iloc[:n_groups].copy()
    frame.insert(0, "group_id", np.arange(len(frame), dtype=int))
    if len(frame) < n_groups:
        raise ValueError(f"Expected at least {n_groups} valid AlSi10Mg groups, found {len(frame)}")

    return frame


def column_map() -> dict[str, int]:
    return dict(COLUMNS)
