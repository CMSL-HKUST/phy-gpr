from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data_prep import load_data
from .model_runner import run as run_models
from .utils import DEFAULT_OUTPUT_DIR, RANDOM_SEED, ensure_output_dirs, set_seed


@dataclass(frozen=True)
class PipelineConfig:
    repeats: int = 5
    test_groups: int = 6
    ng: int = 3
    restarts: int = 5
    diag_alpha: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR


def run_full_pipeline(config: PipelineConfig | None = None) -> dict:
    cfg = config or PipelineConfig()
    ensure_output_dirs(cfg.output_dir)
    set_seed(RANDOM_SEED)

    results = run_models(
        load_data(),
        ng=cfg.ng,
        seed=RANDOM_SEED,
        repeats=cfg.repeats,
        test_groups=cfg.test_groups,
        diag_alpha=cfg.diag_alpha,
        out=cfg.output_dir,
        restarts=cfg.restarts,
    )
    return {**results, "config": cfg}
