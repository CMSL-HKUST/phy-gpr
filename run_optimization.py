from __future__ import annotations

import argparse
from pathlib import Path

from src.optimization import SelectedOptimizationConfig, run_selected_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected-repeat inverse optimization.")
    parser.add_argument("--selected-repeat", type=int, default=5, help="CV repeat to retrain and inspect.")
    parser.add_argument("--test-groups", type=int, default=6, help="Held-out groups in the selected repeat.")
    parser.add_argument("--restarts", type=int, default=5, help="Optimizer starts for upstream GP fitting.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selected_model"), help="Directory for selected-model artifacts.")
    args = parser.parse_args()

    result = run_selected_optimization(
        SelectedOptimizationConfig(
            selected_repeat=args.selected_repeat,
            test_groups=args.test_groups,
            restarts=args.restarts,
            output_dir=args.output_dir,
        )
    )
    print(f"\nDone. Optimization outputs written to {result['output_dir']}")


if __name__ == "__main__":
    main()
