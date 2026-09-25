from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import PipelineConfig, run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AlSi10Mg 11-model GP workflow.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeated group-holdout count.")
    parser.add_argument("--test-groups", type=int, default=6, help="Held-out groups per repeat.")
    parser.add_argument("--restarts", type=int, default=5, help="Optimizer starts for tuned models.")
    parser.add_argument("--diag-alpha", action="store_true", help="Use yield std/sqrt(ng) as GP alpha.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts.")
    args = parser.parse_args()

    result = run_full_pipeline(
        PipelineConfig(
            repeats=args.repeats,
            test_groups=args.test_groups,
            restarts=args.restarts,
            diag_alpha=args.diag_alpha,
            output_dir=args.output_dir,
        )
    )
    print(f"\nDone. Outputs written to {result['output_dir']}")


if __name__ == "__main__":
    main()
