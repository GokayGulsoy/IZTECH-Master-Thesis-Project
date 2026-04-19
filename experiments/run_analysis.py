#!/usr/bin/env python3
"""Unified entrypoint for thesis analysis scripts (Contributions 1, 2, 5).

Each analysis script is a standalone module under ``experiments.analysis``
that produces a self-contained set of figures + JSON in its own
``RESULTS_DIR`` subfolder. They are pre-conditions for
``experiments/generate_figures.py``.

Usage
-----
    # Run a single analysis
    python experiments/run_analysis.py poly        # → results/poly_approx/
    python experiments/run_analysis.py profile     # → results/activation_profiles/
    python experiments/run_analysis.py error       # → results/error_propagation/
    python experiments/run_analysis.py bsgs        # → results/bsgs_eval/

    # Run all of them sequentially
    python experiments/run_analysis.py all
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ANALYSES = {
    "poly": "experiments.analysis.poly_approximation",
    "profile": "experiments.analysis.activation_profiling",
    "error": "experiments.analysis.error_propagation",
    "bsgs": "experiments.analysis.bsgs_eval_strategies",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", choices=list(ANALYSES) + ["all"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    targets = list(ANALYSES) if args.which == "all" else [args.which]
    for key in targets:
        mod_name = ANALYSES[key]
        print(f"\n{'=' * 70}\n→ {mod_name}\n{'=' * 70}")
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "main"):
            raise RuntimeError(f"{mod_name} has no main() function")
        mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
