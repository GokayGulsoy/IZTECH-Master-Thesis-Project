"""Optimization utilities for HyPER-LPAN composition planning.

Two selectors are exposed:

* :func:`select_composition` / :func:`compose_for_task` — cheap
  entropy-threshold heuristic. Task-agnostic, single forward pass.
* :func:`select_composition_mckp` / :func:`compose_for_task_mckp` —
  Multiple-Choice Knapsack DP over measured per-layer attention
  substitution errors. Provably optimal for its (additive) surrogate
  objective; runs in milliseconds.
"""

from .composition_selector import (
    LAYER_COST,
    CompositionPlan,
    compose_for_task,
    compose_for_task_mckp,
    measure_layer_drifts,
    measure_layer_entropies,
    select_composition,
    select_composition_mckp,
)

__all__ = [
    "LAYER_COST",
    "CompositionPlan",
    "compose_for_task",
    "compose_for_task_mckp",
    "measure_layer_drifts",
    "measure_layer_entropies",
    "select_composition",
    "select_composition_mckp",
]
