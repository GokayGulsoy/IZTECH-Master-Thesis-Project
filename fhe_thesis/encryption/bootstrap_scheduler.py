"""Ext 1 — Region-Adaptive Bootstrap Scheduling.

Standard practice for deep CKKS circuits is to insert a bootstrap call
at fixed intervals (e.g. every 2 layers). This wastes budget in regions
where layer depth is light: a single LinearMixing layer consumes ~10
levels, while a single LPAN layer consumes ~26. A region-adaptive
schedule places refresh points at the *latest* boundary that still fits
in the per-bootstrap budget, minimising the total number of bootstraps.

This module provides:

* per-layer depth lookup (from ``depth.py``);
* a greedy scheduler ``schedule_bootstraps(layer_kinds, budget)`` that
  returns the layer indices at which to bootstrap *before* execution;
* an integration helper ``maybe_bootstrap(backend, x, plan, layer_idx)``
  that the protocol's per-layer dispatch can call.

FHE-pure: the schedule depends only on the public composition plan and
the public CKKS parameters, never on ciphertext content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .depth import (
    linear_mixing_layer_depth,
    quad_layer_depth,
    transformer_layer_depth,
)


# Per-layer-kind depth in CKKS levels (matches depth.py constants).
LAYER_DEPTH = {
    "LM": linear_mixing_layer_depth(),
    "Q":  quad_layer_depth(),
    "L":  transformer_layer_depth(),
}

# Approximate cost of one bootstrap, in equivalent layer wall-clock units.
# ~1 LPAN layer of compute (calibrated from FHE benchmarks). Used for
# reporting expected wall-clock savings.
BOOTSTRAP_LAYER_COST = 1.0  # in LPAN-layer-equivalents


@dataclass
class BootstrapWindow:
    start_layer: int      # first layer index in the window (inclusive)
    end_layer: int        # last layer index in the window (inclusive)
    accumulated_depth: int


@dataclass
class BootstrapPlan:
    """A schedule of explicit bootstrap insertion points.

    ``insertion_indices`` is the list of layer indices ``i`` at which
    a bootstrap is inserted *before* executing layer ``i``. Index 0 is
    never an insertion (we start fresh). After the last layer the
    output is decrypted by the protocol.
    """
    insertion_indices: List[int]
    windows: List[BootstrapWindow]
    budget_per_window: int
    total_depth: int

    @property
    def num_bootstraps(self) -> int:
        return len(self.insertion_indices)

    def to_dict(self) -> dict:
        return {
            "insertion_indices": self.insertion_indices,
            "num_bootstraps": self.num_bootstraps,
            "budget_per_window": self.budget_per_window,
            "total_depth": self.total_depth,
            "windows": [
                {"start": w.start_layer, "end": w.end_layer,
                 "depth": w.accumulated_depth}
                for w in self.windows
            ],
        }


def _kind_for_layer(layer_idx: int,
                    linear_mixing_layers: Sequence[int],
                    quad_attention_layers: Sequence[int]) -> str:
    if layer_idx in linear_mixing_layers:
        return "LM"
    if layer_idx in quad_attention_layers:
        return "Q"
    return "L"


def composition_to_kinds(num_layers: int,
                         linear_mixing_layers: Sequence[int],
                         quad_attention_layers: Sequence[int]) -> List[str]:
    """Return a length-num_layers list of kind strings (LM | Q | L)."""
    return [_kind_for_layer(i, linear_mixing_layers, quad_attention_layers)
            for i in range(num_layers)]


def schedule_bootstraps(layer_kinds: Sequence[str],
                        budget_per_window: int,
                        ) -> BootstrapPlan:
    """Greedy region-adaptive bootstrap scheduler.

    Iterate layers in order; accumulate per-layer depth into a window.
    When adding the next layer would push the running total over
    ``budget_per_window``, close the window and insert a bootstrap
    before the next layer.

    Parameters
    ----------
    layer_kinds : sequence of str ('LM' | 'Q' | 'L')
        The composition plan, length = num_layers.
    budget_per_window : int
        Maximum accumulated depth tolerated between two bootstraps
        (i.e. the post-bootstrap level budget exposed by the backend).
    """
    insertions: List[int] = []
    windows: List[BootstrapWindow] = []
    cur_start = 0
    cur_depth = 0
    total_depth = 0
    for i, kind in enumerate(layer_kinds):
        d = LAYER_DEPTH[kind]
        total_depth += d
        if d > budget_per_window:
            raise ValueError(
                f"Layer {i} kind={kind} has depth {d} > "
                f"budget {budget_per_window}; reduce per-layer depth "
                "(e.g. via polynomial degree pruning) or raise the "
                "post-bootstrap budget."
            )
        if cur_depth + d > budget_per_window:
            # Close current window, insert bootstrap before layer i.
            windows.append(BootstrapWindow(
                start_layer=cur_start,
                end_layer=i - 1,
                accumulated_depth=cur_depth,
            ))
            insertions.append(i)
            cur_start = i
            cur_depth = d
        else:
            cur_depth += d
    windows.append(BootstrapWindow(
        start_layer=cur_start,
        end_layer=len(layer_kinds) - 1,
        accumulated_depth=cur_depth,
    ))
    return BootstrapPlan(
        insertion_indices=insertions,
        windows=windows,
        budget_per_window=budget_per_window,
        total_depth=total_depth,
    )


def schedule_uniform(num_layers: int,
                     period: int,
                     layer_kinds: Sequence[str],
                     budget_per_window: int,
                     ) -> BootstrapPlan:
    """Baseline: bootstrap every ``period`` layers (period >= 1).

    Useful for comparing against region-adaptive scheduling.
    """
    insertions = list(range(period, num_layers, period))
    windows: List[BootstrapWindow] = []
    cur_start = 0
    cur_depth = 0
    total_depth = 0
    next_ins = set(insertions)
    for i, kind in enumerate(layer_kinds):
        d = LAYER_DEPTH[kind]
        total_depth += d
        if i in next_ins:
            windows.append(BootstrapWindow(
                start_layer=cur_start,
                end_layer=i - 1,
                accumulated_depth=cur_depth,
            ))
            cur_start = i
            cur_depth = 0
        cur_depth += d
    windows.append(BootstrapWindow(
        start_layer=cur_start,
        end_layer=num_layers - 1,
        accumulated_depth=cur_depth,
    ))
    return BootstrapPlan(
        insertion_indices=insertions,
        windows=windows,
        budget_per_window=budget_per_window,
        total_depth=total_depth,
    )


def maybe_bootstrap(backend, x, plan: BootstrapPlan, layer_idx: int):
    """Apply explicit bootstrap to every ciphertext in ``x`` if scheduled.

    ``x`` is expected to be a packed-tensor object exposing a
    ``cts`` attribute (list of OpenFHE ciphertexts). Modifies in place
    and returns ``x``.
    """
    if layer_idx not in plan.insertion_indices:
        return x
    if not getattr(backend, "supports_bootstrapping", False):
        # Plaintext / no-bootstrap backend — no-op.
        return x
    x.cts = [backend.bootstrap(c) for c in x.cts]
    return x


def compare_plans(layer_kinds: Sequence[str],
                  budget_per_window: int,
                  uniform_period: int = 2,
                  ) -> Dict:
    """Return a side-by-side comparison of region-adaptive vs uniform."""
    adaptive = schedule_bootstraps(layer_kinds, budget_per_window)
    uniform = schedule_uniform(len(layer_kinds), uniform_period,
                               layer_kinds, budget_per_window)
    saved = uniform.num_bootstraps - adaptive.num_bootstraps
    return {
        "layer_kinds": list(layer_kinds),
        "budget_per_window": budget_per_window,
        "adaptive": adaptive.to_dict(),
        "uniform": uniform.to_dict(),
        "bootstraps_saved": saved,
        "estimated_wallclock_saved_layer_eq":
            saved * BOOTSTRAP_LAYER_COST,
    }
