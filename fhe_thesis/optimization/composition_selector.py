"""Ext 3 — Task-Adaptive HyPER-LPAN Composition Selector.

Given a plaintext checkpoint and a small dev loader, this module decides
which layers should run as LinearMixing (cheapest, position-only),
QuadAttention (medium, content-aware), or LPAN (full softmax-poly,
expensive). The selection is data-driven: layers whose attention is
already close to uniform (high entropy across the query positions) can
be safely replaced by a position-mixing primitive; layers with sharply
peaked attention need full content selection.

Selection rule (information-theoretic):
    Let H_l = mean over (sample, head, query) of entropy of attention
    distribution at layer l, normalized by log(seq_len) so h_l ∈ [0, 1].

    For a per-task latency budget B (in units of LinearMixing layer
    cost), choose an assignment a : layer → {LM, Q, L} that minimises::

        cost(a) + alpha * fidelity_penalty(a)

    where cost(a) = sum_l c_{a_l} with c_LM=1, c_Q=1.4, c_L=3.5
    (calibrated from FHE benchmarks), and the fidelity penalty rewards
    keeping high-entropy layers in LinearMixing and low-entropy layers
    in LPAN. Concretely we use a two-threshold rule with thresholds
    chosen to match the budget:

        a_l = LM   if h_l >= tau_high
        a_l = L    if h_l <= tau_low
        a_l = Q    otherwise

    The thresholds are selected by sweeping over the empirical entropy
    distribution; the smallest (tau_low, tau_high) pair whose induced
    cost <= B is returned.

This module is FHE-pure: it runs only on the plaintext model on the
client side at ahead-of-time deployment. The chosen assignment becomes
part of the public model description.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import math
import numpy as np


# Calibrated layer-cost constants (LinearMixing = 1.0 reference).
# Source: results/benchmarks/fhe_benchmark_*.json (Pod, 32 vCPU, max_seq_len=64).
LAYER_COST = {"LM": 1.0, "Q": 1.4, "L": 3.5}


@dataclass
class CompositionPlan:
    linear_mixing_layers: List[int]
    quad_attention_layers: List[int]
    lpan_layers: List[int]
    layer_entropies: List[float]
    tau_low: float
    tau_high: float
    estimated_cost: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml_fragment(self) -> str:
        return (
            f"linear_mixing_layers: {self.linear_mixing_layers}\n"
            f"quad_attention_layers: {self.quad_attention_layers}\n"
            f"lpan_layers: {self.lpan_layers}\n"
        )


def measure_layer_entropies(
    model,
    samples: Sequence[dict],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return per-layer normalised attention entropy over ``samples``.

    ``model`` must be a HuggingFace BERT-style sequence-classification
    model. Each sample dict needs ``input_ids`` and ``attention_mask``
    (1-D numpy/torch arrays of length seq_len).

    Returns a 1-D numpy array of shape (num_layers,) with values in
    [0, 1] (entropy normalised by log(seq_len)).
    """
    import torch

    model.eval()
    model.to(device)

    layer_sums: Dict[int, float] = {}
    layer_counts: Dict[int, int] = {}

    with torch.no_grad():
        for s in samples:
            ids = torch.as_tensor(s["input_ids"]).long().unsqueeze(0).to(device)
            mask = torch.as_tensor(s["attention_mask"]).long().unsqueeze(0).to(device)
            out = model.bert(ids, attention_mask=mask, output_attentions=True)
            valid_len = int(mask.sum().item())
            if valid_len < 2:
                continue
            log_norm = math.log(valid_len)
            # out.attentions: tuple of (1, heads, seq, seq)
            for li, A in enumerate(out.attentions):
                a = A[0, :, :valid_len, :valid_len]  # (heads, q, k)
                # avoid log(0)
                a = a.clamp_min(1e-12)
                ent = -(a * a.log()).sum(dim=-1)  # (heads, q)
                mean_ent = float(ent.mean().item()) / log_norm
                layer_sums[li] = layer_sums.get(li, 0.0) + mean_ent
                layer_counts[li] = layer_counts.get(li, 0) + 1

    n_layers = max(layer_counts) + 1
    out = np.zeros(n_layers, dtype=np.float64)
    for li in range(n_layers):
        out[li] = layer_sums[li] / max(layer_counts[li], 1)
    return out


def _assign(entropies: np.ndarray, tau_low: float, tau_high: float
            ) -> Tuple[List[int], List[int], List[int]]:
    lm, qa, lp = [], [], []
    for li, h in enumerate(entropies):
        if h >= tau_high:
            lm.append(li)
        elif h <= tau_low:
            lp.append(li)
        else:
            qa.append(li)
    return lm, qa, lp


def _cost(lm: List[int], qa: List[int], lp: List[int]) -> float:
    return (LAYER_COST["LM"] * len(lm)
            + LAYER_COST["Q"] * len(qa)
            + LAYER_COST["L"] * len(lp))


def select_composition(
    entropies: np.ndarray,
    *,
    budget: float | None = None,
    tau_low: float | None = None,
    tau_high: float | None = None,
    min_lpan: int = 0,
) -> CompositionPlan:
    """Select a composition plan from layer entropies.

    Two modes:
      * Manual: pass ``tau_low`` and ``tau_high`` directly.
      * Budget: pass ``budget`` (in LinearMixing-cost units, e.g.
        ``budget=0.5 * num_layers * LAYER_COST['L']`` for 50% of full-LPAN
        cost). The smallest threshold pair giving cost <= budget that
        keeps at least ``min_lpan`` layers in LPAN is returned.
    """
    n = len(entropies)
    if tau_low is not None and tau_high is not None:
        lm, qa, lp = _assign(entropies, tau_low, tau_high)
        return CompositionPlan(
            linear_mixing_layers=lm,
            quad_attention_layers=qa,
            lpan_layers=lp,
            layer_entropies=entropies.tolist(),
            tau_low=float(tau_low),
            tau_high=float(tau_high),
            estimated_cost=_cost(lm, qa, lp),
        )

    if budget is None:
        # Default budget: 50% of cost(all-LPAN)
        budget = 0.5 * n * LAYER_COST["L"]

    # Sweep over candidate threshold pairs from the empirical entropy set.
    # We bias toward maximising LinearMixing usage at the high-entropy end,
    # and using LPAN only where it is most needed (lowest entropies).
    sorted_h = np.sort(entropies)
    candidates = [-1.0] + sorted_h.tolist() + [2.0]
    best: CompositionPlan | None = None
    for ti in range(len(candidates)):
        for tj in range(ti, len(candidates)):
            tl, th = candidates[ti], candidates[tj]
            if tl > th:
                continue
            lm, qa, lp = _assign(entropies, tl, th)
            if len(lp) < min_lpan:
                continue
            c = _cost(lm, qa, lp)
            if c > budget:
                continue
            # Prefer plans with more LinearMixing and fewer LPAN layers,
            # then lower cost.
            score = (-len(lm), len(lp), c)
            if best is None or score < (-len(best.linear_mixing_layers),
                                        len(best.lpan_layers),
                                        best.estimated_cost):
                best = CompositionPlan(
                    linear_mixing_layers=lm,
                    quad_attention_layers=qa,
                    lpan_layers=lp,
                    layer_entropies=entropies.tolist(),
                    tau_low=float(tl),
                    tau_high=float(th),
                    estimated_cost=c,
                )

    if best is None:
        # Budget infeasible — fall back to all-Quad
        return CompositionPlan(
            linear_mixing_layers=[],
            quad_attention_layers=list(range(n)),
            lpan_layers=[],
            layer_entropies=entropies.tolist(),
            tau_low=0.0,
            tau_high=1.0,
            estimated_cost=_cost([], list(range(n)), []),
        )
    return best


def compose_for_task(
    model,
    samples: Sequence[dict],
    *,
    budget: float | None = None,
    tau_low: float | None = None,
    tau_high: float | None = None,
    min_lpan: int = 0,
    device: str = "cpu",
) -> CompositionPlan:
    """End-to-end: measure entropies on ``samples`` and pick a plan."""
    h = measure_layer_entropies(model, samples, device=device)
    return select_composition(
        h,
        budget=budget,
        tau_low=tau_low,
        tau_high=tau_high,
        min_lpan=min_lpan,
    )
