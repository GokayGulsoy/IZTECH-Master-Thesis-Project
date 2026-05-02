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
            backbone = getattr(model, getattr(model, "base_model_prefix", "bert"))
            out = backbone(ids, attention_mask=mask, output_attentions=True)
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


# ---------------------------------------------------------------------------
# MCKP-DP selector (provably optimal under additive surrogate)
# ---------------------------------------------------------------------------
#
# Per-layer attention substitution error
# --------------------------------------
# For each layer l we observe the soft-attention tensor A_l of shape
# (heads, q, k). The best content-blind LinearMixing approximator under
# squared-Frobenius loss is the dataset-mean pattern P_l = E[A_l]; the
# residual error is then
#       eps_l(LM) = E[ ||A_l - P_l||_F^2 ]
# which is exactly the *trace of the attention covariance* — the same
# quantity our entropy heuristic upper-bounds via a log-sum inequality.
# Measuring it directly is more faithful than the entropy proxy.
#
# QuadAttention is content-aware but lacks softmax sharpening; we model
# its residual as a fraction (1-gamma_Q) of the LM residual:
#       eps_l(Q)  = (1 - gamma_Q) * eps_l(LM),      gamma_Q in (0,1)
# with gamma_Q calibrated empirically (default 0.5 — Q removes about
# half of the LM-vs-true gap on standard GLUE attention maps).
#
# LPAN is the reference primitive in our menu, so eps_l(L) = 0.
#
# Composition selection then becomes the Multiple-Choice Knapsack
# Problem (MCKP):
#       min_{k}  sum_l eps_l(k_l)   s.t.   sum_l c_{k_l} <= B
# which admits an exact O(L * B') dynamic-programming solution where
# B' is the budget on a discretised cost scale.

def measure_layer_drifts(
    model,
    samples: Sequence[dict],
    *,
    gamma_q: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """Return per-layer substitution errors as an (L, 3) array.

    Columns correspond to [LM, Q, L] in that order. ``LM`` is the
    Frobenius residual of the dataset-mean approximator; ``Q`` is
    ``(1 - gamma_q) * LM``; ``L`` is identically zero.
    """
    import torch

    model.eval()
    model.to(device)

    # Accumulate sum and sum-of-squares of A_l per (head, q, k) cell on
    # a fixed seq_len. We trim each sample to its valid_len and pad with
    # zeros so the variance is computed only over the populated region.
    sum_A: Dict[int, torch.Tensor] = {}
    sumsq_A: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    seq_lens: Dict[int, int] = {}

    with torch.no_grad():
        for s in samples:
            ids = torch.as_tensor(s["input_ids"]).long().unsqueeze(0).to(device)
            mask = torch.as_tensor(s["attention_mask"]).long().unsqueeze(0).to(device)
            backbone = getattr(model, getattr(model, "base_model_prefix", "bert"))
            out = backbone(ids, attention_mask=mask, output_attentions=True)
            valid_len = int(mask.sum().item())
            if valid_len < 2:
                continue
            for li, A in enumerate(out.attentions):
                a = A[0, :, :valid_len, :valid_len].double()  # (heads, q, k)
                if li not in sum_A:
                    sum_A[li] = torch.zeros_like(a)
                    sumsq_A[li] = torch.zeros_like(a)
                    seq_lens[li] = valid_len
                if a.shape != sum_A[li].shape:
                    # mismatched seq_len — skip; selector expects fixed-length inputs
                    continue
                sum_A[li] += a
                sumsq_A[li] += a * a
                counts[li] = counts.get(li, 0) + 1

    n_layers = max(counts) + 1
    drifts = np.zeros((n_layers, 3), dtype=np.float64)
    for li in range(n_layers):
        n = max(counts.get(li, 1), 1)
        mean_A = sum_A[li] / n
        # E[||A - E[A]||_F^2] = E[||A||^2] - ||E[A]||^2  (per cell, summed)
        var_F = float((sumsq_A[li] / n - mean_A * mean_A).clamp_min(0).sum().item())
        drifts[li, 0] = var_F                       # LM
        drifts[li, 1] = max(1.0 - gamma_q, 0.0) * var_F  # Q
        drifts[li, 2] = 0.0                          # L
    return drifts


def select_composition_mckp(
    drifts: np.ndarray,
    *,
    budget: float,
    cost_scale: int = 10,
    min_lpan: int = 0,
) -> CompositionPlan:
    """Provably optimal MCKP-DP composition selector.

    Parameters
    ----------
    drifts : (L, 3) array
        Per-layer substitution error for kinds [LM, Q, L]. Output of
        :func:`measure_layer_drifts`.
    budget : float
        Cost budget in LinearMixing-cost units.
    cost_scale : int
        Multiplier used to discretise costs for the DP table. The
        default of 10 turns the calibrated cost vector
        ``[1.0, 1.4, 3.5]`` into integers ``[10, 14, 35]`` — already
        sufficient resolution for L=12. Increase to reduce rounding.
    min_lpan : int
        Lower bound on the number of LPAN layers in the returned plan.
        Enforced as a side constraint inside the DP via a second axis.

    Returns
    -------
    CompositionPlan
        The (provably) cost-feasible plan minimising
        ``sum_l drifts[l, k_l]`` under ``sum_l c_{k_l} <= budget``.
    """
    n = drifts.shape[0]
    kinds = ("LM", "Q", "L")
    cost_vec = np.array([LAYER_COST[k] for k in kinds], dtype=np.float64)
    cost_int = np.rint(cost_vec * cost_scale).astype(np.int64)  # [10,14,35]
    B_int = int(np.floor(budget * cost_scale))

    INF = float("inf")
    # dp[l][b][m] = (best total drift using layers 0..l-1, used cost <= b,
    #               with EXACTLY m of those layers being LPAN), and
    # prev[l][b][m] records the chosen primitive index. Using an exact
    # count (rather than a saturating min(., min_lpan)) keeps the
    # back-tracking unambiguous.
    M = n + 1  # exact LPAN counts 0..n
    dp = np.full((n + 1, B_int + 1, M), INF, dtype=np.float64)
    prev = np.full((n + 1, B_int + 1, M), -1, dtype=np.int8)
    dp[0, 0, 0] = 0.0

    for l in range(n):
        for b in range(B_int + 1):
            for m in range(M):
                if dp[l, b, m] == INF:
                    continue
                for k_idx, k_name in enumerate(kinds):
                    nb = b + int(cost_int[k_idx])
                    if nb > B_int:
                        continue
                    nm = m + (1 if k_name == "L" else 0)
                    cand = dp[l, b, m] + float(drifts[l, k_idx])
                    if cand < dp[l + 1, nb, nm]:
                        dp[l + 1, nb, nm] = cand
                        prev[l + 1, nb, nm] = k_idx

    # Find best terminal cell respecting the min_lpan side constraint.
    best_b, best_m = -1, -1
    best_val = INF
    for b in range(B_int + 1):
        for m in range(min_lpan, M):
            v = dp[n, b, m]
            if v < best_val:
                best_val = v
                best_b, best_m = b, m

    if best_b < 0:
        # Infeasible: fall back to all-Q (cheapest content-aware option).
        lm, qa, lp = [], list(range(n)), []
        return CompositionPlan(
            linear_mixing_layers=lm,
            quad_attention_layers=qa,
            lpan_layers=lp,
            layer_entropies=[0.0] * n,
            tau_low=float("nan"),
            tau_high=float("nan"),
            estimated_cost=_cost(lm, qa, lp),
        )

    # Reconstruct assignment by walking prev backwards.
    assignment: List[int] = [-1] * n
    b, m = best_b, best_m
    for l in range(n, 0, -1):
        k_idx = int(prev[l, b, m])
        assignment[l - 1] = k_idx
        b -= int(cost_int[k_idx])
        if kinds[k_idx] == "L":
            m -= 1

    lm = [l for l, k in enumerate(assignment) if kinds[k] == "LM"]
    qa = [l for l, k in enumerate(assignment) if kinds[k] == "Q"]
    lp = [l for l, k in enumerate(assignment) if kinds[k] == "L"]
    return CompositionPlan(
        linear_mixing_layers=lm,
        quad_attention_layers=qa,
        lpan_layers=lp,
        layer_entropies=[float(d) for d in drifts[:, 0]],  # store LM-drifts here for inspection
        tau_low=float("nan"),
        tau_high=float("nan"),
        estimated_cost=_cost(lm, qa, lp),
    )


def compose_for_task_mckp(
    model,
    samples: Sequence[dict],
    *,
    budget: float,
    gamma_q: float = 0.5,
    min_lpan: int = 0,
    cost_scale: int = 10,
    device: str = "cpu",
) -> CompositionPlan:
    """End-to-end MCKP-DP selector: measure drifts, run the DP."""
    drifts = measure_layer_drifts(
        model, samples, gamma_q=gamma_q, device=device,
    )
    return select_composition_mckp(
        drifts,
        budget=budget,
        cost_scale=cost_scale,
        min_lpan=min_lpan,
    )
