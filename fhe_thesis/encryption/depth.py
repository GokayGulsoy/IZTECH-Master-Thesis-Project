"""Symbolic multiplicative-depth tracker.

Used to audit the depth budget of the LPAN-FHE protocol *without*
running it. Each operation reports how many CKKS levels it consumes;
we accumulate per-tensor depths and warn if any tensor exceeds the
backend's `initial_levels`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── per-op depth costs (degree-8 polys, Horner-evaluated) ──────────────
DEPTH_COST: Dict[str, int] = {
    "linear": 1,  # plaintext-weight matmul + bias add
    "polyval_deg8": 3,  # ⌈log2(8)⌉ levels
    "ct_ct_mul": 1,  # ciphertext × ciphertext
    "ln_poly": 4,  # Σx² (1) + invsqrt-poly deg-8 (3)
    "softmax_poly": 3,  # deg-8 polyval
    "residual_add": 0,
    "qk_scores": 2,  # ⟨Q[i], K[j]⟩ via dot (1) + mul_plain mask (1)
    "attn_apply": 2,  # mul_plain mask (1) + ct·ct broadcast (1); sum_slots is rotation-only
    "head_concat": 1,  # mul_plain by per-head mask
}


@dataclass
class DepthAudit:
    """Track per-operation depth consumption."""

    initial_levels: int
    log: List[tuple[str, int, int]] = field(default_factory=list)
    current: int = 0

    def consume(self, op: str) -> None:
        cost = DEPTH_COST[op]
        self.current += cost
        self.log.append((op, cost, self.remaining))

    @property
    def remaining(self) -> int:
        return self.initial_levels - self.current

    def report(self) -> str:
        lines = [
            f"Depth audit (budget = {self.initial_levels} levels):",
            f"{'op':<20} {'cost':>5} {'remaining':>10}",
            "-" * 38,
        ]
        for op, cost, rem in self.log:
            lines.append(f"{op:<20} {cost:>5} {rem:>10}")
        lines.append("-" * 38)
        lines.append(f"{'TOTAL':<20} {self.current:>5} {self.remaining:>10}")
        if self.remaining < 0:
            lines.append("⚠️  budget exceeded — bootstrapping required")
        return "\n".join(lines)


def transformer_layer_depth() -> int:
    """Return the *critical-path* depth of one LPAN BERT layer.

    Q, K, V linears run in parallel on independent ciphertexts so they
    contribute one level, not three. Same for the two LN-polys (one
    pre-attn, one pre-FFN), which are sequential. Sequence (per
    ``docs/ckks_protocol.md`` §5):

        LN-poly → Q-linear (∥ K, V) → Q·Kᵀ scores → softmax-poly →
        attn·V → O-linear → residual → LN-poly → W₁ → GELU-poly → W₂
    """
    return (
        DEPTH_COST["ln_poly"]
        + DEPTH_COST["linear"]  # Q (K, V parallel)
        + DEPTH_COST["qk_scores"]  # ⟨Q[i], K[j]⟩ + slot mask
        + DEPTH_COST["softmax_poly"]
        + DEPTH_COST["attn_apply"]  # mul_plain mask + ct·ct broadcast
        + DEPTH_COST["head_concat"]  # zero-pad mask + add
        + DEPTH_COST["linear"]  # O
        + DEPTH_COST["ln_poly"]
        + DEPTH_COST["linear"]  # W₁
        + DEPTH_COST["polyval_deg8"]  # GELU
        + DEPTH_COST["linear"]  # W₂
    )
