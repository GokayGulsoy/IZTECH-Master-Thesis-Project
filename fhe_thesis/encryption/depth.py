"""Symbolic multiplicative-depth tracker.

Used to audit the depth budget of the LPAN-FHE protocol *without*
running it. Each operation reports how many CKKS levels it consumes;
we accumulate per-tensor depths and warn if any tensor exceeds the
backend's `initial_levels`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── per-op depth costs (FHE-pure broadcast via plaintext matmul) ─────
# Each cost reflects the number of CKKS multiplicative levels consumed
# by the *deepest* output of the op. Standardisation maps `x → scale·x +
# shift` are folded into polynomial coefficients (see `_absorb_affine`)
# and consume zero levels.
DEPTH_COST: Dict[str, int] = {
    "linear": 1,            # plaintext-weight matmul + bias add
    "polyval_deg6": 3,      # GELU degree-6: ⌈log2(6)⌉+1 = 3
    "polyval_deg8": 4,      # ⌈log2(8)⌉+1 = 4
    "polyval_deg12": 4,     # softmax degree-12 (BSGS): ⌈log2(12)⌉+1 ≈ 4
    "polyval_deg4": 3,      # LN inv-sqrt degree-4: ⌈log2(4)⌉+1 = 3
    "ct_ct_mul": 1,         # ciphertext × ciphertext
    # ln_poly with mean centring (mirrors PolynomialLayerNorm exactly):
    #   mean_bc = sum(ct)/h            (1)  ← matmul-broadcast costs 1 level
    #   centred = ct − mean_bc         (0)  ← centred at level 1
    #   centred² = ct·ct               (1)  → 2
    #   var     = sum/h                (1)  → 3
    #   inv_σ   = polyval_deg4         (3)  → 6   (absorbed affine, level 6)
    #   y       = centred · inv_σ      (1)  → 7   (deeper operand wins)
    #   y       = γ · y                (1)  → 8
    #   y      += β                    (0)  → 8
    "ln_poly": 8,
    # softmax_poly: just polyval_deg12 with affine absorbed = 4
    "softmax_poly": 4,
    "residual_add": 0,
    "qk_scores": 2,         # ⟨Q[i], K[j]⟩ via dot (1) + matmul-broadcast (1)
    "attn_apply": 3,        # mul_plain mask (1) + matmul broadcast (1) + ct·ct (1)
    "head_concat": 1,       # mul_plain by per-head mask
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
    contribute one level, not three. The actual code path (post-LN BERT)
    is::

        attention_block:  Q-linear (∥K,V) → QK scores → softmax-poly →
                          attn·V → head_concat → O-linear → residual → LN-poly
        ffn_block:        W₁ → GELU-poly → W₂ → residual → LN-poly
    """
    attn = (
        DEPTH_COST["linear"]         # Q (K, V parallel)
        + DEPTH_COST["qk_scores"]
        + DEPTH_COST["softmax_poly"]
        + DEPTH_COST["attn_apply"]
        + DEPTH_COST["head_concat"]
        + DEPTH_COST["linear"]       # O
        + DEPTH_COST["ln_poly"]
    )
    ffn = (
        DEPTH_COST["linear"]         # W₁
        + DEPTH_COST["polyval_deg6"]  # GELU (LPAN uses degree 6)
        + DEPTH_COST["linear"]       # W₂
        + DEPTH_COST["ln_poly"]
    )
    return attn + ffn
