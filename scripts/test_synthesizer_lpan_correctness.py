"""Correctness test: Synthesizer-LPAN attention vs plaintext oracle.

Validates `attn_synthesizer`: O = A·V where A is a learned plaintext
attention pattern (per head). Two heads packed in one ct.

PASS criterion: max abs error < 1e-3 between numpy oracle and decrypted FHE.
"""
from __future__ import annotations
from pathlib import Path
import sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.attention import (
    encode_synthesizer_diagonals,
    attn_synthesizer,
    encode_synthesizer_bsgs,
    attn_synthesizer_bsgs,
)
from fhe_thesis.encryption.colmajor import prepare_colmajor_keys

T0 = time.time()
def log(m): print(f"[t+{time.time()-T0:5.1f}s] {m}", flush=True)


def pack_two_heads(be, X_h0, X_h1, *, L, head_dim):
    n_slots = be.capabilities.n_slots
    H = L * head_dim
    buf = [0.0] * n_slots
    for j in range(head_dim):
        base = j * L
        for i in range(L):
            buf[base + i] = float(X_h0[i, j])
            buf[H + base + i] = float(X_h1[i, j])
    return be.encrypt(buf)


def main():
    L = 8
    head_dim = 4
    num_heads_per_ct = 2
    H = L * head_dim
    rng = np.random.default_rng(7)

    log("Init HEonGPU N=2^15 chain=10...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 15,
        q_prime_bits=(60,) + (50,) * 10,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready  n_slots={be._num_slots}")
    n = prepare_colmajor_keys(be, L=L, max_dim=head_dim)
    log(f"  +{n} keys")

    V_h0 = rng.standard_normal((L, head_dim)) * 0.1
    V_h1 = rng.standard_normal((L, head_dim)) * 0.1

    # Realistic Synthesizer pattern: random softmax-row matrices per head.
    A_logits = rng.standard_normal((num_heads_per_ct, L, L))
    A = np.exp(A_logits) / np.exp(A_logits).sum(axis=-1, keepdims=True)
    log(f"  A shape={A.shape}, row-sum sample = {A[0,0].sum():.3f}")

    V_ct = pack_two_heads(be, V_h0, V_h1, L=L, head_dim=head_dim)

    # Pre-encode diagonals (one-time per layer).
    log("Encode A diagonals...")
    t = time.time()
    diag_pts = encode_synthesizer_diagonals(
        be, A, L=L, head_dim=head_dim, num_heads_per_ct=num_heads_per_ct,
    )
    log(f"  done in {time.time()-t:.3f}s ({len(diag_pts)} pts)")

    log("attn_synthesizer...")
    t = time.time()
    O_ct = attn_synthesizer(
        be, V_ct, diag_pts, L=L, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct,
    )
    log(f"  done in {time.time()-t:.3f}s")

    # Oracle
    O_h0_ref = A[0] @ V_h0
    O_h1_ref = A[1] @ V_h1

    slots = be.decrypt(O_ct)
    max_err = 0.0
    for h_idx, ref in enumerate([O_h0_ref, O_h1_ref]):
        base_h = h_idx * H
        for j in range(head_dim):
            base_j = base_h + j * L
            for i in range(L):
                err = abs(slots[base_j + i] - ref[i, j])
                max_err = max(max_err, err)
    log(f"  O max abs err = {max_err:.2e}")
    assert max_err < 1e-3, f"Synthesizer attention FAILED (err {max_err:.2e})"
    log("PASSED naive variant")

    # ---- BSGS variant ----
    log("BSGS: encode + run...")
    # Need extra rotation keys: ±1..±bs-1 and ±bs..±gs·bs
    # We'll just register a generous range.
    extra = list(range(1, L)) + [-x for x in range(1, L)]
    if hasattr(be, "register_rotation_keys"):
        be.register_rotation_keys(extra)
    bsgs_pts = encode_synthesizer_bsgs(
        be, A, L=L, head_dim=head_dim, num_heads_per_ct=num_heads_per_ct,
    )
    log(f"  bs={bsgs_pts['bs']} gs={bsgs_pts['gs']}")
    O_ct_bsgs = attn_synthesizer_bsgs(
        be, V_ct, bsgs_pts, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct,
    )
    slots_b = be.decrypt(O_ct_bsgs)
    max_err_b = 0.0
    for h_idx, ref in enumerate([O_h0_ref, O_h1_ref]):
        base_h = h_idx * H
        for j in range(head_dim):
            base_j = base_h + j * L
            for i in range(L):
                err = abs(slots_b[base_j + i] - ref[i, j])
                max_err_b = max(max_err_b, err)
    log(f"  BSGS O max abs err = {max_err_b:.2e}")
    assert max_err_b < 1e-3, f"BSGS variant FAILED (err {max_err_b:.2e})"
    log("PASSED BSGS variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
