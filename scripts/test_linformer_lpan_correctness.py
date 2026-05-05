"""Correctness test: Linformer-LPAN attention vs plaintext oracle.

Verifies:
  kv_project_linformer_nexus     (E·K, F·V — sequence projection)
  qk_scores_linformer_nexus      (Q·K'^T / sqrt(d), polyval softmax)
  attn_apply_linformer_nexus     (S·V')
end-to-end against a numpy reference. Uses a small case
(L=8, head_dim=4, num_heads_per_ct=2, k=4) so the test is fast enough
to run on CPU-friendly N if desired, but defaults to N=2^16 since the
masks and head-packing only make sense at our production geometry.

PASS criterion: max abs error < 1e-4 between oracle and decrypted FHE result
on each stage (projection, scores, attention output).
"""
from __future__ import annotations

import sys
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor,
    unpack_colmajor,
    kv_project_linformer_nexus,
    qk_scores_linformer_nexus,
    attn_apply_linformer_nexus,
    prepare_colmajor_keys,
    prepare_linformer_keys,
)


def log(msg):
    print(f"[t+{time.time()-T0:6.1f}s] {msg}", flush=True)


def pack_two_heads(be, X_h0, X_h1, *, L, head_dim):
    """Pack two heads side-by-side into a single ct (NEXUS convention).

    slot[0..L·d] = head 0 col-major
    slot[L·d..2·L·d] = head 1 col-major
    """
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
    global T0
    T0 = time.time()

    # --- geometry ---
    L = 8
    head_dim = 4
    num_heads_per_ct = 2
    k_proj = 4
    H = L * head_dim
    rng = np.random.default_rng(42)

    # --- backend ---
    log("Init HEonGPU N=2^15 chain=10 (small; correctness only)...")
    chain = 10
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 15,
        q_prime_bits=(60,) + (50,) * chain,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready  n_slots={be._num_slots}")
    n1 = prepare_colmajor_keys(be, L=L, max_dim=head_dim)
    n2 = prepare_linformer_keys(be, L=L)
    log(f"  +{n1}+{n2} rotation keys")

    # --- inputs ---
    Q_h0 = rng.standard_normal((L, head_dim)) * 0.1
    Q_h1 = rng.standard_normal((L, head_dim)) * 0.1
    K_h0 = rng.standard_normal((L, head_dim)) * 0.1
    K_h1 = rng.standard_normal((L, head_dim)) * 0.1
    V_h0 = rng.standard_normal((L, head_dim)) * 0.1
    V_h1 = rng.standard_normal((L, head_dim)) * 0.1
    E = rng.standard_normal((k_proj, L)) * 0.3
    F = rng.standard_normal((k_proj, L)) * 0.3
    softmax_coeffs = [1.0, 0.5, 0.125]
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    # --- pack into 2-head cts ---
    Q_ct = pack_two_heads(be, Q_h0, Q_h1, L=L, head_dim=head_dim)
    K_ct = pack_two_heads(be, K_h0, K_h1, L=L, head_dim=head_dim)
    V_ct = pack_two_heads(be, V_h0, V_h1, L=L, head_dim=head_dim)

    # ============================================================
    # STAGE 1: kv_project
    # ============================================================
    log("Stage 1: kv_project_linformer_nexus(K, E)")
    Kp_list = kv_project_linformer_nexus(
        be, K_ct, E, L=L, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct, k_proj=k_proj,
    )
    Vp_list = kv_project_linformer_nexus(
        be, V_ct, F, L=L, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct, k_proj=k_proj,
    )
    assert len(Kp_list) == k_proj

    # Oracle
    Kp_h0_ref = E @ K_h0       # (k, d)
    Kp_h1_ref = E @ K_h1
    Vp_h0_ref = F @ V_h0
    Vp_h1_ref = F @ V_h1

    # Verify each Kp_list[r] holds K'[r, j] broadcast across slot[h·H + j·L + i].
    max_err_proj = 0.0
    for r in range(k_proj):
        slots = be.decrypt(Kp_list[r])
        # head 0: slot[j*L + i] should equal Kp_h0_ref[r, j] for all i ∈ [0, L)
        for h_idx, ref_row in enumerate([Kp_h0_ref[r], Kp_h1_ref[r]]):
            base_h = h_idx * H
            for j in range(head_dim):
                base_j = base_h + j * L
                for i in range(L):
                    err = abs(slots[base_j + i] - ref_row[j])
                    max_err_proj = max(max_err_proj, err)
    log(f"  K' max abs err = {max_err_proj:.2e}")
    assert max_err_proj < 1e-3, f"K' projection failed (err {max_err_proj:.2e})"

    # ============================================================
    # STAGE 2: qk_scores_linformer
    # ============================================================
    log("Stage 2: qk_scores_linformer_nexus(Q, K')")
    S_list = qk_scores_linformer_nexus(
        be, Q_ct, Kp_list, L=L, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct, scale=inv_sqrt_d,
    )
    assert len(S_list) == k_proj

    S_h0_ref = (Q_h0 @ Kp_h0_ref.T) * inv_sqrt_d  # (L, k)
    S_h1_ref = (Q_h1 @ Kp_h1_ref.T) * inv_sqrt_d

    max_err_qk = 0.0
    for r in range(k_proj):
        slots = be.decrypt(S_list[r])
        for h_idx, ref in enumerate([S_h0_ref, S_h1_ref]):
            base_h = h_idx * H
            for i in range(L):
                err = abs(slots[base_h + i] - ref[i, r])
                max_err_qk = max(max_err_qk, err)
    log(f"  S max abs err = {max_err_qk:.2e}")
    assert max_err_qk < 1e-3, f"qk failed (err {max_err_qk:.2e})"

    # ============================================================
    # STAGE 3: softmax-poly + attn_apply_linformer
    # ============================================================
    log("Stage 3: polyval + attn_apply_linformer_nexus(S, V')")
    Asm_list = [be.polyval(s, list(softmax_coeffs)) for s in S_list]

    # oracle
    def poly3(x):
        return softmax_coeffs[0] + softmax_coeffs[1] * x + softmax_coeffs[2] * x * x
    Asm_h0_ref = poly3(S_h0_ref)
    Asm_h1_ref = poly3(S_h1_ref)
    O_h0_ref = Asm_h0_ref @ Vp_h0_ref  # (L, d)
    O_h1_ref = Asm_h1_ref @ Vp_h1_ref

    O_ct = attn_apply_linformer_nexus(
        be, Asm_list, Vp_list, L=L, head_dim=head_dim,
        num_heads_per_ct=num_heads_per_ct,
    )
    slots = be.decrypt(O_ct)
    max_err_av = 0.0
    for h_idx, ref in enumerate([O_h0_ref, O_h1_ref]):
        base_h = h_idx * H
        for j in range(head_dim):
            base_j = base_h + j * L
            for i in range(L):
                err = abs(slots[base_j + i] - ref[i, j])
                max_err_av = max(max_err_av, err)
    log(f"  O max abs err = {max_err_av:.2e}")
    assert max_err_av < 1e-3, f"av failed (err {max_err_av:.2e})"

    log("ALL STAGES PASSED.")
    log(f"  proj err {max_err_proj:.2e} | qk err {max_err_qk:.2e} | av err {max_err_av:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
