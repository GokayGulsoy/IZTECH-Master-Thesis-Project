"""Probe: amortize W encoding across N inputs in ONE linear.

Reorganizes the linear loop so encoded W plaintexts are SHARED across
multiple inputs:
  for giant g:
      encode_pts(g)               # one-time per giant
      for input i in batch:
          baby_rots[i] = ...
          y[i] += accumulate_giant(baby_rots[i], pts)
      free pts

Compares to baseline (encode-per-input) at one (cpc=256) sub-block.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor,
    prepare_colmajor_keys,
    build_colmajor_linear_plan_streaming,
    linear_colmajor_streaming,
)


def linear_batched_shared_encoding(be, x_cts, plan):
    """Batched linear with W encoding shared across all inputs in x_cts."""
    n_slots = be.capabilities.n_slots
    L = plan.L
    in_dim_padded = plan.in_dim_padded

    # 1. Replicate every input.
    x_reps = []
    for x_ct in x_cts:
        x_rep = x_ct
        cur = in_dim_padded * L
        while cur < n_slots:
            x_rep = be.add(x_rep, be.rotate(x_rep, -cur))
            cur <<= 1
        x_reps.append(x_rep)

    # 2. Galois keys.
    needed = set(plan.baby_shifts + plan.giant_shifts) - {0}
    if needed:
        be.register_rotation_keys(needed)

    # 3. Pre-rotate babies for each input.
    baby_rots_per_input = [
        be._ops.prepare_baby_rotations(x_rep, be._gk, plan.baby_shifts)
        for x_rep in x_reps
    ]

    # 4. Iterate giants — encode masks ONCE, apply across all inputs.
    Ys = [None] * len(x_cts)
    for g in range(plan.gs):
        baby_idx = plan.per_giant_baby_idx[g]
        if not baby_idx:
            continue

        # Encode this giant's masks (ONCE for all inputs).
        diag_arrays = plan.per_giant_diag_arrays[g]
        masks = []
        for arr in diag_arrays:
            pt = be._encode(arr.tolist())
            while be._ops.depth_of_plaintext(pt) < plan.ct_depth:
                be._ops.mod_drop_inplace_pt(pt)
            masks.append(pt)

        # Apply to each input.
        for i, baby_rots in enumerate(baby_rots_per_input):
            giant_ct = be._ops.accumulate_giant(
                baby_rots, be._gk, baby_idx, masks, plan.giant_shifts[g],
            )
            if Ys[i] is None:
                Ys[i] = giant_ct
            else:
                be._ops.add_inplace(Ys[i], giant_ct)
                del giant_ct

        # Evict masks before next giant.
        del masks

    return Ys


def main():
    L = 128
    cpc = 256
    N = 1 << 16
    chain = 28
    print(f"N={N} chain={chain} L={L} cpc={cpc}")

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * chain,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    prepare_colmajor_keys(be, L=L, max_dim=cpc)

    rng = np.random.default_rng(0)
    W = rng.standard_normal((cpc, cpc)) * 0.05
    plan = build_colmajor_linear_plan_streaming(
        be, W, L=L, in_dim=cpc, out_dim=cpc, bias=None, ct_depth=0,
    )

    # ---- Baseline: 1 input ----
    X = rng.standard_normal((L, cpc)) * 0.1
    x_ct = pack_colmajor(be, X, L=L, head_dim=cpc)
    t = time.time()
    Y_single = linear_colmajor_streaming(be, x_ct, plan)
    t_single = time.time() - t
    print(f"\nBaseline (1 input):                  {t_single*1000:7.1f}ms")

    # ---- Test batch sizes ----
    for batch_n in [4, 8, 16, 32]:
        x_cts = [pack_colmajor(be, rng.standard_normal((L, cpc)) * 0.1,
                              L=L, head_dim=cpc) for _ in range(batch_n)]
        # Warm-up
        try:
            t = time.time()
            Ys = linear_batched_shared_encoding(be, x_cts, plan)
            dt = time.time() - t
            per_input = dt / batch_n
            print(f"Batched N={batch_n:3d}  total={dt:7.2f}s  per-input={per_input*1000:7.1f}ms  speedup vs 1x: {t_single/per_input:.1f}×")
            del Ys, x_cts
            import gc; gc.collect()
        except Exception as e:
            print(f"Batched N={batch_n}: FAIL  {type(e).__name__}: {e}")
            break

    print("\nIf speedup grows with N, encoding-amortization works.")
    print("If wall time grows ~linearly with N, batching saturated the GPU.")


if __name__ == "__main__":
    main()
