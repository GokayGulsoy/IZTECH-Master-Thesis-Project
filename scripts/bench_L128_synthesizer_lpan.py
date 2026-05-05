"""1-layer Synthesizer-LPAN bench (BATCH=4).

Replaces:
  - Wq, Wk linears        → ELIMINATED (no Q, no K needed)
  - qk_scores_nexus       → ELIMINATED (no Q·K^T)
  - softmax-poly on S     → ELIMINATED (already absorbed into A at train time)
  - attn_apply_nexus (av) → attn_synthesizer_nexus (pt·ct only, no ct·ct)

Keeps: Wv, Wo, W1, W2 batched linears, residuals, LN, GELU.

Reports per-input wall and projects 12-layer to validate the <100s path.
"""
import gc, time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor_multi,
    encode_synthesizer_bsgs,
    attn_synthesizer_bsgs_nexus,
    layernorm_colmajor_multi,
    linear_colmajor_multi_streaming_batched,
    add_multi,
    prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 128
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads
    inter = 4 * hidden
    import os
    BATCH = int(os.environ.get("BATCH", 8))
    NUM_HEAD_BUNDLES = 3            # 12 heads / 4-per-ct = 3 cts
    HEADS_PER_CT = 4

    log(f"Config L={L} hidden={hidden} heads={num_heads} BATCH={BATCH}")
    N = 1 << 16
    chain = int(os.environ.get("CHAIN", 20))   # Synthesizer eliminates softmax-poly depth
    log(f"Init backend N={N} chain={chain}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * chain,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")
    n_new = prepare_colmajor_keys(be, L=L, max_dim=256)
    log(f"  +{n_new} keys (colmajor)")
    # BSGS keys for synth_av: babies +1..bs-1 and -L+1..-L+bs-1, giants ±g·bs
    bs = 1
    while bs * bs < L:
        bs <<= 1
    if bs * bs > L:
        bs >>= 1
    gs = L // bs
    bsgs_shifts = set()
    for b in range(1, bs):
        bsgs_shifts.add(b)
    for b in range(bs):
        bsgs_shifts.add(b - L)
    for g in range(1, gs):
        bsgs_shifts.add(g * bs)
    n_bsgs = be.register_rotation_keys(sorted(bsgs_shifts))
    log(f"  +{n_bsgs} keys (BSGS bs={bs} gs={gs})")

    rng = np.random.default_rng(0)
    # Note: NO Wq, NO Wk — Synthesizer eliminates them
    Wv = rng.standard_normal((hidden, hidden)) * 0.05
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    W1 = rng.standard_normal((inter, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, inter)) * 0.05
    gamma1, beta1 = np.ones(hidden), np.zeros(hidden)
    gamma2, beta2 = np.ones(hidden), np.zeros(hidden)

    invsqrt_coeffs = [1.0, -0.5, 0.375]

    # ------- Synthesizer: pre-encoded plaintext attention pattern --------
    # Realistic init: row-stochastic (softmax) random pattern per head.
    # In production: A is distilled from BERT teacher's averaged attention.
    log("Build & encode Synthesizer BSGS plaintexts (one-time per layer)...")
    t = time.time()
    A_logits = rng.standard_normal((num_heads, L, L)) * 0.3
    A_full = np.exp(A_logits) / np.exp(A_logits).sum(axis=-1, keepdims=True)
    A_per_bundle = [
        A_full[k * HEADS_PER_CT:(k + 1) * HEADS_PER_CT]
        for k in range(NUM_HEAD_BUNDLES)
    ]
    bsgs_pts_per_bundle = [
        encode_synthesizer_bsgs(
            be, A_per_bundle[k], L=L, head_dim=head_dim,
            num_heads_per_ct=HEADS_PER_CT,
        ) for k in range(NUM_HEAD_BUNDLES)
    ]
    log(f"  encoded BSGS pts in {time.time()-t:.2f}s")

    # No mask cache needed for BSGS variant (plaintexts are all stored).

    log(f"Pack {BATCH} inputs...")
    x_per_input = []
    for _ in range(BATCH):
        x = rng.standard_normal((L, hidden)) * 0.1
        x_per_input.append(pack_colmajor_multi(be, x, L=L, hidden=hidden))
    log(f"  done; depth={be._ops.depth(x_per_input[0][0])}")

    times = {}
    layer_t = time.time()

    def stamp(name, dt):
        times[name] = times.get(name, 0.0) + dt

    # --- ONLY Wv (Wq, Wk eliminated) ---
    t = time.time()
    Wv_per = linear_colmajor_multi_streaming_batched(
        be, x_per_input, Wv, L=L, in_dim=hidden, out_dim=hidden,
    )
    stamp("Wv", time.time() - t)

    # --- Synthesizer-BSGS attention per input ---
    t = time.time()
    O_per = []
    for i in range(BATCH):
        V = Wv_per[i]
        O_i = [attn_synthesizer_bsgs_nexus(
                be, V[k], bsgs_pts_per_bundle[k],
                head_dim=head_dim, num_heads_per_ct=HEADS_PER_CT,
            ) for k in range(NUM_HEAD_BUNDLES)]
        O_per.append(O_i)
    stamp("synth_av_bsgs", time.time() - t)
    del Wv_per; gc.collect()

    # --- Wo (batched) ---
    t = time.time()
    Wo_out_per = linear_colmajor_multi_streaming_batched(
        be, O_per, Wo, L=L, in_dim=hidden, out_dim=hidden,
    )
    stamp("Wo", time.time() - t)
    del O_per; gc.collect()

    # --- residual + LN1 ---
    t = time.time()
    res1_per = [add_multi(be, Wo_out_per[i], x_per_input[i]) for i in range(BATCH)]
    stamp("residual1", time.time() - t)
    del Wo_out_per; gc.collect()

    t = time.time()
    h1_per = [layernorm_colmajor_multi(
            be, res1_per[i], L=L, hidden=hidden,
            invsqrt_power_coeffs=invsqrt_coeffs,
            invsqrt_interval=(-1.0, 1.0),
            gamma=gamma1, beta=beta1,
        ) for i in range(BATCH)]
    stamp("LN1", time.time() - t)
    del res1_per; gc.collect()

    # --- W1 (batched) ---
    t = time.time()
    H_per = linear_colmajor_multi_streaming_batched(
        be, h1_per, W1, L=L, in_dim=hidden, out_dim=inter,
    )
    stamp("W1", time.time() - t)

    # --- GELU ---
    t = time.time()
    G_per = [[be.polyval(c, [0.5, 0.398, 0.0, -0.0066]) for c in H_i] for H_i in H_per]
    stamp("gelu", time.time() - t)
    del H_per; gc.collect()

    # --- W2 (batched) ---
    t = time.time()
    H2_per = linear_colmajor_multi_streaming_batched(
        be, G_per, W2, L=L, in_dim=inter, out_dim=hidden,
    )
    stamp("W2", time.time() - t)
    del G_per; gc.collect()

    # --- residual + LN2 ---
    t = time.time()
    res2_per = [add_multi(be, H2_per[i], h1_per[i]) for i in range(BATCH)]
    stamp("residual2", time.time() - t)
    del H2_per, h1_per; gc.collect()

    t = time.time()
    out_per = [layernorm_colmajor_multi(
            be, res2_per[i], L=L, hidden=hidden,
            invsqrt_power_coeffs=invsqrt_coeffs,
            invsqrt_interval=(-1.0, 1.0),
            gamma=gamma2, beta=beta2,
        ) for i in range(BATCH)]
    stamp("LN2", time.time() - t)

    layer_dt = time.time() - layer_t
    print()
    print("=" * 72)
    print(f"SYNTHESIZER-LPAN  BATCH={BATCH}  L={L}  hidden={hidden}")
    print(f"  Layer total wall: {layer_dt:.2f}s")
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        per_input = v / BATCH
        print(f"     {k:<10s} total={v*1000:7.0f}ms  per-input={per_input*1000:6.0f}ms")
    per_input_layer = layer_dt / BATCH
    print(f"  Per-input layer: {per_input_layer:.2f}s")
    print(f"  Per-input 12-layer: {12 * per_input_layer:.1f}s")
    print(f"  Baseline single-input (no batch, full LPAN): 833s")
    print(f"  Batched-LPAN (prior result):                 534s")
    print(f"  Synthesizer-LPAN speedup vs LPAN baseline: {833 / (12 * per_input_layer):.2f}×")
    print(f"  vs batched-LPAN: {534 / (12 * per_input_layer):.2f}×")
    if 12 * per_input_layer < 100:
        print(f"  *** UNDER 100 SECONDS ON SINGLE H100 ***")
    print("=" * 72)


if __name__ == "__main__":
    main()
