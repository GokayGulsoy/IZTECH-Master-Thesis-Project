"""Phase 7e-5: end-to-end column-major encoder layer bench at BERT-base.

Pipeline per layer:
  hidden=768 col-major ct
  → for h in [0, num_heads):
       Q_h = linear_colmajor(Wq[h] @ x[:, :768] -> head_dim=64)   # 768 → 64
       K_h = linear_colmajor(Wk[h])
       V_h = linear_colmajor(Wv[h])
       S_h = qk_scores_nexus(Q_h, K_h)
       Asm = softmax_poly(S_h)
       O_h = attn_apply_nexus(Asm, V_h)
       scatter O_h into slots [h*head_dim*L, (h+1)*head_dim*L) of full ct
  → Wo (hidden=768 → hidden=768) col-major linear
  → residual + col-major LN
  → W1 (768 → 3072)         (split into 4 head-sized chunks if it fits, else
                              run per-chunk col-major linear and accumulate
                              into chunked ciphertext set)
  → poly GELU
  → W2 (3072 → 768)
  → residual + col-major LN

For now we run hidden→head linears (Wq/k/v) at full hidden width 768 → head_dim=64.
W1/W2 use the same `linear_colmajor` (768→3072 needs 3072·L = 98304 slots
which exceeds N=2^16 ⇒ store FFN intermediate as 4 separate head-sized
ciphertexts (each of width 768)).
"""
import gc
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor, unpack_colmajor,
    linear_colmajor, layernorm_colmajor,
    qk_scores_nexus, attn_apply_nexus,
    prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 32
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads     # 64
    inter = 4 * hidden                  # 3072

    log(f"Config L={L} hidden={hidden} heads={num_heads} head_dim={head_dim}")

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be.capabilities.n_slots}")

    # Keys: max_dim=hidden_padded (1024) covers all linear_colmajor + LN
    # rotations. attn kernels also use ±d*L for d ∈ [0, L) — already covered
    # since L=32 < 1024.
    log("Pre-register rotation keys (max_dim=1024)...")
    t = time.time()
    n_new = prepare_colmajor_keys(be, L=L, max_dim=1024)
    log(f"  +{n_new} keys in {time.time()-t:.1f}s")

    rng = np.random.default_rng(0)
    # Per-head weights, then stacked to full (hidden, hidden) for fused linears.
    Wq_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    Wk_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    Wv_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    # Stack head dim → full hidden output.
    Wq = Wq_h.reshape(num_heads * head_dim, hidden)
    Wk = Wk_h.reshape(num_heads * head_dim, hidden)
    Wv = Wv_h.reshape(num_heads * head_dim, hidden)
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    # FFN: we run W1 column-blocked into 4 slabs of shape (head_dim*num_heads, hidden) = (768, 768)
    # so that each slab fits (768·L = 24576 ≤ 32768).
    W1 = rng.standard_normal((inter, hidden)) * 0.05      # 3072 × 768
    W2 = rng.standard_normal((hidden, inter)) * 0.05      # 768 × 3072
    gamma1, beta1 = np.ones(hidden), np.zeros(hidden)
    gamma2, beta2 = np.ones(hidden), np.zeros(hidden)

    invsqrt_coeffs = [1.0, -0.5, 0.375]
    softmax_coeffs = [1.0, 0.5, 0.125]

    x = rng.standard_normal((L, hidden)) * 0.1
    log("Encrypt input col-major...")
    x_ct = pack_colmajor(be, x, L=L, head_dim=hidden)

    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    times = {}

    def stamp(name, dt, depth=None):
        times[name] = times.get(name, 0.0) + dt
        d = f" depth={depth}" if depth is not None else ""
        log(f"  {name:<22s} {dt*1000:8.1f} ms{d}")

    # ──────────── ATTENTION ────────────
    log("=== ATTENTION (fused QKV + 4-head packing) ===")
    # 3 fused linears: Wq/Wk/Wv each maps hidden → hidden, packing all
    # 12 heads into one ct (head h occupies slots [h·head_dim·L, (h+1)·head_dim·L)).
    gc.collect()
    t = time.time()
    Q_full = linear_colmajor(be, x_ct, Wq, L=L, in_dim=hidden, out_dim=hidden)
    stamp("Wq_fused", time.time() - t, depth=be._ops.depth(Q_full))
    gc.collect()
    t = time.time()
    K_full = linear_colmajor(be, x_ct, Wk, L=L, in_dim=hidden, out_dim=hidden)
    stamp("Wk_fused", time.time() - t)
    gc.collect()
    t = time.time()
    V_full = linear_colmajor(be, x_ct, Wv, L=L, in_dim=hidden, out_dim=hidden)
    stamp("Wv_fused", time.time() - t)

    # Group heads by num_heads_per_ct = 4. Each group: extract 4 heads side-by-side
    # via one rotation, run packed attention, scatter back.
    num_heads_per_ct = 4
    n_groups = num_heads_per_ct  # NOT — n_groups = num_heads // num_heads_per_ct
    n_groups = num_heads // num_heads_per_ct
    head_block = head_dim * L  # 2048 slots per head
    group_block = num_heads_per_ct * head_block  # 8192 slots per group

    O_full = None
    for g in range(n_groups):
        gc.collect()
        # Extract group g: rotate left by g·group_block. The group's slots
        # land at [0, group_block).
        if g == 0:
            Qg, Kg, Vg = Q_full, K_full, V_full
        else:
            t = time.time()
            shift = g * group_block
            Qg = be.rotate(Q_full, shift)
            Kg = be.rotate(K_full, shift)
            Vg = be.rotate(V_full, shift)
            stamp(f"extract[g{g}]", time.time() - t)

        t = time.time()
        Sg = qk_scores_nexus(be, Qg, Kg, L=L, head_dim=head_dim,
                             scale=inv_sqrt_d,
                             num_heads_per_ct=num_heads_per_ct)
        stamp(f"qk[g{g}]", time.time() - t)

        t = time.time()
        Asm = be.polyval(Sg, list(softmax_coeffs))
        stamp(f"softmax[g{g}]", time.time() - t)

        t = time.time()
        Og = attn_apply_nexus(be, Asm, Vg, L=L, head_dim=head_dim,
                              num_heads_per_ct=num_heads_per_ct)
        stamp(f"av[g{g}]", time.time() - t)

        # Scatter: Og lives in slots [0, group_block); shift to
        # [g·group_block, (g+1)·group_block).
        t = time.time()
        if g == 0:
            O_full = Og
        else:
            Og_shifted = be.rotate(Og, -g * group_block)
            O_full = be.add(O_full, Og_shifted)
        stamp(f"scatter[g{g}]", time.time() - t)

    # ──────────── Wo + residual + LN1 ────────────
    log("=== Wo + LN1 ===")
    t = time.time()
    O = linear_colmajor(be, O_full, Wo, L=L, in_dim=hidden, out_dim=hidden)
    stamp("Wo", time.time() - t, depth=be._ops.depth(O))

    t = time.time()
    res1 = be.add(O, x_ct)
    stamp("residual1", time.time() - t)

    t = time.time()
    h1 = layernorm_colmajor(
        be, res1, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma1, beta=beta1,
    )
    stamp("LN1", time.time() - t, depth=be._ops.depth(h1))

    # ──────────── FFN W1 → GELU → W2 ────────────
    # W1 output dim 3072 doesn't fit in one ct (3072·32 = 98304 > 32768).
    # Split into 4 slabs of out_dim=768 each.
    log("=== FFN ===")
    slabs = []
    for s in range(4):
        gc.collect()
        Ws = W1[s * hidden:(s + 1) * hidden, :]   # (768, 768)
        t = time.time()
        Hs = linear_colmajor(be, h1, Ws, L=L, in_dim=hidden, out_dim=hidden)
        stamp(f"W1[s{s}]", time.time() - t, depth=be._ops.depth(Hs))
        t = time.time()
        Gs = be.polyval(Hs, [0.5, 0.398, 0.0, -0.0066])
        stamp(f"gelu[s{s}]", time.time() - t, depth=be._ops.depth(Gs))
        slabs.append(Gs)

    # W2: out shape (768,) = Σ_s W2[:, s_block] · gelu_s_block.
    # Because each Gs holds the FULL inter activation in its own ct? No —
    # each Gs holds slab s of size 768, in col-major layout of width 768.
    # W2 maps inter=3072 → hidden=768. So per slab: linear_colmajor on Gs
    # using W2[:, s*768:(s+1)*768] (shape (768, 768)), then sum.
    log("=== W2 ===")
    H2 = None
    for s in range(4):
        gc.collect()
        Ws2 = W2[:, s * hidden:(s + 1) * hidden]   # (768, 768)
        t = time.time()
        Y = linear_colmajor(be, slabs[s], Ws2, L=L, in_dim=hidden, out_dim=hidden)
        stamp(f"W2[s{s}]", time.time() - t, depth=be._ops.depth(Y))
        H2 = Y if H2 is None else be.add(H2, Y)

    # ──────────── residual + LN2 ────────────
    log("=== LN2 ===")
    t = time.time()
    res2 = be.add(H2, h1)
    stamp("residual2", time.time() - t)

    t = time.time()
    out = layernorm_colmajor(
        be, res2, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma2, beta=beta2,
    )
    stamp("LN2", time.time() - t, depth=be._ops.depth(out))

    # ──────────── summary ────────────
    print()
    print("=" * 70)
    print(f"COL-MAJOR PER-LAYER BENCH  (L={L} hidden={hidden} heads={num_heads})")
    print("=" * 70)
    total = sum(times.values())
    # group by stage
    groups = {
        "Wq/Wk/Wv (3 fused)":   times.get("Wq_fused", 0.0) + times.get("Wk_fused", 0.0) + times.get("Wv_fused", 0.0),
        "extract (groups)":     sum(v for k, v in times.items() if k.startswith("extract[")),
        "qk (groups)":           sum(v for k, v in times.items() if k.startswith("qk[")),
        "softmax (groups)":      sum(v for k, v in times.items() if k.startswith("softmax[")),
        "av (groups)":           sum(v for k, v in times.items() if k.startswith("av[")),
        "scatter":               sum(v for k, v in times.items() if k.startswith("scatter[")),
        "Wo":                    times.get("Wo", 0.0),
        "LN1":                   times.get("LN1", 0.0),
        "W1 (4 slabs)":          sum(v for k, v in times.items() if k.startswith("W1[")),
        "GELU (4 slabs)":        sum(v for k, v in times.items() if k.startswith("gelu[")),
        "W2 (4 slabs)":          sum(v for k, v in times.items() if k.startswith("W2[")),
        "LN2":                   times.get("LN2", 0.0),
        "residual":              times.get("residual1", 0.0) + times.get("residual2", 0.0),
    }
    for name, dt in groups.items():
        print(f"  {name:<32s}  {dt*1000:8.1f} ms   ({100*dt/total:5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL (1 layer)':<32s}  {total*1000:8.1f} ms")
    print(f"  {'PROJECTED 12-layer e2e':<32s}  {total*12:8.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
