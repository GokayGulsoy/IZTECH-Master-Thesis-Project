"""Phase 7e: full e2e BERT-style encoder bench with all Phase 7 optimizations.

Runs N encoder layers end-to-end:
  - Per-head Q/K/V projection (Halevi-Shoup BSGS, cached)
  - Diagonal attention (Phase 7d C++ batched)
  - LPAN softmax/GELU/LN polynomials (Phase 7c low-depth LN)
  - Output projection + W1 + W2

Reports per-layer breakdown and total e2e wall time.

Config aimed at fitting current slot budget (N=2^16):
  hidden=512, 8 heads, head_dim=64, L=32  (BERT-Small @ short seq)
  block_attn = max(L, head_dim) = 64
  2*L*block_attn = 4096 << 32768 ✓
"""
import time, gc
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.matrix_packing import MatrixPackedTensor, next_pow2
from fhe_thesis.encryption.coefficients import PolyCoeffs
from fhe_thesis.encryption.ops_matrix import (
    enc_linear_matrix, enc_layernorm_matrix, enc_gelu_matrix,
    enc_self_attention_diagonal,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    hidden, n_heads, L, n_layers = 512, 8, 32, 6
    head_dim = hidden // n_heads
    block = next_pow2(hidden)  # 512
    log(f"Config: hidden={hidden} heads={n_heads} head_dim={head_dim} "
        f"L={L} n_layers={n_layers} block={block}")

    log("Init backend...")
    be = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth}")

    rng = np.random.default_rng(42)
    def W(out_d, in_d): return rng.standard_normal((out_d, in_d)) * 0.05
    def b(d): return np.zeros(d)

    layers = []
    for _ in range(n_layers):
        layers.append({
            "Wq": W(hidden, hidden), "bq": b(hidden),
            "Wk": W(hidden, hidden), "bk": b(hidden),
            "Wv": W(hidden, hidden), "bv": b(hidden),
            "Wo": W(hidden, hidden), "bo": b(hidden),
            "W1": W(4*hidden, hidden), "b1": b(4*hidden),
            "W2": W(hidden, 4*hidden), "b2": b(hidden),
            "g1": np.ones(hidden), "be1": np.zeros(hidden),
            "g2": np.ones(hidden), "be2": np.zeros(hidden),
        })

    sm = PolyCoeffs(power_coeffs=np.array([1.0, 0.5, 0.125]),
                    interval=(-4.0, 4.0), degree=2, per_head=False)
    invsqrt = [1.0, -0.5, 0.375]
    invsqrt_iv = (0.01, 4.0)
    gelu_pc = [0.5, 0.398, 0.0, -0.0066]
    gelu_iv = (-4.0, 4.0)

    x = rng.standard_normal((L, hidden)) * 0.05
    log(f"Encrypt input ({L}, {hidden})...")
    t0 = time.time()
    ct = MatrixPackedTensor.encrypt(be, x, block=block)
    log(f"  encrypted in {(time.time()-t0)*1000:.0f}ms; ct depth={be._ops.depth(ct.cts[0])}, n_cts={len(ct.cts)}")

    times = []
    e2e_t0 = time.time()
    for li, p in enumerate(layers):
        log(f"--- Layer {li} ---")
        layer_t0 = time.time()
        stage_t = {}

        t = time.time()
        attn = enc_self_attention_diagonal(
            be, ct, p["Wq"], p["bq"], p["Wk"], p["bk"], p["Wv"], p["bv"],
            p["Wo"], p["bo"], sm, n_heads,
        )
        stage_t["attn+Wo"] = time.time() - t
        d = be._ops.depth(attn.cts[0])
        log(f"  attn+Wo:    {stage_t['attn+Wo']*1000:7.0f}ms  depth={d}")

        # If we're running out of depth, bootstrap.
        if d >= be._max_depth - 8:
            t = time.time()
            attn = MatrixPackedTensor.from_ciphertexts(
                [be.bootstrap(c) for c in attn.cts],
                seq_len=attn.seq_len, hidden_dim=attn.hidden_dim,
                block=attn.block, tokens_per_ct=attn.tokens_per_ct,
                num_slots=attn.num_slots,
            )
            stage_t["bootstrap_post_attn"] = time.time() - t
            log(f"  bootstrap_a:{stage_t['bootstrap_post_attn']*1000:7.0f}ms  depth={be._ops.depth(attn.cts[0])}")

        t = time.time()
        ln1 = enc_layernorm_matrix(
            be, attn, gamma=p["g1"], beta=p["be1"],
            invsqrt_power_coeffs=invsqrt, invsqrt_interval=invsqrt_iv,
        )
        stage_t["ln1"] = time.time() - t
        d = be._ops.depth(ln1.cts[0])
        log(f"  ln1:        {stage_t['ln1']*1000:7.0f}ms  depth={d}")

        if d >= be._max_depth - 8:
            t = time.time()
            ln1 = MatrixPackedTensor.from_ciphertexts(
                [be.bootstrap(c) for c in ln1.cts],
                seq_len=ln1.seq_len, hidden_dim=ln1.hidden_dim,
                block=ln1.block, tokens_per_ct=ln1.tokens_per_ct,
                num_slots=ln1.num_slots,
            )
            stage_t["bootstrap_post_ln1"] = time.time() - t
            log(f"  bootstrap_l:{stage_t['bootstrap_post_ln1']*1000:7.0f}ms  depth={be._ops.depth(ln1.cts[0])}")

        t = time.time()
        h1 = enc_linear_matrix(be, ln1, p["W1"], bias=p["b1"])
        stage_t["W1"] = time.time() - t
        log(f"  W1:         {stage_t['W1']*1000:7.0f}ms  depth={be._ops.depth(h1.cts[0])}")

        t = time.time()
        g = enc_gelu_matrix(be, h1, gelu_pc, gelu_iv)
        stage_t["gelu"] = time.time() - t
        log(f"  gelu:       {stage_t['gelu']*1000:7.0f}ms  depth={be._ops.depth(g.cts[0])}")

        t = time.time()
        h2 = enc_linear_matrix(be, g, p["W2"], bias=p["b2"])
        stage_t["W2"] = time.time() - t
        log(f"  W2:         {stage_t['W2']*1000:7.0f}ms  depth={be._ops.depth(h2.cts[0])}")

        d = be._ops.depth(h2.cts[0])
        if d >= be._max_depth - 8:
            t = time.time()
            h2 = MatrixPackedTensor.from_ciphertexts(
                [be.bootstrap(c) for c in h2.cts],
                seq_len=h2.seq_len, hidden_dim=h2.hidden_dim,
                block=h2.block, tokens_per_ct=h2.tokens_per_ct,
                num_slots=h2.num_slots,
            )
            stage_t["bootstrap_post_w2"] = time.time() - t
            log(f"  bootstrap_w:{stage_t['bootstrap_post_w2']*1000:7.0f}ms  depth={be._ops.depth(h2.cts[0])}")

        t = time.time()
        ct = enc_layernorm_matrix(
            be, h2, gamma=p["g2"], beta=p["be2"],
            invsqrt_power_coeffs=invsqrt, invsqrt_interval=invsqrt_iv,
        )
        stage_t["ln2"] = time.time() - t
        log(f"  ln2:        {stage_t['ln2']*1000:7.0f}ms  depth={be._ops.depth(ct.cts[0])}")

        d = be._ops.depth(ct.cts[0])
        if d >= be._max_depth - 8:
            t = time.time()
            ct = MatrixPackedTensor.from_ciphertexts(
                [be.bootstrap(c) for c in ct.cts],
                seq_len=ct.seq_len, hidden_dim=ct.hidden_dim,
                block=ct.block, tokens_per_ct=ct.tokens_per_ct,
                num_slots=ct.num_slots,
            )
            stage_t["bootstrap_eol"] = time.time() - t
            log(f"  bootstrap_e:{stage_t['bootstrap_eol']*1000:7.0f}ms  depth={be._ops.depth(ct.cts[0])}")

        layer_t = time.time() - layer_t0
        log(f"  >> layer total: {layer_t:.2f}s")
        times.append((layer_t, stage_t))
        gc.collect()

    e2e = time.time() - e2e_t0
    print()
    print("=" * 60)
    print(f"E2E {n_layers}-layer ({hidden}d, {n_heads}h, L={L}): {e2e:.1f} s")
    print(f"  avg per-layer: {e2e/n_layers:.2f} s")
    print(f"  vs NEXUS BERT-base e2e: 37 s")
    print("=" * 60)


if __name__ == "__main__":
    main()
