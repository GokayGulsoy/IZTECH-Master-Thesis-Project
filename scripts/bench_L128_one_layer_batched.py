"""1-layer batched bench: shared W encoding across N inputs.

Linears use linear_colmajor_multi_streaming_batched (W encoded ONCE per giant,
applied to all N inputs). Attention/LN/etc remain per-input (those don't share
encoded W; they're ct-ct ops).

Reports per-input wall and projects 12-layer.
"""
import gc, time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor_multi,
    qk_scores_nexus, attn_apply_nexus,
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
    BATCH = 4

    log(f"Config L={L} hidden={hidden} heads={num_heads} BATCH={BATCH}")
    N = 1 << 16
    chain = 28
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
    log(f"  +{n_new} keys")

    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((hidden, hidden)) * 0.05
    Wk = rng.standard_normal((hidden, hidden)) * 0.05
    Wv = rng.standard_normal((hidden, hidden)) * 0.05
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    W1 = rng.standard_normal((inter, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, inter)) * 0.05
    gamma1, beta1 = np.ones(hidden), np.zeros(hidden)
    gamma2, beta2 = np.ones(hidden), np.zeros(hidden)

    invsqrt_coeffs = [1.0, -0.5, 0.375]
    softmax_coeffs = [1.0, 0.5, 0.125]
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    # Mask plaintexts are identical across inputs — ONE cache shared by all.
    attn_caches = [{} for _ in range(3)]

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

    # --- Wq Wk Wv (shared W, batched across N inputs) ---
    QKV_cts = {}
    for tag, W in [("Wq", Wq), ("Wk", Wk), ("Wv", Wv)]:
        t = time.time()
        Y_per = linear_colmajor_multi_streaming_batched(
            be, x_per_input, W, L=L, in_dim=hidden, out_dim=hidden,
        )
        stamp(tag, time.time() - t)
        QKV_cts[tag] = Y_per

    # --- attention per input ---
    t = time.time()
    S_per = []
    for i in range(BATCH):
        Q = QKV_cts["Wq"][i]
        K = QKV_cts["Wk"][i]
        S_i = [qk_scores_nexus(
                be, Q[k], K[k], L=L, head_dim=head_dim, scale=inv_sqrt_d,
                num_heads_per_ct=4, cache=attn_caches[k],
            ) for k in range(len(Q))]
        S_per.append(S_i)
    stamp("qk", time.time() - t)
    del QKV_cts["Wq"], QKV_cts["Wk"]; gc.collect()

    t = time.time()
    Asm_per = [[be.polyval(s, list(softmax_coeffs)) for s in S_i] for S_i in S_per]
    stamp("softmax", time.time() - t)
    del S_per; gc.collect()

    t = time.time()
    O_per = []
    for i in range(BATCH):
        V = QKV_cts["Wv"][i]
        O_i = [attn_apply_nexus(
                be, Asm_per[i][k], V[k], L=L, head_dim=head_dim,
                num_heads_per_ct=4, cache=attn_caches[k],
            ) for k in range(len(V))]
        O_per.append(O_i)
    stamp("av", time.time() - t)
    del QKV_cts, Asm_per; gc.collect()

    # --- Wo (batched) ---
    t = time.time()
    Wo_out_per = linear_colmajor_multi_streaming_batched(
        be, O_per, Wo, L=L, in_dim=hidden, out_dim=hidden,
    )
    stamp("Wo", time.time() - t)
    del O_per; gc.collect()

    # --- residual + LN1 (per input) ---
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
    print(f"BATCH={BATCH}  L={L}  hidden={hidden}  N={N}  chain={chain}")
    print(f"  Layer total wall: {layer_dt:.2f}s")
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        per_input = v / BATCH
        print(f"     {k:<10s} total={v*1000:7.0f}ms  per-input={per_input*1000:6.0f}ms")
    per_input_layer = layer_dt / BATCH
    print(f"  Per-input layer time: {per_input_layer:.2f}s")
    print(f"  Per-input 12-layer projection: {12 * per_input_layer:.1f}s")
    print(f"  Baseline (no batch): 833s/12-layer")
    print(f"  Speedup: {833 / (12 * per_input_layer):.2f}×")
    print("=" * 72)


if __name__ == "__main__":
    main()
