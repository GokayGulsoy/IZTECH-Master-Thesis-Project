"""1-layer Synthesizer-LPAN bench (BATCH=4).

Replaces:
  - Wq, Wk linears        → ELIMINATED (no Q, no K needed)
  - qk_scores_nexus       → ELIMINATED (no Q·K^T)
  - softmax-poly on S     → ELIMINATED (already absorbed into A at train time)
  - attn_apply_nexus (av) → attn_synthesizer (pt·ct only, no ct·ct)

Keeps: Wv, Wo, W1, W2 batched linears, residuals, LN, GELU.

Reports per-input wall and projects 12-layer to validate the <100s path.
"""
import argparse
import gc, time
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.colmajor import pack_colmajor_multi, prepare_colmajor_keys
from fhe_thesis.encryption.attention import (
    encode_synthesizer_bsgs,
    attn_synthesizer_bsgs,
)
from fhe_thesis.encryption.layernorm import layernorm_colmajor_multi
from fhe_thesis.encryption.linear import linear_colmajor_multi_streaming_batched
from fhe_thesis.encryption.multi import add_multi


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


DEFAULT_GELU_POWER_COEFFS = [0.5, 0.398, 0.0, -0.0066]
DEFAULT_LN_POWER_COEFFS = [1.0, -0.5, 0.375]
DEFAULT_LN_INTERVAL = (-1.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="1-layer Synthesizer-LPAN benchmark")
    parser.add_argument("--checkpoint", default=None, help="Exported bench checkpoint JSON")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to benchmark from the export JSON")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: BATCH env or 8)")
    parser.add_argument("--chain", type=int, default=None, help="CKKS chain length (default: CHAIN env or 20)")
    return parser.parse_args()


def load_bench_checkpoint(path, layer_idx):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    layers = payload.get("layers", [])
    for layer in layers:
        if int(layer.get("layer_idx", -1)) == layer_idx:
            return payload.get("metadata", {}), layer
    raise KeyError(f"Layer {layer_idx} not found in {path}")


def main():
    args = parse_args()
    L = args.seq_len
    hidden = args.hidden
    num_heads = args.num_heads
    head_dim = hidden // num_heads
    inter = 4 * hidden
    BATCH = args.batch or int(os.environ.get("BATCH", 8))
    HEADS_PER_CT = min(4, num_heads)
    if num_heads % HEADS_PER_CT != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by HEADS_PER_CT={HEADS_PER_CT}"
        )
    NUM_HEAD_BUNDLES = num_heads // HEADS_PER_CT

    gelu_coeffs = list(DEFAULT_GELU_POWER_COEFFS)
    ln1_coeffs = list(DEFAULT_LN_POWER_COEFFS)
    ln2_coeffs = list(DEFAULT_LN_POWER_COEFFS)
    ln1_interval = DEFAULT_LN_INTERVAL
    ln2_interval = DEFAULT_LN_INTERVAL
    exported_patterns = None

    if args.checkpoint is not None:
        meta, layer_payload = load_bench_checkpoint(args.checkpoint, args.layer)
        L = int(meta.get("seq_len", L))
        hidden = int(meta.get("hidden_size", hidden))
        num_heads = int(meta.get("num_attention_heads", num_heads))
        inter = int(meta.get("intermediate_size", 4 * hidden))
        head_dim = hidden // num_heads
        HEADS_PER_CT = min(4, num_heads)
        if num_heads % HEADS_PER_CT != 0:
            raise ValueError(
                f"Checkpoint num_heads={num_heads} is incompatible with HEADS_PER_CT={HEADS_PER_CT}"
            )
        NUM_HEAD_BUNDLES = num_heads // HEADS_PER_CT
        exported_patterns = np.asarray(layer_payload["attention_pattern"], dtype=np.float64)
        if exported_patterns.shape != (num_heads, L, L):
            raise ValueError(
                f"Checkpoint attention pattern shape {exported_patterns.shape} "
                f"does not match ({num_heads}, {L}, {L})"
            )
        gelu_coeffs = list(layer_payload["gelu"]["power_coeffs"])
        ln1_coeffs = list(layer_payload["ln1"]["power_coeffs"])
        ln1_interval = tuple(layer_payload["ln1"]["interval"])
        ln2_coeffs = list(layer_payload["ln2"]["power_coeffs"])
        ln2_interval = tuple(layer_payload["ln2"]["interval"])
        log(f"Loaded checkpoint {args.checkpoint} layer={args.layer}")

    log(f"Config L={L} hidden={hidden} heads={num_heads} BATCH={BATCH}")
    N = 1 << 16
    chain = args.chain or int(os.environ.get("CHAIN", 20))   # Synthesizer eliminates softmax-poly depth
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

    # ------- Synthesizer: pre-encoded plaintext attention pattern --------
    # Realistic init: row-stochastic (softmax) random pattern per head.
    # In production: A is distilled from BERT teacher's averaged attention.
    log("Build & encode Synthesizer BSGS plaintexts (one-time per layer)...")
    t = time.time()
    if exported_patterns is None:
        A_logits = rng.standard_normal((num_heads, L, L)) * 0.3
        A_full = np.exp(A_logits) / np.exp(A_logits).sum(axis=-1, keepdims=True)
    else:
        A_full = exported_patterns
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
        O_i = [attn_synthesizer_bsgs(
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
            invsqrt_power_coeffs=ln1_coeffs,
            invsqrt_interval=ln1_interval,
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
    G_per = [[be.polyval(c, gelu_coeffs) for c in H_i] for H_i in H_per]
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
            invsqrt_power_coeffs=ln2_coeffs,
            invsqrt_interval=ln2_interval,
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
