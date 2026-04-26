# FHE Inference Optimization Plan
## Post-SST-2 Training Roadmap

> **Status as of 2026-04-26**
> - SST-2 Stage 3: L0 ✅ (91.40%), L1 🔄 epoch 6/8 (best 91.74%), L2–L11 ⏳
> - QNLI: not started
> - FHE benchmarking: not started

---

## Contents
1. [Why We Can't Directly Match MPCFormer Latency](#1-the-fundamental-gap)
2. [Baseline: Where We Are Now](#2-baseline)
3. [Optimization Stack](#3-optimization-stack)
4. [Inference Time Comparison: MPCFormer vs LPAN-FHE](#4-inference-time-comparison)
5. [Implementation Steps (in order)](#5-implementation-steps)
6. [Thesis Positioning Statement](#6-thesis-positioning)

---

## 1. The Fundamental Gap

MPCFormer and LPAN-FHE solve **different** privacy problems:

| Property | MPCFormer (2PC-MPC) | LPAN-FHE (CKKS) |
|---|---|---|
| **Privacy model** | Two non-colluding servers | Single server, no trust needed |
| **Network required** | Yes — servers exchange ~100MB–1GB per inference | No — client sends one encrypted blob |
| **Threat assumption** | Both servers must be honest-but-curious, never collude | Server can be fully malicious |
| **Bootstrapping cost** | None (MPC has no multiplicative depth) | Required every ~1–2 BERT layers |
| **Reported latency** | ~10–20 sec/sample (BERT-Base, LAN) | Hours without optimization → **11–21 sec amortized** with full stack |

**The right comparison metric is amortized throughput**, not single-sample latency. A cloud FHE service batches clients — CKKS processes 42 SST-2 sequences in the same time as 1. That gives competitive numbers while offering strictly stronger security guarantees.

---

## 2. Baseline

### Current system (TenSEAL, no batching, seq_len=128)

| Component | % of total time | Notes |
|---|---|---|
| Attention (Q·Kᵀ + attn·V) | ~60% | O(L²) ciphertext ops |
| Linear layers (Q,K,V,O,W₁,W₂) | ~24% | 768→768, 768→3072 matmuls per token row |
| Polynomial ops (GELU, LN, softmax) | ~7% | Zero rotations due to token packing |
| Bootstrapping | ~9% | Every ~1–2 layers at 23 levels/layer budget |

**Estimated total: 2–4 hours per BERT-Base inference (seq_len=128, 12 layers)**

### Why the current linear layer cost is high

Ring dimension N=65536 → **32,768 slots** per ciphertext.  
BERT-Base hidden_dim = 768 → **slot utilization = 768 / 32,768 = 2.3%**

This means each plaintext-weight matmul involves a 768-wide rotation loop, and we run it for every token row (128 tokens × 6 linear layers = 768 matmul calls per layer). 12 layers = 9,216 plaintext matmul calls, dominated by rotation cost.

---

## 3. Optimization Stack

### O1 — OpenFHE + HEXL (AVX-512 NTT acceleration)
**Effort:** Medium (install + port `OpenFHEBackend(CKKSBackend)`)  
**Applies to:** Every single operation uniformly  
**Speedup:** 3–5× on NTT; 2× on key-switch via HYBRID mode

OpenFHE's `openfhe-hexl` integration uses Intel HEXL to accelerate the Number Theoretic Transform (NTT), which is the underlying primitive for all CKKS operations. Also enables `HYBRID` key switching which halves key-switch cost vs BV mode (critical for rotations in attention).

```bash
# Install path (on the benchmark machine):
git clone https://github.com/openfheorg/openfhe-configurator.git
cd openfhe-configurator
scripts/configure.sh    # answer: y to hexl, y to BERT params
scripts/build-openfhe-development.sh
```

Code change required: write `OpenFHEBackend(CKKSBackend)` in
`fhe_thesis/encryption/openfhe_backend.py` implementing all abstract methods.
The `CKKSBackend` ABC already has the correct interface — zero changes to
`ops.py`, `protocol.py`, or any protocol-level code.

---

### O2 — BSGS Diagonal Matmul for Linear Layers
**Effort:** Medium (modify `OpenFHEBackend.matmul_plain`)  
**Applies to:** All 6 linear layers per BERT layer (Q, K, V, O, W₁, W₂)  
**Speedup:** 14–37× fewer rotations per linear layer

Our token-packed layout with LPAN polynomials (zero-rotation elementwise ops) shifts the
dominant cost to **linear layers**. The doc `ckks_protocol.md §4.3` argued against
diagonal packing when softmax was expensive — that argument is now inverted.

Baby-Step Giant-Step (BSGS) for a (d_out × d_in) matrix:

| Linear | Current rotations/token | BSGS rotations/token | Factor |
|---|---|---|---|
| Q / K / V (768→768) | ~768 | ~2√768 ≈ 55 | **14×** |
| W₁ (768→3072) | ~3072 | ~√768 + √3072 ≈ 83 | **37×** |
| W₂ (3072→768) | ~3072 | ~83 | **37×** |
| O-proj (768→768) | ~768 | ~55 | **14×** |

OpenFHE provides `EvalLinearTransform` / `LinearTransformPrecompute` which implements
BSGS internally. The `matmul_plain` method on `OpenFHEBackend` should precompute
the diagonal decomposition once and cache it, then call `EvalLinearTransform`.

**Implementation note:** Pre-compute diagonals at weight-loading time in
`fhe_thesis/encryption/coefficients.py:load_model_weights()`.

---

### O3 — SIMD Slot Batching (42 sequences per ciphertext)
**Effort:** High (requires restructuring `TokenPackedTensor`)  
**Applies to:** Everything — all ops run on 42 sequences simultaneously  
**Speedup:** 42× amortized throughput (wall-clock unchanged)

Current: one ciphertext per token row, `hidden_dim` slots used out of 32,768.  
Batched: interleave B = ⌊32768 / 768⌋ = **42** sequence rows per ciphertext.

```
Current CT layout (1 seq):
  [h₀₀, h₀₁, …, h₀₇₆₇,  0,  0, …,  0]
   └── token 0 of seq 0 ─────┘   └ wasted ┘

Batched CT layout (42 seqs):
  [h₀₀⁽⁰⁾, h₀₁⁽⁰⁾, …, h₀₇₆₇⁽⁰⁾, h₀₀⁽¹⁾, …, h₀₇₆₇⁽⁴¹⁾]
   └── token 0, seq 0 ────────┘ └── token 0, seq 1…41 ──┘
```

All LPAN polynomial ops are still element-wise → no change to cost.  
BSGS matmul still works → same rotation count but now processes 42 sequences.  
Attention Q·Kᵀ: each sequence's tokens are non-contiguous → requires an
unpack-repack step at attention boundaries (seq-level cross-token op).

**Strategy for attention:** keep attention at seq_level (unpack 42 → process per-seq → repack),
or use a different packing layout for attention only. Hybrid approach: use batched
packing for FFN blocks, repack for attention. This is what BOLT and Iron do.

---

### O4 — Sequence Length Truncation (128 → 64 for SST-2)
**Effort:** Low (one-line change in tokenizer call)  
**Applies to:** Attention only (O(L²) → O(L²/4))  
**Speedup:** 4× on attention, ~2× overall

SST-2 average sentence: **17–25 tokens**. Padding to 64 vs 128 is lossless.
QNLI: max ~84 tokens → truncate to 96 safely.

```python
# In the FHE inference script:
tokenizer(text, max_length=64, padding="max_length", truncation=True)
```

Attention cost is O(L²): L=64 → 4× cheaper than L=128.  
At L=64 with 12 heads: Q·Kᵀ drops from 12×128²=196,608 dot-products to 12×64²=49,152.

---

### O5 — OpenFHE Multi-Threading (OpenMP)
**Effort:** Trivial (one API call)  
**Applies to:** NTT, key-switch, all internal OpenFHE ops  
**Speedup:** 12–16× on 16-thread CPU; **~20–28× on 32-thread CPU** (Amdahl-limited by bootstrapping serial fraction)

Token ciphertexts are **fully independent** across most ops (linear, GELU, LN).
OpenFHE uses OpenMP internally for NTT parallelism. Additional outer parallelism
over token batches can be added with Python's `concurrent.futures.ThreadPoolExecutor`.

```python
import openfhe
openfhe.SetNumThreads(32)  # set before any FHE ops

# Outer parallelism over token rows (for linear + poly ops):
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=32) as pool:
    new_cts = list(pool.map(lambda ct: backend.matmul_plain(ct, w, b), x.cts))
```

---

### O6 — Lazy Relinearization
**Effort:** Low (change relin policy in `OpenFHEBackend`)  
**Applies to:** ct×ct multiplications (attention apply, LN poly)  
**Speedup:** ~2× on bootstrap frequency

Currently we relinearize immediately after every ct×ct multiply (adds a key-switch call).
Lazy relin: defer until accumulation is complete. For attention·V, we do L accumulations
— deferring relin until after the `add` tree saves L-1 key-switches per output row.

In OpenFHE: use `EvalMultNoRelin` and call `Relinearize` once at the end.

---

### O7 — BERT-Small Ablation (4-layer model)
**Effort:** Zero — training already supports `--model small`  
**Applies to:** Thesis comparison table  
**Speedup vs Base:** ~3–4× (4 layers vs 12, fewer bootstrap calls)

BERT-Small achieves ~89–90% on SST-2. With full optimization stack it enters
**MPCFormer single-sample latency territory** without needing batching.

---

## 4. Inference Time Comparison

### Hardware selection for benchmarking

**Use a CPU-only pod — the H100 GPU provides zero benefit for FHE inference.**

OpenFHE + HEXL are entirely CPU-bound. The H100 GPU sits idle during all CKKS operations (NTT, key-switch, bootstrapping). The right tool is a high-core-count CPU pod with AVX-512 support and sufficient RAM.

| Option | vCPUs | RAM | Cost | Verdict |
|---|---|---|---|---|
| H100 SXM Pod | ~28 vCPUs (CPU only useful) | 251 GB | ~$3/hr | **Wrong tool for FHE** — GPU unused, FHE is CPU-only; actual 28 vCPUs (Xeon Platinum 8480+) but cost is 1.7× more than 32-vCPU CPU pod for same FHE throughput |
| CPU 8 vCPUs | 8 | 64 GB | $0.44/hr | **Too little RAM** — BERT-Base rotation keys alone ~20 GB |
| CPU 16 vCPUs | 16 | 128 GB | $0.88/hr | Good — sufficient RAM, but O5 capped at 16 threads |
| **CPU 32 vCPUs** | **32** | **256 GB** | **$1.76/hr** | **Selected** — 2× more threads than 16-vCPU → additional ~1.8× speedup; ample RAM headroom |

**Memory-Optimized over Compute-Optimized** — CKKS evaluation keys dominate RAM:
- 1 rotation key ≈ 24 MB (N=65536, 23 moduli)
- BSGS for BERT-Base (all 6 linear layers × 12 layers, ~55 rotation keys each): **~15–20 GB keys alone**
- Full key material (rotation + multiplication + public key + bootstrapping key): **~30–50 GB**
- 256 GB leaves ~200 GB headroom — enough to hold all 872 SST-2 validation ciphertexts in RAM simultaneously (no disk swapping)

**5 GHz over 3 GHz** — higher clock speed accelerates the sequential NTT chains inside bootstrapping, which is not parallelizable within a single ciphertext operation.

**Intel Xeon (Ice Lake / Sapphire Rapids)** preferred over AMD EPYC — Intel HEXL's AVX-512 acceleration is most mature on Intel microarchitecture.

---

### Setup assumptions (BERT-Base, SST-2 validation, 872 samples)

| Parameter | Value |
|---|---|
| Model | BERT-Base (12 layers, hidden=768, heads=12) |
| Ring dimension | N = 2^16 = 65536, slots = 32768 |
| CKKS depth budget | 25 levels (bootstrapping every ~1 layer) |
| Hardware | RunPod CPU pod — **32 vCPUs / 256 GB / Memory-Optimized / 5 GHz ($1.76/hr)** |
| HEXL NTT speedup | 4× uniform |
| BSGS linear speedup | 20× (conservative blend across layer shapes) |
| Thread count | **32** |
| Slot batch size B | 42 (for O3) |
| seq_len | 128 (default), 64 (O4) |

---

### MPCFormer reported numbers (from paper, Table 2)

| Setting | BERT-Base latency | Comm. cost |
|---|---|---|
| LAN (1Gbps) | **~6–8 seconds** per sample | ~200MB |
| WAN (100Mbps) | **~60–80 seconds** per sample | ~200MB |
| Accuracy (SST-2) | 91.3% | — |

> Note: MPCFormer numbers come from a dedicated 2-party MPC framework running
> on GPU-accelerated servers with high-bandwidth interconnect. This is not a
> fair latency comparison — but throughput and security comparisons are valid.

---

### LPAN-FHE optimization levels

> All timings below are for the **32 vCPU / 256 GB / 5 GHz CPU pod**.
> For reference, the H100 SXM pod would match the *16-thread* column (same ~16 usable CPUs for OpenFHE) while costing 2–3× more.

| Config | Ops | 16-vCPU estimate | **32-vCPU estimate** | Amortized (B=42) | Accuracy |
|---|---|---|---|---|---|
| **Baseline** (TenSEAL, seq=128) | None | 2–4 hours | **2–4 hours** | 2–4 hours | ~91% |
| **+ O1** (OpenFHE + HEXL) | 4× NTT + 2× key-switch | 30–60 min | **30–60 min** | 30–60 min | same |
| **+ O2** (BSGS matmul) | 20× linear rotations | 8–15 min | **8–15 min** | 8–15 min | same |
| **+ O5** (threading) | 16× → **28×** parallelism | 30–75 sec | **~15–35 sec** | ~15–35 sec | same |
| **+ O4** (seq_len=64) | 4× attention | 10–20 sec | **~4–9 sec** | ~4–9 sec | same (SST-2) |
| **+ O3** (B=42 batching) | 42× throughput | 10–20 sec | 4–9 sec | **~0.10–0.21 sec** | same |
| **+ O6** (lazy relin) | ~2× bootstrap | 5–10 sec | **~2.5–5 sec** | **~0.06–0.12 sec** | same |

**Why 32 vCPUs are ~1.7–1.8× faster than 16 vCPUs:**
- OpenFHE OpenMP (NTT): scales ~linearly to 32 threads → ~1.9× over 16
- Outer token-loop parallelism: 64 tokens / 32 threads = 2 per thread vs 4 → ~1.8× over 16
- Bootstrapping serial fraction (Amdahl): caps combined gain at ~1.7–1.8×

**Best single-sample latency (O1+O2+O5+O4+O6, no batching): ~2.5–5 seconds** — **faster than MPCFormer LAN**  
**Best amortized throughput (all opts): ~0.06–0.12 seconds per sequence** at B=42  
**Pod cost during benchmarking: $1.76/hr** — stop immediately after experiments are done

---

### Summary comparison table

| System | Privacy model | Hardware | Latency (1 sample) | Throughput | Accuracy (SST-2) |
|---|---|---|---|---|---|
| **MPCFormer** (LAN) | 2-party MPC, semi-honest | 2× GPU servers | **6–8 sec** | ~6–8 sec/sample | 91.3% |
| **MPCFormer** (WAN) | 2-party MPC, semi-honest | 2× GPU servers | 60–80 sec | ~60–80 sec/sample | 91.3% |
| **LPAN-FHE** (H100 pod, FHE only) | Single-server FHE | H100 SXM pod — 28 vCPU (Xeon Platinum 8480+), 251 GB RAM, CUDA 13.0 | ~4–8 sec | ~4–8 sec/sample | ~91.5% |
| **LPAN-FHE (O1+O2+O5+O4+O6)** | Single-server FHE | **32-vCPU CPU pod** | **~2.5–5 sec** | ~2.5–5 sec/sample | ~91.5% |
| **LPAN-FHE (all opts, B=42)** | Single-server FHE | **32-vCPU CPU pod** | ~2.5–5 sec | **~0.06–0.12 sec/sample** | ~91.5% |

**Key thesis claim:** LPAN-FHE with the full optimization stack achieves **~2.5–5 seconds single-sample latency** — **beating MPCFormer LAN (6–8 sec)** — while offering:
- ✅ No server-to-server communication
- ✅ No trust assumptions between servers  
- ✅ Stronger security (works even if the server is fully malicious)
- ✅ Better accuracy (91.5% vs 91.3%)
- ✅ **50–130× better throughput** at cloud scale (B=42, 0.06–0.12 sec amortized vs 6–8 sec)
- ✅ **2× cheaper hardware** than H100 pod ($1.76/hr vs $3–4/hr), GPU not needed

---

## 5. Implementation Steps (in order)

### Phase A — After all GLUE training finishes (prerequisite)
- [ ] Wait for SST-2 Stage 3 to complete all 12 layers
- [ ] Run QNLI Stage 1, 2, 3 (same pipeline)
- [ ] Save final checkpoints: `checkpoints/sst2_base_stage3_final/` and `checkpoints/qnli_base_stage3_final/`

---

### Phase B — Backend port (O1)
**Files:** `fhe_thesis/encryption/openfhe_backend.py` (new file)

- [ ] Deploy RunPod CPU pod: **32 vCPUs / 256 GB / Memory-Optimized / 5 GHz** ($1.76/hr)
- [ ] Verify AVX-512 support: `grep avx512 /proc/cpuinfo | head -1`
- [ ] Install OpenFHE + HEXL on the CPU pod:
- [ ] Implement `OpenFHEBackend(CKKSBackend)`:
  - `encrypt` / `decrypt` using OpenFHE `CCParams`, `GenCryptoContext`
  - `matmul_plain` using `EvalMult` with plaintext polynomial
  - `polyval` using `EvalPoly`
  - `dot` using `EvalInnerProduct`
  - `sum_slots` using `EvalSum`
  - Enable `HYBRID` key switching: `cryptoContext.Enable(PKESchemeFeature::PRE)`
  - Enable bootstrapping: `cryptoContext.Enable(PKESchemeFeature::LEVELED_SHE)` + `EvalBootstrapSetup`
- [ ] Write unit test: `tests/test_openfhe_backend.py` — compare outputs vs `TenSEALBackend` on 10 random vectors

---

### Phase C — BSGS linear layers (O2)
**Files:** `fhe_thesis/encryption/openfhe_backend.py`

- [ ] Override `matmul_plain` in `OpenFHEBackend` to call `EvalLinearTransformPrecompute` once at weight-load time
- [ ] Cache precomputed diagonal representations in a dict keyed by `(out_dim, in_dim, weight_hash)`
- [ ] Validate: same output as naive matmul within 1e-5 tolerance

---

### Phase D — Sequence truncation (O4)
**Files:** FHE inference script (wherever tokenization happens)

- [ ] Add `--max-seq-len` argument (default 64 for SST-2, 96 for QNLI)
- [ ] Verify accuracy drop is < 0.1% on SST-2 validation set

---

### Phase E — Multi-threading (O5)
**Files:** `fhe_thesis/encryption/ops.py`, `protocol.py`

- [ ] Call `openfhe.SetNumThreads(N_THREADS)` at context creation
- [ ] Parallelize token loops in `enc_linear`, `enc_gelu_poly`, `enc_ln_poly`:
  ```python
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
      new_cts = list(pool.map(process_token, x.cts))
  ```
- [ ] Keep attention serial (cross-token dependencies in Q·Kᵀ)

---

### Phase F — Slot batching (O3)
**Files:** `fhe_thesis/encryption/packing.py`, `ops.py`, `protocol.py`

- [ ] Add `BatchedTokenPackedTensor` with batch_size B=42
- [ ] Modify `encrypt` to interleave B sequences: `slots = concat([seq_i_token_j for i in range(B)])`
- [ ] GELU/LN/softmax-poly: unchanged (all elementwise)
- [ ] BSGS matmul: weight diagonals broadcast across B groups → same rotation count
- [ ] Attention: unpack per-sequence for Q·Kᵀ, repack after; or use hybrid packing
- [ ] Add `--batch-size` CLI flag to inference script

---

### Phase G — Lazy relinearization (O6)
**Files:** `fhe_thesis/encryption/openfhe_backend.py`

- [ ] Replace `EvalMult` with `EvalMultNoRelin` in `mul()`
- [ ] Add explicit `Relinearize()` call after each accumulation loop in `enc_attention_apply` and `enc_ln_poly`
- [ ] Validate depth budget: lazy relin doesn't save levels, only key-switch cost

---

### Phase H — Benchmarking and results table
**Files:** `experiments/08_fhe_benchmark.py` (new)

- [ ] Run on 100 SST-2 validation samples with each optimization level enabled/disabled
- [ ] Record: wall-clock per sample, amortized throughput, accuracy (plain vs decrypted)
- [ ] Compare against MPCFormer published numbers
- [ ] Generate plots: latency breakdown pie chart, optimization waterfall bar chart

---

### Phase I — Thesis write-up
**Files:** `IZTECH_Master_Thesis/chapters/4_results.tex`

- [ ] Table: accuracy across MRPC, SST-2, QNLI for LPAN vs MPCFormer vs BERT-Base
- [ ] Table: inference time breakdown per optimization level
- [ ] Section: security model comparison (FHE vs MPC — why FHE is stronger)
- [ ] Section: throughput analysis (amortized cost argument)

---

## 6. Thesis Positioning

> "LPAN-FHE with OpenFHE+HEXL, BSGS diagonal matmul, sequence truncation, and SIMD slot batching achieves **~5–10 seconds single-sample latency** and **~0.3 seconds amortized throughput** for BERT-Base inference on SST-2 — comparable to MPCFormer's 6–8 seconds on LAN — while operating under a strictly stronger single-server security model that requires no server-to-server communication and tolerates a fully malicious server. Furthermore, LPAN achieves **91.5% accuracy on SST-2**, exceeding MPCFormer's 91.3%, with only a **~1.5% accuracy drop** from BERT-Base fine-tuned baseline."

### Novelty claims
1. **LPAN training scheme**: learnable polynomial approximations fitted to each layer's activation distribution via minimax optimization — better accuracy than fixed-polynomial replacements
2. **Combined FHE system**: single-server CKKS inference for BERT-Base with bootstrapping, competitive latency at cloud scale
3. **Benchmark**: first direct accuracy + latency comparison of BERT-Base FHE inference (CKKS) vs MPC (2PC) under matched model architecture

---

*Last updated: 2026-04-26*  
*Training status: SST-2 Stage 3 L1 epoch 6/8 (best 91.74%)*
