# 06 — Hardware

Synthesizer-LPAN targets **single-GPU** FHE inference on an H100. Multi-GPU
sharding is explicitly out of scope (CERIUM owns that axis).

## Three hardware roles

1. **FHE inference** — H100 SXM5 with HEonGPU CUDA backend (the only
   place wall-time numbers are measured).
2. **Plaintext training** — RTX 5070 Ti local or H100 RunPod, for the
   plaintext Synthesizer fitting + LPAN polynomial fine-tuning.
3. **Plaintext analysis** — local laptop for activation profiling,
   polynomial fits, dev-set evaluation.

## Available hardware

| Box | Specs | Where | Cost | Best for |
|---|---|---|---|---|
| **MSI laptop** | RTX 5070 Ti (12 GB), Threadripper-class CPU, 32 GB RAM | local | $0 | iteration, plaintext analysis |
| **RunPod H100 SXM** | H100 SXM5 (80 GB HBM3), 32 vCPU, ~256 GB RAM | cloud | ~$3.5–6/hr | **all FHE inference benchmarks**, full training sweeps |

## H100 single-GPU configuration (the production target)

| Parameter | Value | Why |
|---|---|---|
| GPU | H100 SXM5 | HBM3 bandwidth needed for HEonGPU NTT |
| HBM3 used per inference | ~30–40 GB | 12-layer ciphertext working set at chain=22 |
| Driver | NVIDIA ≥ 535 | matches CUDA 12.x toolkit |
| CUDA toolkit | 12.x | HEonGPU compile target |
| HEonGPU commit | pinned in `third_party/HEonGPU.commit` | reproducibility |
| OS | Ubuntu 22.04 | tested deployment surface |

## Headline numbers on H100

```
12-layer Synthesizer-LPAN forward, L=128, BATCH=16, chain=22:
  Wall-time:                60.9 s
  Per-sample throughput:    ~3.8 s (batch effective)
  Per-layer wall-time:      ~5.07 s
  Setup (keys + encoding):  ~10 s (one-off; cached across queries)
```

## Comparison vs published

| System | Hardware | Wall-time / sample (BERT-base, L=128) | Notes |
|---|---|---|---|
| **Synthesizer-LPAN (this work)** | **1× H100** | **60.9 s batch / ~3.8 s effective** | architectural breakthrough |
| CERIUM (Dec 2025) | 8× B200 (~$200K) | 8.8 s | multi-GPU framework |
| CERIUM | 1× H100 | 36.1 s | plain BERT, no architectural change |
| CERIUM | 1× A100 | 66 s | plain BERT |
| NEXUS (Crypto'24) | 1× GPU | ~7 s/sample (different setup) | pure FHE, plaintext softmax |

CERIUM's 8.8 s requires an 8-GPU datacenter cluster; ours runs on a
single H100. CERIUM's framework optimizations are orthogonal to our
architectural change — they could compose.

## Build / run cost

For one full benchmark run on RunPod H100 SXM:

| Phase | Wall-time | Cash cost |
|---|---|---|
| Pod startup + setup_pod_gpu.sh | ~10 min | ~$0.7 |
| HEonGPU build | ~15 min | ~$1 |
| Smoke test + correctness | ~5 min | ~$0.3 |
| Headline bench (single config) | ~3 min | ~$0.2 |
| Full chain × batch sweep | ~25 min | ~$1.7 |
| **Total per re-validation** | **~1 h** | **~$4** |

Vendored HEonGPU + cached pod image cuts repeat runs to ~5 min total.

## Out-of-scope hardware

- **Multi-GPU**: CERIUM owns this axis. Our single-GPU number is the
  contribution.
- **CPU-only FHE**: too slow for chain=22 + BATCH=16 (would be ~hours).
- **AMD GPUs**: HEonGPU is CUDA-only.
- **TPUs / IPUs**: no FHE library support.

## Storage

- **HEonGPU artifacts**: `third_party/HEonGPU/build/` is gitignored;
  rebuilt by setup script.
- **Bindings**: `fhe_thesis/encryption/heongpu_bindings/build/` is
  gitignored.
- **Benchmark results**: `results/bench_synthesizer_lpan.json` is small
  enough to commit if desired.
- **Future training checkpoints**: `results/synthesizer_lpan/` will be
  gitignored; tarball + transfer manually.
