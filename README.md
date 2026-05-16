# Synthesizer-LPAN

> **Single-GPU sub-100-second FHE BERT inference via architectural
> elimination of Q, K, and softmax.**

İzmir Institute of Technology (İYTE) — M.Sc. Thesis
**Author:** Gökay Gülsoy · Department of Computer Engineering

| Configuration | 12-layer fwd, L=128, single H100 | Speedup |
|---|---|---|
| Honest LPAN baseline (full softmax-poly) | 833 s | 1.00× |
| **Synthesizer-LPAN + BSGS, BATCH=16, chain=22** | **60.9 s** | **13.67×** |

## What this is

A privacy-preserving BERT-base inference stack running entirely under
**CKKS Fully Homomorphic Encryption** (HEonGPU CUDA backend). The
core contribution is **Synthesizer-LPAN** — *Synthesizer Learnable
Polynomial Activation Network* — which replaces standard self-attention
with a frozen, learned, plaintext attention pattern $A \in \mathbb{R}^{L \times L}$
(softmax absorbed at training time), eliminating:

- $W_q$, $W_k$ projections,
- the $Q \cdot K^\top$ ciphertext-by-ciphertext multiplication,
- the depth-12 softmax-poly evaluation.

Combined with **BSGS-fused mask × diagonal** ($2L \to 2\sqrt{L}$
rotations), **batched LPAN polynomial evaluation** (BATCH=16), and a
tuned CKKS chain budget (chain=22), this yields a single-GPU
sub-100 s end-to-end coherent FHE BERT result on plain HEonGPU.

## Threat model

**Pure non-interactive FHE.** No MPC handshakes, no TEE, no
mid-circuit decryption. Single round, server never sees plaintext.
Matches the strongest threat model in the literature
(NEXUS / CERIUM); strictly stronger than BOLT / Iron / MPCFormer / THE-X.

## Repository layout

```
fhe_thesis/                  library
├── encryption/              CKKS protocol + HEonGPU backend
│   ├── attention.py         Synthesizer attention (naive + BSGS)
│   ├── linear.py            BSGS / streaming / multi linear projections
│   ├── layernorm.py         cubic-invsqrt LayerNorm
│   ├── colmajor.py          NEXUS column-major slot layout
│   ├── multi.py             multi-ciphertext bundle ops
│   ├── heongpu_backend.py   HEonGPU CUDA wrapper
│   └── heongpu_bindings/    pybind11 sources + build.sh
├── poly/                    Chebyshev / Remez / weighted minimax
├── models/                  PyTorch surgery (replace activations, Synthesizer swap)
└── training/                KD trainer, checkpoint loaders, Stage-4 CLI/export

scripts/                     CLI entry points
├── setup_pod_gpu.sh         one-shot HEonGPU build
├── bench_L128_synthesizer_lpan.py   headline 60.9 s benchmark
└── test_synthesizer_lpan_correctness.py

experiments/                 thin wrappers around production CLIs
├── run_synth_lpan_stage4.py
└── export_synth_lpan.py

third_party/HEonGPU/         vendored CKKS backend (commit pinned)
docs/                        full design docs (start with docs/README.md)
IZTECH_Master_Thesis/        LaTeX thesis source
research_papers/             literature review PDFs
```

## Quick start

```bash
# 1. Create venv (Python 3.10)
python3.10 -m venv fhe_venv && source fhe_venv/bin/activate

# 2. Install PyTorch with the CUDA build that matches your driver
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the project (pulls all other Python deps from pyproject.toml)
pip install -e .

# 4. Build the vendored HEonGPU CUDA backend
bash scripts/setup_pod_gpu.sh

# 5. Smoke-test
python scripts/smoke_heongpu_backend.py

# 6. Run the headline benchmark
python scripts/bench_L128_synthesizer_lpan.py
# → 60.9 s on a single H100 SXM5

# 7. Train Stage-1 to Stage-3 LPAN teacher chain
python -m fhe_thesis.training.run_staged_lpan --model base --task sst2 --stage all

# 8. Train Stage-4 Synthesizer-LPAN from the Stage-3 LPAN checkpoint
python -m fhe_thesis.training.run_synth_lpan --model base --task sst2 --stage 4

# 9. Export a bench checkpoint JSON from the trained Stage-4 model
python -m fhe_thesis.training.export_synth_lpan \
  --checkpoint-dir results/synthesizer_lpan/sst2/base/best_model
```

## Documentation

| File | Topic |
|---|---|
| [docs/README.md](docs/README.md) | Index of all design documents |
| [docs/00_PROJECT_OVERVIEW.md](docs/00_PROJECT_OVERVIEW.md) | Goals, contributions, headline numbers |
| [docs/01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md) | Synthesizer-LPAN math + algorithm |
| [docs/02_FHE_PROTOCOL.md](docs/02_FHE_PROTOCOL.md) | HEonGPU CKKS configuration, depth budget |
| [docs/03_THREAT_MODEL.md](docs/03_THREAT_MODEL.md) | Pure-FHE leakage analysis |
| [docs/04_OPTIMIZATIONS.md](docs/04_OPTIMIZATIONS.md) | BSGS, batching, chain tuning |
| [docs/05_REPRODUCING_RESULTS.md](docs/05_REPRODUCING_RESULTS.md) | Build + bench commands |
| [docs/06_HARDWARE.md](docs/06_HARDWARE.md) | H100 single-GPU target |
| [docs/07_REPO_LAYOUT.md](docs/07_REPO_LAYOUT.md) | Module map |
| [docs/TECHNIQUES_JOURNEY.md](docs/TECHNIQUES_JOURNEY.md) | Full narrative — what was tried, what failed, where we landed |

## Comparison with concurrent work

| System | Hardware | Wall-time | Threat model |
|---|---|---|---|
| **Synthesizer-LPAN (this work)** | **1× H100** | **60.9 s** | pure FHE |
| CERIUM (CMU+NVIDIA, Dec 2025) | 8× B200 | 8.8 s | pure FHE |
| CERIUM | 1× H100 | 36.1 s | pure FHE |
| NEXUS (Crypto'24) | 1× GPU | ~7 s/sample (different setup) | pure FHE |
| BOLT (S&P'24) | GPU + 2PC | 10.9 s | FHE+MPC (weaker) |

CERIUM is a **framework**-level optimization (DSL + compiler + runtime,
multi-GPU). Synthesizer-LPAN is an **architectural** contribution
(eliminates Wq, Wk, Q·Kᵀ, softmax-poly entirely on plain CKKS). The
two are orthogonal and compose.

## Citation

```bibtex
@mastersthesis{gulsoy2027synthesizerlpan,
  author = {Gulsoy, Gokay},
  title  = {{Synthesizer-LPAN: Sub-100-Second Single-GPU FHE BERT
            Inference via Architectural Elimination of Q, K, and Softmax}},
  school = {Izmir Institute of Technology},
  year   = {2027},
}
```

## License

MIT — see [LICENSE](LICENSE).
