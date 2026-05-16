# 05 — Reproducing Results

All commands assume an Ubuntu 22.04 + CUDA 12.x + H100 (or compatible
NVIDIA GPU) machine, with a Python venv:

```bash
python3.10 -m venv fhe_venv
source fhe_venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## 1. Build vendored HEonGPU + Python bindings

HEonGPU is vendored at [`third_party/HEonGPU/`](../third_party/HEonGPU/);
the upstream commit is pinned in
[`third_party/HEonGPU.commit`](../third_party/HEonGPU.commit).

One-shot build script (tested on a stock RunPod H100 instance):

```bash
bash scripts/setup_pod_gpu.sh
```

This:

1. Installs system deps (cmake, ninja, libgmp-dev, libntl-dev, …).
2. Creates a persistent RunPod venv at `/workspace/fhe_venv`.
3. Builds HEonGPU under `/workspace/HEonGPU/build/`.
4. Builds the pybind11 bindings in
   [`fhe_thesis/encryption/heongpu_bindings/`](../fhe_thesis/encryption/heongpu_bindings/).
5. Leaves the pod ready for future same-pod resumes via:

```bash
source scripts/activate_pod_env.sh
```

On RunPod, `/workspace` persists across stop/start of the same pod. It does
not migrate to a different pod unless you attach a RunPod network volume or
export the data elsewhere.

## 2. Smoke-test the HEonGPU backend

```bash
python scripts/smoke_heongpu_backend.py
```

Confirms: backend constructs, keys generate, simple add / mul_plain /
rotate work, decrypt round-trips a small ciphertext.

## 3. Correctness test (Synthesizer-LPAN vs plaintext)

```bash
python scripts/test_synthesizer_lpan_correctness.py
```

Runs the naive and BSGS Synthesizer attention on a small toy $A$, $V$
and asserts:

- naive output ≈ plaintext $A V$ within CKKS noise tolerance
- BSGS output ≈ naive output bit-for-bit (no extra approximation)
- LayerNorm cubic invsqrt error within tolerated minimax bound
- per-layer end-to-end output matches plaintext within 1e-3 relative

## 4. Headline benchmark

```bash
python scripts/bench_L128_synthesizer_lpan.py
```

Default config: $L=128$, BATCH=16, chain=22, BSGS enabled, 12 layers.
Expected output on a single H100 SXM5:

```
[bench] BATCH=16 chain=22 L=128 layers=12  BSGS=True
[bench]   per-layer wall-time:  ~5.07 s
[bench]   12-layer total:        60.9 s
[bench]   per-sample (batch=16): ~3.8 s
```

The script also supports sweeps via env vars:

```bash
BATCHES=4,8,16  CHAINS=21,22,24  python scripts/bench_L128_synthesizer_lpan.py
```

This dumps a JSON ladder to `results/bench_synthesizer_lpan.json`
matching the wall-time ladder in [04_OPTIMIZATIONS.md](04_OPTIMIZATIONS.md).

## 5. Stage-1 to Stage-4 training + bench export

Build the Stage-1 to Stage-3 LPAN teacher chain first:

```bash
python -m fhe_thesis.training.run_staged_lpan \
  --model base \
  --task sst2 \
  --stage all
```

This writes the clean teacher chain to:

- `results/synthesizer_lpan/sst2/base/baseline/`
- `results/synthesizer_lpan/sst2/base/staged_lpan_s1_gelu/`
- `results/synthesizer_lpan/sst2/base/staged_lpan_s2_softmax/`
- `results/synthesizer_lpan/sst2/base/staged_lpan_s3_ln_kd/`

Train the Synthesizer-attention Stage-4 model from an existing Stage-3 LPAN
checkpoint:

```bash
python -m fhe_thesis.training.run_synth_lpan \
  --model base \
  --task sst2 \
  --stage 4
```

Export the learned per-layer Synthesizer patterns and polynomial coefficients
to the JSON format consumed by the benchmark script:

```bash
python -m fhe_thesis.training.export_synth_lpan \
  --checkpoint-dir results/synthesizer_lpan/sst2/base/best_model
```

Then bench the exported layer payload:

```bash
python scripts/bench_L128_synthesizer_lpan.py \
    --checkpoint results/synthesizer_lpan/sst2/base/bench_checkpoint.json \
    --layer 0
```

## Common pitfalls

- **`ModuleNotFoundError: fhe_thesis`** → run commands from the repo root
  or install the project with `pip install -e .`
- **`HEonGPU import fails`** → re-run `scripts/setup_pod_gpu.sh`;
  ensure CUDA driver matches HEonGPU's compiled CUDA toolkit version.
- **Out-of-memory at BATCH=16** → drop to BATCH=8 (still ~3.6× speedup)
  or chain=24.
- **Numerical mismatch with plaintext** → check polynomial input range
  hasn't drifted outside the fitted $[-4, 4]$ for GELU; LN cubic
  invsqrt range may also need refit per task.
- **Chain=21 fails** → expected; LN's cubic invsqrt needs 6 levels +
  buffer. Use chain=22.

## End-to-end sanity check (one paste)

```bash
source fhe_venv/bin/activate && \
  python -c "
import importlib
mods = [
    'fhe_thesis.config',
    'fhe_thesis.tasks',
    'fhe_thesis.encryption.backend',
    'fhe_thesis.encryption.heongpu_backend',
    'fhe_thesis.encryption.colmajor',
    'fhe_thesis.encryption.multi',
    'fhe_thesis.encryption.linear',
    'fhe_thesis.encryption.layernorm',
    'fhe_thesis.encryption.attention',
    'fhe_thesis.poly.approximation',
    'fhe_thesis.poly.chebyshev',
    'fhe_thesis.models.activations',
    'fhe_thesis.models.profiling',
    'fhe_thesis.models.replacement',
    'fhe_thesis.models.backbone',
    'fhe_thesis.training.checkpoints',
    'fhe_thesis.training.run_staged_lpan',
    'fhe_thesis.training.run_synth_lpan',
    'fhe_thesis.training.export_synth_lpan',
    'fhe_thesis.training.trainer',
]
for m in mods:
    importlib.import_module(m)
    print('OK', m)
print('all imports OK')
"
```
