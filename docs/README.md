# HyPER-LPAN Documentation

Living reference for the HyPER-LPAN FHE Transformer thesis project.
Read these files first when starting a new conversation or reviewing
the design.

| Doc | Purpose |
|---|---|
| [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md) | Thesis goals, target venues, timeline, validated baselines |
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | HyPER-LPAN design: LinearMixing / QuadAttention / LPAN composition |
| [02_FHE_PROTOCOL.md](02_FHE_PROTOCOL.md) | CKKS protocol, depth budget, packing, OpenFHE backend |
| [03_THREAT_MODEL.md](03_THREAT_MODEL.md) | Pure non-interactive FHE; leakage analysis vs BOLT/Iron/NEXUS |
| [04_EXTENSIONS.md](04_EXTENSIONS.md) | The five extensions on `feature/hyper-lpan-extensions` |
| [05_REPRODUCING_RESULTS.md](05_REPRODUCING_RESULTS.md) | End-to-end commands: train, select composition, FHE benchmark |
| [06_HARDWARE.md](06_HARDWARE.md) | Local MSI 5070 Ti vs RunPod H100 vs Pod (32-vCPU Threadripper) |
| [07_REPO_LAYOUT.md](07_REPO_LAYOUT.md) | Module map, branch structure, file conventions |

## Branch state (May 1, 2026)

```
* feature/hyper-lpan-extensions   ← HEAD (5 extensions + cleanup)
  feature/ckks-protocol           ← validated baseline
  main
```

## Validated metrics

| Config | SST-2 | MRPC F1 | Notes |
|---|---|---|---|
| Plain BERT-base FP32 | 92.7 | 88.9 | reference |
| LPAN | 91.28 | 82.35 | full softmax-poly all 12 layers |
| HyPER-LPAN canonical (LM4+Q4+L4) | **90.83** | 82.60 | -0.45 from LPAN; uncompetitive on MRPC |
| HyPER-LPAN + 5 ext (projected) | 91.0–91.5 | 86–88 | selector picks per-task composition |

## Pod latency target

5–7 s/sample on 32-vCPU Threadripper 7960X with AVX-512 + HEXL.
See [06_HARDWARE.md](06_HARDWARE.md) for the full breakdown.
