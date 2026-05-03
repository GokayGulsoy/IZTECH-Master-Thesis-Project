# AutoFHE — Automated Adaption of CNNs for Efficient Evaluation over FHE

**Citation**: Ao, Boddeti. *USENIX Security 2024*. Michigan State University.

**One-line summary**: NAS framework that **automatically assigns layer-wise mixed-degree polynomial activations** to a CNN and jointly optimizes the **placement of bootstrap operations**, producing a Pareto front of accuracy-vs-latency under RNS-CKKS.

## Core contribution

- **Layer-wise mixed-degree polynomials**: each ReLU replaced by a polynomial of independently chosen degree. Different layers have different sensitivity to approximation error.
- **Joint optimization**: polynomial choice **and** bootstrap placement are searched together via multi-objective evolutionary algorithm.
- **Hybrid approximation + training**: inherit weights from pretrained ReLU network, then fine-tune with mixed-degree polynomials.
- Diverse Pareto solutions (no single weighted-sum loss).

## Key numbers (CIFAR-10/100, RNS-CKKS)

- **1.32 – 1.8× faster** than methods using uniform high-degree polynomials (e.g., MPCNN minimax-13).
- **+2.56% accuracy** over methods using uniform low-degree polynomials.
- **103× faster + 3.46% more accurate** than CNNs under TFHE.

## What it improves over prior art

- **MPCNN** (state-of-the-art before AutoFHE): uses a uniform high-precision (α=13) polynomial → 70%+ of inference time spent on bootstrapping; needs a custom bootstrap schedule per CNN.
- **SAFENet**: only 2 polynomial choices (degree 2 or 3), uses scalarized weighted-sum.
- **AESPA**: polynomial-basis normalization but uniform degree.

## Relevance to HyPER-LPAN

- **Most directly analogous prior art for our composition selector** (in the CNN/FHE world).
- Same insight: **layer-wise heterogeneous primitive selection** is better than uniform replacement.
- Same approach: search over the trade-off front rather than picking one configuration.
- Differences from our MCKP-DP:
  - AutoFHE: continuous polynomial-degree search via evolutionary multi-objective (NSGA-II family). Discrete primitives in our case.
  - AutoFHE: **search includes weight fine-tuning cost** — expensive.
  - HyPER-LPAN: closed-form drift estimator + exact DP — milliseconds.
- AutoFHE targets **CNN ResNets**, we target **transformers**.
- AutoFHE's **bootstrap-placement co-optimization** is something we should consider adding (currently we use a fixed bootstrap budget [3,3]).

## Direct citation use

- "AutoFHE [Ao & Boddeti, USENIX Sec'24] is the closest prior work in spirit: layer-wise mixed-degree polynomial activations selected via multi-objective NAS for FHE-friendly ResNets. HyPER-LPAN brings this idea to transformers and replaces the evolutionary search with an exact dynamic-programming solution to the multiple-choice knapsack — feasible because our per-layer primitive set is small and discrete."

## Open question

- AutoFHE's joint bootstrap-placement optimization is something we should evaluate post-MRPC. Currently we fix the bootstrap schedule.
