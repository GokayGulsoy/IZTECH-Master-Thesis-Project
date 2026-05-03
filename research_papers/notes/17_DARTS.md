# DARTS — Differentiable Architecture Search

**Citation**: Liu, Simonyan, Yang. *ICLR 2019*. CMU + DeepMind. Code: github.com/quark0/darts.

**One-line summary**: NAS via **continuous relaxation**: replace discrete operation choice with a softmax-mixture, then optimize architecture and weights jointly with gradient descent in a **bilevel** problem.

## Core contribution

- **Continuous relaxation**: edge `(i,j)` carries a mixture
  `ō(i,j)(x) = Σ_o softmax(α_o^(i,j)) · o(x)`
  where the architecture is encoded by continuous variables `α`.
- **Bilevel optimization**:
  ```
  min_α   L_val(w*(α), α)
  s.t.    w*(α) = argmin_w L_train(w, α)
  ```
- Final architecture: pick `argmax_o α_o` per edge.
- Applicable to both convolutional and recurrent search.

## Key numbers

- CIFAR-10: **2.76% test error** with **3.3M params** — competitive with regularized evolution but **3 orders of magnitude less** compute.
- ImageNet (mobile): 26.7% top-1.
- Penn Treebank: 55.7 perplexity, beating extensively-tuned LSTM.

## Limitations

- **Approximate**: gradient through `argmin_w` is approximated (one-step or higher-order).
- **Search-time accuracy ≠ final accuracy**: rich literature on DARTS instability and pruning bias toward skip-connections (DARTS+, P-DARTS, Fair-DARTS).
- Still requires **full training cost** of the relaxed super-network.

## Relevance to HyPER-LPAN

- DARTS is the canonical citation for **differentiable NAS**.
- Our composition selector (MCKP-DP) is **categorically different**: it is **discrete, exact, and decoupled from training**.
- DARTS searches over a continuous relaxation requiring training; we solve a knapsack over per-layer drift profiles measured from a single forward pass — **~5 ms** vs DARTS's GPU-days.
- Use DARTS as the foil that motivates an **exact-DP alternative** when the per-layer cost has integer structure (which is true for our 3-primitive setting).

## Direct citation use

- "Continuous-relaxation NAS methods such as DARTS [Liu et al., ICLR'19] search architectures via gradient descent on a softmax-mixture super-network, requiring full training of the relaxed model. In contrast, our composition selector exploits the integer structure of FHE primitive costs to solve a multiple-choice knapsack exactly via dynamic programming, reducing the search cost from GPU-days to milliseconds."

## Why exact DP is possible for us

- Only K=3 primitive choices per layer (LM, Quad, LPAN).
- Costs are **integer multiples of a base unit** (after scaling by `cost_scale=10`).
- Per-layer drift `ε_l(k)` has a closed-form estimate (`tr Cov(A)`) from one forward pass.
- → MCKP solvable in `O(L · B · K)` ≈ `O(12 · 100 · 3) = 3600` steps ≈ 5 ms.
