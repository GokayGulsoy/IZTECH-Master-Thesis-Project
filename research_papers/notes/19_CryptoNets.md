# CryptoNets — Applying Neural Networks to Encrypted Data

**Citation**: Dowlin, Gilad-Bachrach, Laine, Lauter, Naehrig, Wernsing. *ICML 2016*. Microsoft Research + Princeton.

**One-line summary**: First demonstration of neural network inference under HE — small CNN on **MNIST** at **99% accuracy**, **~58 982 predictions/hour** throughput, 250 s latency per batch (4096 inputs).

## Core contribution

- Replace ReLU with the **square activation `f(x) = x²`** — the lowest-degree polynomial that introduces non-linearity.
- Replace max-pooling with average-pooling (no comparison needed).
- Use **batched ciphertexts** to amortize cost: one ciphertext = 4096 simultaneous predictions.
- Demonstrates that despite homomorphic encryption's reputation for impracticality, a careful network-and-encryption co-design enables real workloads.

## Key numbers

- MNIST: **99% accuracy**, 58 982 predictions/hour, 250 s/batch latency.
- One ciphertext per channel; SIMD across the 4096-element batch dimension.

## Limitations

- Only works for **shallow** networks: x² activation amplifies values exponentially with depth (gradient explosion in training, magnitude blow-up in inference).
- ReLU → x² loses ~1–2% accuracy on simple tasks; much more on deep networks.

## Relevance to HyPER-LPAN

- The **historical baseline** for FHE-friendly polynomial activations.
- Cite in §2.3 (non-linearity bottleneck) — every paper since CryptoNets has been chasing better polynomial activation choices.
- Justifies our **QuadAttention primitive** (which uses `x²`-style polynomial softmax replacement): the simplest polynomial that still preserves nonlinear interaction.
- Demonstrates the **batched-ciphertext** strategy we generalize to token packing.

## Direct citation use

- "Polynomial-activation neural networks date back to CryptoNets [Gilad-Bachrach et al., ICML'16], which replaced ReLU with `x²` to enable a small CNN on MNIST under HE. Subsequent work (LoLa, Faster CryptoNets, AESPA, AutoFHE) has refined polynomial activations for deeper networks; HyPER-LPAN extends this line to transformer attention via its three-primitive design."
