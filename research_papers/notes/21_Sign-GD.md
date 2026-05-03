# Sign-GD — Distributed Optimization (peripheral)

**One-line summary**: Theoretical paper on sign-based gradient descent for distributed optimization. **Peripheral** to HyPER-LPAN — included in the corpus but not directly used.

## Why it's in the folder

Likely included for reference on:
- Sign function evaluation theory (related to FHE comparison primitives).
- Communication-efficient distributed training (relevant if we ever train **under** FHE rather than just inferring).

## Relevance to HyPER-LPAN

**Not directly used.** We are doing inference, not training, under FHE. Sign-based gradient descent's analysis of bias/variance trade-offs in 1-bit gradients is interesting but does not change our methodology.

If we ever extend HyPER-LPAN to **secure fine-tuning**, this paper becomes relevant — until then, no citation needed.

## Note for thesis

Skip in related-work section unless we add a future-work paragraph on private fine-tuning.
