# THEF — Privacy-Preserving Framework for Transformer Inference (HE + TEE)

**Citation**: Li, Liao, Yu, Zhang. *IEEE TrustCom 2024*. SJTU + Nanhu Lab.

**One-line summary**: Hybrid HE (CKKS) + Intel SGX TEE — uses TEE for non-linears to bypass FHE polynomial approximation cost.

## Core contribution

- HE handles linear ops (matmul) on the cloud.
- Sensitive non-linear computations are decrypted inside SGX, evaluated in plaintext within the enclave, then re-encrypted.
- Optimizations: expanding-encoding matmul, batch acceleration, adaptive scheme.

## Key numbers

- Reports ≥5× less communication and ≥10× faster runtime end-to-end vs. unspecified SOTA on BERT-base.
- Accuracy "close to plaintext".

## Threat model

**Trusts Intel SGX**. SGX has had multiple side-channel breaks (Foreshadow, ÆPIC Leak, MDS family). **Strictly weaker** than ours — adds a hardware trust assumption beyond the cryptographic one.

## Relevance to HyPER-LPAN

- A **counter-example** to our pure-FHE approach: shows what happens if you compromise on threat model.
- Useful as a "weaker-threat-model baseline" in our threat-model comparison table.
- Demonstrates that the FHE non-linearity bottleneck is so severe that researchers reach for hardware enclaves to escape it — strengthens the motivation for our LPAN polynomial design.

## Direct comparison

- "THEF [Li et al., TrustCom'24] sidesteps the FHE non-linearity bottleneck by evaluating non-linears in an Intel SGX enclave, accepting the well-documented side-channel risks of TEE hardware. HyPER-LPAN keeps the pure-FHE threat model and instead reduces non-linearity cost architecturally, via the layer-wise primitive selector."
