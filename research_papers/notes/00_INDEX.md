# Research Papers — Key-Takeaway Notes

Reference notes for the local paper corpus in `../` plus the external
positioning papers added during the May 2026 literature re-pass. Each note
follows the same template:

- **Citation** (authors, year, venue)
- **One-line summary**
- **Core contribution**
- **Key technical mechanism** (with concrete numbers)
- **Threat model / setting** (when applicable)
- **Relevance to Synthesizer-LPAN**
- **Direct comparison points / how to cite**

## Categories

### A. Closest competitors (private transformer inference)
- [01_NEXUS.md](01_NEXUS.md) — closest baseline, same threat model
- [02_BOLT.md](02_BOLT.md) — SP'24 MPC SOTA
- [03_Iron.md](03_Iron.md) — NeurIPS'22 hybrid HE+2PC
- [04_MPCFormer.md](04_MPCFormer.md) — ICLR'23 KD + MPC approximation
- [05_THE-X.md](05_THE-X.md) — ACL'22 HE-only with weakened threat model
- [06_THEF.md](06_THEF.md) — TrustCom'24 HE+TEE hybrid (weaker threat model)
- [07_PrivacyRestore.md](07_PrivacyRestore.md) — DP+plaintext-server hybrid

### B. CKKS / FHE foundations
- [08_CKKS_2017.md](08_CKKS_2017.md) — original CKKS
- [09_HE_Standard_2018.md](09_HE_Standard_2018.md) — parameter recommendations
- [10_HighPrecision_Bootstrapping.md](10_HighPrecision_Bootstrapping.md) — high-precision RNS-CKKS bootstrap
- [11_Fast_Amortized_Bootstrap.md](11_Fast_Amortized_Bootstrap.md) — sub-linear amortized BS
- [12_Efficient_Homomorphic_Comparison.md](12_Efficient_Homomorphic_Comparison.md) — sign/comparison polynomials
- [13_PEGASUS.md](13_PEGASUS.md) — CKKS↔FHEW switching
- [14_Thesis_Background_Blueprint.md](14_Thesis_Background_Blueprint.md) — internal blueprint document

### C. Architecture / training prior art (Ext W and Ext 3)
- [15_PoWER-BERT.md](15_PoWER-BERT.md) — direct prior art for Ext W (token elimination)
- [16_Length_Adaptive.md](16_Length_Adaptive.md) — extends PoWER-BERT
- [17_DARTS.md](17_DARTS.md) — differentiable NAS, contrast with our MCKP-DP
- [18_Universal_Transformers.md](18_Universal_Transformers.md) — adaptive computation prior art

### D. CNN baselines
- [19_CryptoNets.md](19_CryptoNets.md) — first CNN-on-FHE (x² activation)
- [20_AutoFHE.md](20_AutoFHE.md) — mixed-degree polynomial NAS for ResNets

### E. Peripheral
- [21_Sign-GD.md](21_Sign-GD.md) — distributed optimization theory (1-paragraph)
- 22 / Gantt chart — project planning, no research note

### F. Added in May 2026: mandatory architecture / positioning papers
- [22_Synthesizer.md](22_Synthesizer.md) — architectural parent of Synthesizer-LPAN
- [23_Cerium.md](23_Cerium.md) — concurrent compiler/runtime SOTA, changes our claim surface
- [24_Linformer.md](24_Linformer.md) — low-rank sequence projection baseline
- [25_Performer.md](25_Performer.md) — kernelized linear-attention baseline
- [26_Nystromformer.md](26_Nystromformer.md) — landmark-based low-rank attention baseline
- [27_AFT.md](27_AFT.md) — attention-free but still query-conditioned mixer
- [28_FNet.md](28_FNet.md) — fixed Fourier token mixer, strongest alternative future branch

### G. Synthesis outputs
- [29_RELATED_WORK_MATRIX.md](29_RELATED_WORK_MATRIX.md) — thesis-safe claim matrix and comparison language

## How to use these notes

When citing in `IZTECH_Master_Thesis/` LaTeX, pull the **exact numbers** and **threat-model
position** from the relevant note rather than re-reading the PDF. The "Direct comparison
points" section gives ready-to-use sentences for related-work / discussion. For current,
claim-safe thesis framing after Cerium, start from [29_RELATED_WORK_MATRIX.md](29_RELATED_WORK_MATRIX.md).
