"""Model-agnostic LPAN-FHE protocol orchestration.

Builds an end-to-end encrypted forward pass for any BERT variant in
``fhe_thesis.config.MODEL_REGISTRY``. All shapes (num_layers, hidden,
heads) are read from the model's HuggingFace config, so the same code
path runs on Tiny / Mini / Small / Base.

Public API
----------
* :func:`encrypt_ffn_block`        — single FFN + LN block, one layer
* :func:`encrypt_attention_block`  — one MHA block, one layer
* :func:`encrypt_layer`            — full transformer layer (attn + FFN)
* :func:`encrypt_inference`        — full encoder + classifier head

Each routine takes a plaintext numpy input, runs the encrypted
pipeline under the supplied :class:`CKKSBackend`, decrypts at the end,
and returns the numpy result plus a latency dict.

Threading / seq-len options
---------------------------
All block functions accept an optional ``n_jobs`` parameter (default 1)
which is forwarded to token-parallel ops in ``ops.py`` (O5).

``encrypt_inference`` additionally accepts ``max_seq_len`` (default None)
to truncate the token sequence before encryption (O4). SST-2 average
sentence is 17–25 tokens, so ``max_seq_len=64`` is lossless in practice
while cutting O(L²) attention cost by 4×.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fhe_thesis.config import MODEL_REGISTRY

from .backend import CKKSBackend
from .coefficients import PolyCoeffs, load_coefficients
from .ops import (
    enc_gelu_poly,
    enc_linear,
    enc_ln_poly,
    enc_self_attention,
    enc_self_quad_attention,
)
from .matrix_packing import MatrixPackedTensor, next_pow2
from .ops_matrix import (
    enc_gelu_matrix,
    enc_layernorm_matrix,
    enc_linear_matrix,
    enc_self_attention_matrix,
)
from .packing import TokenPackedTensor


# ──────────────────────────────────────────────────────────────────────
# Weight bundle pulled from a HuggingFace BertModel
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LayerWeights:
    """Plaintext weights for one BERT encoder layer."""

    Wq: np.ndarray
    bq: np.ndarray
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray  # post-attn LN
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray  # post-FFN LN


@dataclass
class LinearMixingLayerWeights:
    """Plaintext weights for one multi-head linear-mixing BERT encoder layer.

    Replaces the Q/K/V/O attention weights with per-head position mixing
    matrices plus an output projection.
    """

    P_weights: np.ndarray   # (num_heads, max_seq_len, max_seq_len) per-head position mixing
    P_biases: np.ndarray    # (num_heads, max_seq_len)
    Wo: np.ndarray          # (hidden, hidden) output projection
    bo: np.ndarray          # (hidden,)
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray    # post-mixing LN
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray    # post-FFN LN
    num_heads: int = 12


@dataclass
class QuadLayerWeights:
    """Plaintext weights for one 2Quad attention BERT encoder layer.

    Same Q/K/V/O projection shapes as standard attention; only the
    softmax-poly is replaced (eliminated) by the squaring + scalar /L
    transform inside the encrypted protocol.
    """

    Wq: np.ndarray
    bq: np.ndarray
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray  # post-attn LN
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray  # post-FFN LN
    num_heads: int = 12


@dataclass
class ModelWeights:
    """All encoder layers + classifier head for a BERT variant."""

    model_key: str
    num_layers: int
    hidden: int
    num_heads: int
    layers: List[LayerWeights] = field(default_factory=list)
    pooler_W: np.ndarray | None = None  # CLS pooler
    pooler_b: np.ndarray | None = None
    cls_W: np.ndarray | None = None  # classifier head
    cls_b: np.ndarray | None = None


# Backbone-prefix table used to generalise state-dict access across
# BERT, RoBERTa and DistilBERT. Each entry is
# ``(backbone_attr, encoder_layer_path, pooler_path_or_None)`` where
# the layer path is the dotted string between the backbone prefix and
# the per-layer index.
_PROTOCOL_BACKBONES: Tuple[Tuple[str, str, str | None], ...] = (
    ("bert",       "encoder.layer",       "pooler.dense"),
    ("roberta",    "encoder.layer",       "pooler.dense"),  # roberta also has a pooler
    ("distilbert", "transformer.layer",   None),
)


def _infer_backbone_prefix(state_dict_keys) -> Tuple[str, str, str | None]:
    """Detect which backbone a state-dict belongs to.

    Returns ``(backbone, layer_prefix_template, pooler_prefix_or_None)``
    where ``layer_prefix_template`` is a format string with one ``{i}``
    placeholder, e.g. ``"bert.encoder.layer.{i}"``. Used by the protocol
    weight loaders so they work uniformly on BERT/RoBERTa/DistilBERT
    fine-tuned checkpoints.
    """
    keys = list(state_dict_keys)
    for backbone, layer_path, pooler_path in _PROTOCOL_BACKBONES:
        sentinel = f"{backbone}."
        if any(k.startswith(sentinel) for k in keys):
            layer_template = f"{backbone}.{layer_path}.{{i}}"
            pooler_full = f"{backbone}.{pooler_path}" if pooler_path else None
            return backbone, layer_template, pooler_full
    raise ValueError(
        f"State dict matches none of the known backbones "
        f"{[b for b,_,_ in _PROTOCOL_BACKBONES]}; "
        f"sample keys: {keys[:5]}"
    )


def load_model_weights(
    model_key: str,
    *,
    checkpoint_path: str | None = None,
    num_labels: int = 2,
) -> ModelWeights:
    """Pull plaintext weights from a HuggingFace BertForSequenceClassification.

    If ``checkpoint_path`` is None we use the original pretrained model
    from ``MODEL_REGISTRY``. Otherwise we load the LPAN-trained
    checkpoint, which has the same parameter shapes.
    """
    from transformers import AutoModelForSequenceClassification  # heavy import

    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=num_labels
    )
    sd = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    backbone, layer_template, pooler_prefix = _infer_backbone_prefix(sd.keys())

    layers: List[LayerWeights] = []
    for i in range(cfg["layers"]):
        p = layer_template.format(i=i)
        layers.append(
            LayerWeights(
                Wq=sd[f"{p}.attention.self.query.weight"],
                bq=sd[f"{p}.attention.self.query.bias"],
                Wk=sd[f"{p}.attention.self.key.weight"],
                bk=sd[f"{p}.attention.self.key.bias"],
                Wv=sd[f"{p}.attention.self.value.weight"],
                bv=sd[f"{p}.attention.self.value.bias"],
                Wo=sd[f"{p}.attention.output.dense.weight"],
                bo=sd[f"{p}.attention.output.dense.bias"],
                W1=sd[f"{p}.intermediate.dense.weight"],
                b1=sd[f"{p}.intermediate.dense.bias"],
                W2=sd[f"{p}.output.dense.weight"],
                b2=sd[f"{p}.output.dense.bias"],
                ln1_gamma=sd[f"{p}.attention.output.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.output.LayerNorm.bias"],
                ln2_gamma=sd[f"{p}.output.LayerNorm.weight"],
                ln2_beta=sd[f"{p}.output.LayerNorm.bias"],
            )
        )

    return ModelWeights(
        model_key=model_key,
        num_layers=cfg["layers"],
        hidden=cfg["hidden"],
        num_heads=cfg["heads"],
        layers=layers,
        pooler_W=sd.get(f"{pooler_prefix}.weight") if pooler_prefix else None,
        pooler_b=sd.get(f"{pooler_prefix}.bias") if pooler_prefix else None,
        cls_W=sd.get("classifier.weight"),
        cls_b=sd.get("classifier.bias"),
    )


@dataclass
class LinearMixingModelWeights:
    """Weights for a linear-mixing BERT variant (no attention)."""

    model_key: str
    num_layers: int
    hidden: int
    layers: List[LinearMixingLayerWeights] = field(default_factory=list)
    pooler_W: np.ndarray | None = None
    pooler_b: np.ndarray | None = None
    cls_W: np.ndarray | None = None
    cls_b: np.ndarray | None = None


def load_linear_mixing_weights(
    model_key: str,
    *,
    checkpoint_path: str,
    num_labels: int = 2,
) -> LinearMixingModelWeights:
    """Load weights from a multi-head linear-mixing fine-tuned checkpoint.

    The checkpoint contains ``pos_mix_weight`` / ``pos_mix_bias`` /
    ``out_proj`` parameters instead of Q/K/V/O (produced by the
    LinearMixing stage of the unified ``train_hyper_lpan`` pipeline).
    """
    from safetensors.torch import load_file as _load_safetensors
    from pathlib import Path

    cfg = MODEL_REGISTRY[model_key]
    ckpt = Path(checkpoint_path)

    sf = ckpt / "model.safetensors"
    bin_path = ckpt / "pytorch_model.bin"
    if sf.exists():
        sd = {k: v.numpy() for k, v in _load_safetensors(str(sf)).items()}
    else:
        import torch
        raw = torch.load(str(bin_path), map_location="cpu", weights_only=False)
        sd = {k: v.numpy() for k, v in raw.items()}

    num_heads = cfg.get("heads", 12)
    backbone, layer_template, pooler_prefix = _infer_backbone_prefix(sd.keys())
    layers: List[LinearMixingLayerWeights] = []
    for i in range(cfg["layers"]):
        p = layer_template.format(i=i)
        layers.append(
            LinearMixingLayerWeights(
                P_weights=sd[f"{p}.attention.pos_mix_weight"],
                P_biases=sd[f"{p}.attention.pos_mix_bias"],
                Wo=sd[f"{p}.attention.out_proj.weight"],
                bo=sd[f"{p}.attention.out_proj.bias"],
                W1=sd[f"{p}.intermediate.dense.weight"],
                b1=sd[f"{p}.intermediate.dense.bias"],
                W2=sd[f"{p}.output.dense.weight"],
                b2=sd[f"{p}.output.dense.bias"],
                ln1_gamma=sd[f"{p}.attention.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.LayerNorm.bias"],
                ln2_gamma=sd[f"{p}.output.LayerNorm.weight"],
                ln2_beta=sd[f"{p}.output.LayerNorm.bias"],
                num_heads=num_heads,
            )
        )

    return LinearMixingModelWeights(
        model_key=model_key,
        num_layers=cfg["layers"],
        hidden=cfg["hidden"],
        layers=layers,
        pooler_W=sd.get(f"{pooler_prefix}.weight") if pooler_prefix else None,
        pooler_b=sd.get(f"{pooler_prefix}.bias") if pooler_prefix else None,
        cls_W=sd.get("classifier.weight"),
        cls_b=sd.get("classifier.bias"),
    )


# ──────────────────────────────────────────────────────────────────────
# Encrypted block primitives (model-agnostic)
# ──────────────────────────────────────────────────────────────────────


def encrypt_ffn_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run FFN + post-LN under FHE.

    Pipeline: ``W₁ → GELU-poly → W₂ → residual → LN-poly``.
    ``n_jobs`` parallelises token-level loops (O5).
    """
    timings: Dict[str, float] = {}

    t = time.time()
    h = enc_linear(backend, x, layer.W1, layer.b1, n_jobs=n_jobs)
    timings["W1"] = time.time() - t

    t = time.time()
    g = coeffs["GELU"]
    h = enc_gelu_poly(backend, h, g.power_coeffs, g.interval, n_jobs=n_jobs)
    timings["GELU"] = time.time() - t

    t = time.time()
    h = enc_linear(backend, h, layer.W2, layer.b2, n_jobs=n_jobs)
    timings["W2"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln2_gamma,
        beta=layer.ln2_beta,
        n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_attention_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run MHA + post-LN under FHE.

    Pipeline: ``MHA → residual → LN-poly``.
    ``n_jobs`` parallelises token-level linear / LN loops (O5).
    """
    timings: Dict[str, float] = {}

    t = time.time()
    sm = coeffs["Softmax"]
    h = enc_self_attention(
        backend,
        x,
        layer.Wq,
        layer.bq,
        layer.Wk,
        layer.bk,
        layer.Wv,
        layer.bv,
        layer.Wo,
        layer.bo,
        softmax_coeffs=sm,
        num_heads=num_heads,
        n_jobs=n_jobs,
    )
    timings["MHA"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln1_gamma,
        beta=layer.ln1_beta,
        n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_layer(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run one full BERT encoder layer under FHE."""
    h, t_attn = encrypt_attention_block(backend, x, layer, coeffs, num_heads, n_jobs=n_jobs)
    h, t_ffn = encrypt_ffn_block(backend, h, layer, coeffs, n_jobs=n_jobs)
    timings = {f"attn.{k}": v for k, v in t_attn.items()}
    timings.update({f"ffn.{k}": v for k, v in t_ffn.items()})
    return h, timings


# ──────────────────────────────────────────────────────────────────────
# Linear mixing: FHE-friendly attention replacement
# ──────────────────────────────────────────────────────────────────────


def encrypt_linear_mixing_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LinearMixingLayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
    kept_token_indices: Optional[np.ndarray] = None,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run multi-head linear mixing + post-LN under FHE (replaces MHA block).

    Pipeline:
      For each head h: z_h = P_h @ x_h  (position mixing on head_dim slice)
      Concat heads → W_out @ z + b  (output projection)
      → residual → LN-poly

    ALL operations are plaintext × ciphertext.  Zero ct×ct multiplications.

    Parameters
    ----------
    kept_token_indices : np.ndarray or None
        When word elimination has dropped padding/low-importance tokens,
        this 1-D int array gives the *original* positions of the
        surviving tokens (length must equal ``x.seq_len``).  ``P_weights``
        and ``P_biases`` are sub-matrix-selected on these axes so the
        learned position-mixing pattern follows the kept positions.
        ``None`` (default) keeps the contiguous prefix ``[:L, :L]``
        — backward compatible with pre-elimination calls.
    """
    timings: Dict[str, float] = {}
    L = x.seq_len  # number of tokens
    D = x.hidden_dim
    H = layer.num_heads
    d = D // H  # head dimension

    # 1. Multi-head position mixing
    # For each output token j, for each head h:
    #   z_h[j] = sum_i P_h[j,i] * x_h[i] + P_bias_h[j]
    # where x_h[i] is the h-th head slice of token i
    t = time.time()

    # P_weights: (H, max_seq_len, max_seq_len) — select either the
    # contiguous prefix (default) or arbitrary kept positions (when
    # word elimination is active).
    if kept_token_indices is not None:
        idx = np.asarray(kept_token_indices, dtype=np.int64)
        if idx.shape[0] != L:
            raise ValueError(
                f"kept_token_indices length {idx.shape[0]} != x.seq_len {L}"
            )
        P_w = layer.P_weights[:, idx, :][:, :, idx]  # (H, L, L)
        P_b = layer.P_biases[:, idx]                 # (H, L)
    else:
        P_w = layer.P_weights[:, :L, :L]  # (H, L, L)
        P_b = layer.P_biases[:, :L]       # (H, L)

    # For each output token j, we need to combine per-head position mixing
    # Each ciphertext x.cts[i] packs D values: [head0_d0..head0_d63, head1_d0..head1_d63, ...]
    # We need to do per-head weighted sums across input tokens
    mixed_cts = []
    for j in range(L):
        # Build a per-head scaling vector: for each dim d in [0, D),
        # the scaling factor is P_h[j, i] where h = d // head_dim
        # We accumulate across input tokens i
        acc = None
        for i in range(L):
            # Create scale vector: repeat P_h[j,i] across each head's dims
            scale_vec = np.zeros(D, dtype=np.float64)
            for h in range(H):
                scale_vec[h * d:(h + 1) * d] = float(P_w[h, j, i])
            term = backend.mul_plain(x.cts[i], scale_vec)
            if acc is None:
                acc = term
            else:
                acc = backend.add(acc, term)
        # Add bias: repeat P_bias_h[j] across each head's dims
        bias_vec = np.zeros(D, dtype=np.float64)
        for h in range(H):
            bias_vec[h * d:(h + 1) * d] = float(P_b[h, j])
        acc = backend.add_plain(acc, bias_vec)
        mixed_cts.append(acc)
    pos_mixed = TokenPackedTensor.from_ciphertexts(mixed_cts, hidden_dim=D)
    timings["pos_mix"] = time.time() - t

    # 2. Output projection: W_out @ z (per-token plaintext matmul)
    t = time.time()
    feat_mixed = enc_linear(backend, pos_mixed, layer.Wo, layer.bo, n_jobs=n_jobs)
    timings["out_proj"] = time.time() - t

    # 3. Residual connection
    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(feat_mixed.cts[i], x.cts[i]) for i in range(L)],
        hidden_dim=D,
    )
    timings["residual"] = time.time() - t

    # 4. Post-mixing LayerNorm (polynomial)
    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln1_gamma,
        beta=layer.ln1_beta,
        n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_ffn_block_mixing(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LinearMixingLayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """FFN block using LinearMixingLayerWeights (same logic, different type)."""
    timings: Dict[str, float] = {}

    t = time.time()
    h = enc_linear(backend, x, layer.W1, layer.b1, n_jobs=n_jobs)
    timings["W1"] = time.time() - t

    t = time.time()
    g = coeffs["GELU"]
    h = enc_gelu_poly(backend, h, g.power_coeffs, g.interval, n_jobs=n_jobs)
    timings["GELU"] = time.time() - t

    t = time.time()
    h = enc_linear(backend, h, layer.W2, layer.b2, n_jobs=n_jobs)
    timings["W2"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln2_gamma,
        beta=layer.ln2_beta,
        n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_layer_linear_mix(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LinearMixingLayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
    kept_token_indices: Optional[np.ndarray] = None,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """One full encoder layer with linear mixing (no attention)."""
    h, t_mix = encrypt_linear_mixing_block(
        backend, x, layer, coeffs, n_jobs=n_jobs,
        kept_token_indices=kept_token_indices,
    )
    h, t_ffn = encrypt_ffn_block_mixing(backend, h, layer, coeffs, n_jobs=n_jobs)
    timings = {f"mix.{k}": v for k, v in t_mix.items()}
    timings.update({f"ffn.{k}": v for k, v in t_ffn.items()})
    return h, timings


def encrypt_inference_linear_mixing(
    backend: CKKSBackend,
    x_plain: np.ndarray,
    weights: LinearMixingModelWeights,
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
    max_seq_len: Optional[int] = None,
    n_jobs: int = 1,
    kept_token_indices: Optional[np.ndarray] = None,
    bootstrap_plan: Optional[object] = None,
    measure_depth: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Full linear-mixing encoder + classifier under FHE.

    Same interface as ``encrypt_inference`` but uses
    ``LinearMixingModelWeights`` and ``encrypt_layer_linear_mix``.
    Expected latency: ~3-10s per sample (vs. minutes with attention).

    ``kept_token_indices`` enables word elimination (see
    ``encrypt_inference_hybrid`` docstring for semantics).
    """
    timings: Dict[str, float] = {}

    if max_seq_len is not None and x_plain.shape[0] > max_seq_len:
        x_plain = x_plain[:max_seq_len]
        if kept_token_indices is not None:
            kept_token_indices = kept_token_indices[: x_plain.shape[0]]

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings["encrypt"] = time.time() - t

    h = ct_x
    if measure_depth:
        timings["level.initial"] = float(backend.get_level(h.cts[0]))
    for i, layer in enumerate(weights.layers):
        if bootstrap_plan is not None:
            from .bootstrap_scheduler import maybe_bootstrap
            t_bs = time.time()
            h = maybe_bootstrap(backend, h, bootstrap_plan, i)
            bs_dt = time.time() - t_bs
            if bs_dt > 0:
                timings[f"L{i}.bootstrap"] = bs_dt
        h, layer_t = encrypt_layer_linear_mix(
            backend, h, layer, coeffs[i], n_jobs=n_jobs,
            kept_token_indices=kept_token_indices,
        )
        for k, v in layer_t.items():
            timings[f"L{i}.{k}"] = v
        if measure_depth:
            timings[f"L{i}.level_after"] = float(backend.get_level(h.cts[0]))

    if weights.cls_W is not None:
        t = time.time()
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        timings["classifier"] = time.time() - t
    else:
        out_ct = h

    t = time.time()
    out = out_ct.decrypt(backend)
    timings["decrypt"] = time.time() - t

    timings["total"] = sum(timings.values())
    return out, timings


def encrypt_inference(
    backend: CKKSBackend,
    x_plain: np.ndarray,
    weights: ModelWeights,
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
    max_seq_len: Optional[int] = None,
    n_jobs: int = 1,
    bootstrap_plan: Optional[object] = None,
    measure_depth: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Full encoder + (optional) classifier head under FHE.

    Parameters
    ----------
    max_seq_len : int | None
        Truncate ``x_plain`` to this many tokens before encryption (O4).
        ``None`` means no truncation. Recommended: 64 for SST-2, 96 for QNLI.
    n_jobs : int
        Token-level parallelism forwarded to all block ops (O5).
        1 = serial, -1 = all CPUs.

    The classifier head is a plaintext Linear; we apply it with one
    final ``enc_linear`` if ``weights.cls_W`` is set. Returns the
    decrypted output and a flat latency dict.
    """
    timings: Dict[str, float] = {}

    # O4 — sequence truncation
    if max_seq_len is not None and x_plain.shape[0] > max_seq_len:
        x_plain = x_plain[:max_seq_len]

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings["encrypt"] = time.time() - t

    h = ct_x
    if measure_depth:
        timings["level.initial"] = float(backend.get_level(h.cts[0]))
    for i, layer in enumerate(weights.layers):
        if bootstrap_plan is not None:
            from .bootstrap_scheduler import maybe_bootstrap
            t_bs = time.time()
            h = maybe_bootstrap(backend, h, bootstrap_plan, i)
            bs_dt = time.time() - t_bs
            if bs_dt > 0:
                timings[f"L{i}.bootstrap"] = bs_dt
        h, layer_t = encrypt_layer(backend, h, layer, coeffs[i], weights.num_heads, n_jobs=n_jobs)
        for k, v in layer_t.items():
            timings[f"L{i}.{k}"] = v
        if measure_depth:
            timings[f"L{i}.level_after"] = float(backend.get_level(h.cts[0]))

    # Pooler + classifier on the [CLS] token only.
    if weights.cls_W is not None:
        t = time.time()
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
            # tanh is non-poly; the LPAN pipeline trains with the pooler
            # frozen so we approximate it as identity here. (See thesis
            # §3 — the [CLS] head is folded into the classifier in
            # downstream LPAN runs.)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        timings["classifier"] = time.time() - t
    else:
        out_ct = h

    t = time.time()
    out = out_ct.decrypt(backend)
    timings["decrypt"] = time.time() - t

    timings["total"] = sum(timings.values())
    return out, timings


# ──────────────────────────────────────────────────────────────────────
# Matrix-packed encoder — same pipeline, B× fewer ciphertexts
# ──────────────────────────────────────────────────────────────────────


def encrypt_attention_block_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
) -> Tuple[MatrixPackedTensor, Dict[str, float]]:
    """MHA + post-LN under FHE on a matrix-packed activation."""
    timings: Dict[str, float] = {}

    t = time.time()
    sm = coeffs["Softmax"]
    h = enc_self_attention_matrix(
        backend, x,
        layer.Wq, layer.bq, layer.Wk, layer.bk,
        layer.Wv, layer.bv, layer.Wo, layer.bo,
        softmax_coeffs=sm, num_heads=num_heads,
    )
    timings["MHA"] = time.time() - t

    t = time.time()
    res_cts = [backend.add(h.cts[i], x.cts[i]) for i in range(len(x.cts))]
    res = MatrixPackedTensor.from_ciphertexts(
        res_cts, seq_len=x.seq_len, hidden_dim=x.hidden_dim,
        block=x.block, tokens_per_ct=x.tokens_per_ct, num_slots=x.num_slots,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_layernorm_matrix(
        backend, res, ln.power_coeffs, ln.interval,
        gamma=layer.ln1_gamma, beta=layer.ln1_beta,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_ffn_block_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
) -> Tuple[MatrixPackedTensor, Dict[str, float]]:
    """FFN + post-LN under FHE on a matrix-packed activation.

    Note: ``W1`` expands hidden→4·hidden which may exceed the per-token
    block. We allocate a wider intermediate by re-packing only when the
    expansion overflows (BERT-base: 768→3072 fits if block≥4096, else
    needs MPT re-pack — addressed in a follow-up if/when triggered).
    """
    timings: Dict[str, float] = {}

    # Sanity: FFN expansion width must fit within block.
    out_dim = layer.W1.shape[0]
    if next_pow2(out_dim) > x.block:
        raise ValueError(
            f"FFN expansion out_dim={out_dim} (next_pow2={next_pow2(out_dim)}) "
            f"exceeds packing block={x.block}; re-pack with a larger block "
            f"or use token-packed FFN."
        )

    t = time.time()
    h = enc_linear_matrix(backend, x, layer.W1, bias=layer.b1)
    timings["W1"] = time.time() - t

    t = time.time()
    g = coeffs["GELU"]
    h = enc_gelu_matrix(backend, h, g.power_coeffs, g.interval)
    timings["GELU"] = time.time() - t

    t = time.time()
    h = enc_linear_matrix(backend, h, layer.W2, bias=layer.b2)
    timings["W2"] = time.time() - t

    t = time.time()
    res_cts = [backend.add(h.cts[i], x.cts[i]) for i in range(len(x.cts))]
    res = MatrixPackedTensor.from_ciphertexts(
        res_cts, seq_len=x.seq_len, hidden_dim=x.hidden_dim,
        block=x.block, tokens_per_ct=x.tokens_per_ct, num_slots=x.num_slots,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_layernorm_matrix(
        backend, res, ln.power_coeffs, ln.interval,
        gamma=layer.ln2_gamma, beta=layer.ln2_beta,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_layer_matrix(
    backend: CKKSBackend,
    x: MatrixPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
) -> Tuple[MatrixPackedTensor, Dict[str, float]]:
    """One full BERT encoder layer (matrix-packed)."""
    h, t_attn = encrypt_attention_block_matrix(backend, x, layer, coeffs, num_heads)
    h, t_ffn = encrypt_ffn_block_matrix(backend, h, layer, coeffs)
    timings = {f"attn.{k}": v for k, v in t_attn.items()}
    timings.update({f"ffn.{k}": v for k, v in t_ffn.items()})
    return h, timings


def encrypt_inference_matrix(
    backend: CKKSBackend,
    x_plain: np.ndarray,
    weights: ModelWeights,
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
    max_seq_len: Optional[int] = None,
    block: int = 0,
    bootstrap_plan: Optional[object] = None,
    measure_depth: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Matrix-packed end-to-end FHE inference.

    Mirrors :func:`encrypt_inference` but every encoder layer uses the
    matrix-packed kernels in :mod:`ops_matrix`. The classifier head and
    pooler still use the token-packed primitives (one ct only).

    Parameters
    ----------
    block : int
        Packing block (per-token slot stride). 0 → ``next_pow2(out_dim_max)``
        across the layer set so the FFN expansion fits in-block.
    """
    timings: Dict[str, float] = {}

    if max_seq_len is not None and x_plain.shape[0] > max_seq_len:
        x_plain = x_plain[:max_seq_len]

    # Pick a block large enough for every linear in the model. The widest
    # expansion is W1's out_dim (4·hidden for BERT) — round to next pow2.
    if block <= 0:
        max_dim = max(
            (max(layer.W1.shape[0], layer.W2.shape[0]) for layer in weights.layers),
            default=weights.hidden,
        )
        block = next_pow2(max(max_dim, weights.hidden))

    t = time.time()
    ct_x = MatrixPackedTensor.encrypt(backend, x_plain, block=block)
    timings["encrypt"] = time.time() - t
    timings["pack.tokens_per_ct"] = float(ct_x.tokens_per_ct)
    timings["pack.num_cts"] = float(len(ct_x.cts))
    timings["pack.block"] = float(ct_x.block)

    h = ct_x
    if measure_depth:
        timings["level.initial"] = float(backend.get_level(h.cts[0]))
    for i, layer in enumerate(weights.layers):
        if bootstrap_plan is not None:
            from .bootstrap_scheduler import maybe_bootstrap
            t_bs = time.time()
            # bootstrap_plan operates on TokenPackedTensor today; treat
            # MatrixPackedTensor as a list of cts for refresh purposes.
            for j in range(len(h.cts)):
                refreshed = maybe_bootstrap(backend, h.cts[j], bootstrap_plan, i)
                if refreshed is not h.cts[j]:
                    h.cts[j] = refreshed
            bs_dt = time.time() - t_bs
            if bs_dt > 0:
                timings[f"L{i}.bootstrap"] = bs_dt
        h, layer_t = encrypt_layer_matrix(
            backend, h, layer, coeffs[i], weights.num_heads
        )
        for k, v in layer_t.items():
            timings[f"L{i}.{k}"] = v
        if measure_depth:
            timings[f"L{i}.level_after"] = float(backend.get_level(h.cts[0]))

    # Classifier head: extract [CLS] (token 0) into a single-ct
    # token-packed tensor so we can reuse the existing classifier path.
    t = time.time()
    if weights.cls_W is not None:
        # Decrypt-then-take-cls would defeat FHE; instead extract the
        # [CLS] block (slots [0..hidden_dim)) of the first ciphertext.
        cls_ct = h.cts[0]
        # Mask out everything but the first hidden_dim slots.
        n_slots = h.num_slots
        mask = [0.0] * n_slots
        for j in range(h.hidden_dim):
            mask[j] = 1.0
        cls_ct = backend.mul_plain(cls_ct, mask)
        cls = TokenPackedTensor.from_ciphertexts([cls_ct], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        timings["classifier"] = time.time() - t
        t = time.time()
        out = out_ct.decrypt(backend)
        timings["decrypt"] = time.time() - t
    else:
        timings["classifier"] = 0.0
        t = time.time()
        out = h.decrypt(backend)
        timings["decrypt"] = time.time() - t

    timings["total"] = sum(timings.values())
    return out, timings


# ──────────────────────────────────────────────────────────────────────
# 2Quad attention: shallow polynomial attention (no softmax)
# ──────────────────────────────────────────────────────────────────────


def encrypt_quad_attention_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: "QuadLayerWeights",
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run 2Quad MHA + post-LN under FHE (no polynomial-softmax chain).

    Pipeline: ``2Quad-MHA → residual → LN-poly``.  Compared to LPAN's
    ``encrypt_attention_block``, the inner softmax (depth ~3-5,
    4-5 ct×ct) is replaced with a single squaring + scalar /L
    (depth 2, 1 ct×ct).
    """
    timings: Dict[str, float] = {}

    t = time.time()
    h = enc_self_quad_attention(
        backend, x,
        layer.Wq, layer.bq, layer.Wk, layer.bk,
        layer.Wv, layer.bv, layer.Wo, layer.bo,
        num_heads=layer.num_heads, n_jobs=n_jobs,
    )
    timings["MHA"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend, res, ln.power_coeffs, ln.interval,
        gamma=layer.ln1_gamma, beta=layer.ln1_beta, n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_ffn_block_quad(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: "QuadLayerWeights",
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """FFN block using QuadLayerWeights (identical FFN to LPAN, different bundle type)."""
    timings: Dict[str, float] = {}

    t = time.time()
    h = enc_linear(backend, x, layer.W1, layer.b1, n_jobs=n_jobs)
    timings["W1"] = time.time() - t

    t = time.time()
    g = coeffs["GELU"]
    h = enc_gelu_poly(backend, h, g.power_coeffs, g.interval, n_jobs=n_jobs)
    timings["GELU"] = time.time() - t

    t = time.time()
    h = enc_linear(backend, h, layer.W2, layer.b2, n_jobs=n_jobs)
    timings["W2"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend, res, ln.power_coeffs, ln.interval,
        gamma=layer.ln2_gamma, beta=layer.ln2_beta, n_jobs=n_jobs,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_layer_quad(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: "QuadLayerWeights",
    coeffs: Dict[str, PolyCoeffs],
    n_jobs: int = 1,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """One full encoder layer with 2Quad attention."""
    h, t_attn = encrypt_quad_attention_block(backend, x, layer, coeffs, n_jobs=n_jobs)
    h, t_ffn = encrypt_ffn_block_quad(backend, h, layer, coeffs, n_jobs=n_jobs)
    timings = {f"quad.{k}": v for k, v in t_attn.items()}
    timings.update({f"ffn.{k}": v for k, v in t_ffn.items()})
    return h, timings


# ──────────────────────────────────────────────────────────────────────
# Hybrid model: LPAN (deep) + 2Quad (mid) + LinearMixing (early)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HybridModelWeights:
    """Weights for a HyPER-LPAN model.

    A list of per-layer weight bundles (one of LinearMixingLayerWeights /
    QuadLayerWeights / LayerWeights), in encoder order.  The pipeline
    dispatches to the right block function based on the bundle's runtime
    type, so adding a fourth attention variant later is purely additive.
    """

    model_key: str
    num_layers: int
    hidden: int
    num_heads: int
    layers: List[object] = field(default_factory=list)  # per-layer bundles
    pooler_W: Optional[np.ndarray] = None
    pooler_b: Optional[np.ndarray] = None
    cls_W: Optional[np.ndarray] = None
    cls_b: Optional[np.ndarray] = None


def load_hybrid_weights(
    model_key: str,
    *,
    checkpoint_path: str,
    linear_mixing_layers: Sequence[int],
    quad_attention_layers: Sequence[int],
    num_labels: int = 2,
) -> HybridModelWeights:
    """Load a HyPER-LPAN checkpoint, dispatching per-layer to the right bundle.

    Parameters
    ----------
    linear_mixing_layers, quad_attention_layers : sequences of int
        Same lists passed to ``apply_hybrid_attention`` at training time.
        Layers not in either list are loaded as standard LPAN
        ``LayerWeights``.

    The checkpoint is the safetensors file produced by
    ``HyperLPANPipeline`` (``best_model/model.safetensors``).
    """
    from pathlib import Path

    from safetensors.torch import load_file as _load_safetensors

    cfg = MODEL_REGISTRY[model_key]
    ckpt = Path(checkpoint_path)
    sf = ckpt / "model.safetensors"
    bin_path = ckpt / "pytorch_model.bin"
    if sf.exists():
        sd = {k: v.numpy() for k, v in _load_safetensors(str(sf)).items()}
    else:
        import torch as _torch
        raw = _torch.load(str(bin_path), map_location="cpu", weights_only=False)
        sd = {k: v.numpy() for k, v in raw.items()}

    num_heads = cfg.get("heads", 12)
    backbone, layer_template, pooler_prefix = _infer_backbone_prefix(sd.keys())
    lm_set = set(linear_mixing_layers)
    quad_set = set(quad_attention_layers)
    overlap = lm_set & quad_set
    if overlap:
        raise ValueError(
            f"Layers {sorted(overlap)} cannot be both linear-mixing and quad"
        )

    layers: List[object] = []
    for i in range(cfg["layers"]):
        p = layer_template.format(i=i)
        # FFN + final LN are identical across all variants
        ffn_args = dict(
            W1=sd[f"{p}.intermediate.dense.weight"],
            b1=sd[f"{p}.intermediate.dense.bias"],
            W2=sd[f"{p}.output.dense.weight"],
            b2=sd[f"{p}.output.dense.bias"],
            ln2_gamma=sd[f"{p}.output.LayerNorm.weight"],
            ln2_beta=sd[f"{p}.output.LayerNorm.bias"],
        )
        if i in lm_set:
            layers.append(LinearMixingLayerWeights(
                P_weights=sd[f"{p}.attention.pos_mix_weight"],
                P_biases=sd[f"{p}.attention.pos_mix_bias"],
                Wo=sd[f"{p}.attention.out_proj.weight"],
                bo=sd[f"{p}.attention.out_proj.bias"],
                ln1_gamma=sd[f"{p}.attention.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.LayerNorm.bias"],
                num_heads=num_heads,
                **ffn_args,
            ))
        elif i in quad_set:
            layers.append(QuadLayerWeights(
                Wq=sd[f"{p}.attention.query.weight"],
                bq=sd[f"{p}.attention.query.bias"],
                Wk=sd[f"{p}.attention.key.weight"],
                bk=sd[f"{p}.attention.key.bias"],
                Wv=sd[f"{p}.attention.value.weight"],
                bv=sd[f"{p}.attention.value.bias"],
                Wo=sd[f"{p}.attention.out_proj.weight"],
                bo=sd[f"{p}.attention.out_proj.bias"],
                ln1_gamma=sd[f"{p}.attention.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.LayerNorm.bias"],
                num_heads=num_heads,
                **ffn_args,
            ))
        else:
            # LPAN: standard BertSelfAttention + BertSelfOutput.LayerNorm path
            layers.append(LayerWeights(
                Wq=sd[f"{p}.attention.self.query.weight"],
                bq=sd[f"{p}.attention.self.query.bias"],
                Wk=sd[f"{p}.attention.self.key.weight"],
                bk=sd[f"{p}.attention.self.key.bias"],
                Wv=sd[f"{p}.attention.self.value.weight"],
                bv=sd[f"{p}.attention.self.value.bias"],
                Wo=sd[f"{p}.attention.output.dense.weight"],
                bo=sd[f"{p}.attention.output.dense.bias"],
                ln1_gamma=sd[f"{p}.attention.output.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.output.LayerNorm.bias"],
                **ffn_args,
            ))

    return HybridModelWeights(
        model_key=model_key,
        num_layers=cfg["layers"],
        hidden=cfg["hidden"],
        num_heads=num_heads,
        layers=layers,
        pooler_W=sd.get(f"{pooler_prefix}.weight") if pooler_prefix else None,
        pooler_b=sd.get(f"{pooler_prefix}.bias") if pooler_prefix else None,
        cls_W=sd.get("classifier.weight"),
        cls_b=sd.get("classifier.bias"),
    )


def encrypt_layer_dispatch(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: object,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
    n_jobs: int = 1,
    kept_token_indices: Optional[np.ndarray] = None,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run one encoder layer, dispatching by bundle type.

    Hot path of the hybrid pipeline: avoids ``isinstance`` cascading
    by deferring to the existing per-variant block functions.

    ``kept_token_indices`` is forwarded to LinearMixing layers only —
    Quad and LPAN attention are token-position-agnostic and don't need
    to know which original positions survived word elimination.
    """
    if isinstance(layer, LinearMixingLayerWeights):
        return encrypt_layer_linear_mix(
            backend, x, layer, coeffs, n_jobs=n_jobs,
            kept_token_indices=kept_token_indices,
        )
    if isinstance(layer, QuadLayerWeights):
        return encrypt_layer_quad(backend, x, layer, coeffs, n_jobs=n_jobs)
    if isinstance(layer, LayerWeights):
        return encrypt_layer(backend, x, layer, coeffs, num_heads, n_jobs=n_jobs)
    raise TypeError(f"Unknown layer bundle type: {type(layer).__name__}")


def encrypt_inference_hybrid(
    backend: CKKSBackend,
    x_plain: np.ndarray,
    weights: HybridModelWeights,
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
    max_seq_len: Optional[int] = None,
    n_jobs: int = 1,
    kept_token_indices: Optional[np.ndarray] = None,
    bootstrap_plan: Optional[object] = None,
    measure_depth: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """End-to-end HyPER-LPAN encrypted forward pass.

    Per-layer dispatch on bundle type (LinearMixing / Quad / LPAN) keeps
    the full ``encrypt_inference`` interface while letting each layer use
    its cheapest available primitives.

    Notes
    -----
    The polynomial coefficients ``coeffs[i]`` may omit the ``"Softmax"``
    key for layers that are LinearMixing or Quad — neither uses the LPAN
    softmax polynomial.  Only ``"GELU"`` and ``"LN"`` are required for
    every layer.

    Word elimination
    ----------------
    When ``kept_token_indices`` is provided, ``x_plain`` is assumed to
    already contain only the surviving tokens (length = ``len(kept_token_indices)``).
    The indices give the original positions of those tokens in the
    pre-eliminated sequence and are forwarded to LinearMixing layers so
    they can sub-matrix-select their position-mixing weights correctly.
    Caller is responsible for filtering ``x_plain`` upstream (use
    ``fhe_thesis.encryption.elimination.apply_elimination``).
    """
    timings: Dict[str, float] = {}

    if max_seq_len is not None and x_plain.shape[0] > max_seq_len:
        x_plain = x_plain[:max_seq_len]
        if kept_token_indices is not None:
            kept_token_indices = kept_token_indices[: x_plain.shape[0]]

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings["encrypt"] = time.time() - t

    h = ct_x
    if measure_depth:
        timings["level.initial"] = float(backend.get_level(h.cts[0]))
    for i, layer in enumerate(weights.layers):
        if bootstrap_plan is not None:
            from .bootstrap_scheduler import maybe_bootstrap
            t_bs = time.time()
            h = maybe_bootstrap(backend, h, bootstrap_plan, i)
            bs_dt = time.time() - t_bs
            if bs_dt > 0:
                timings[f"L{i}.bootstrap"] = bs_dt
        h, layer_t = encrypt_layer_dispatch(
            backend, h, layer, coeffs[i],
            num_heads=weights.num_heads, n_jobs=n_jobs,
            kept_token_indices=kept_token_indices,
        )
        for k, v in layer_t.items():
            timings[f"L{i}.{k}"] = v
        if measure_depth:
            timings[f"L{i}.level_after"] = float(backend.get_level(h.cts[0]))

    if weights.cls_W is not None:
        t = time.time()
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        timings["classifier"] = time.time() - t
    else:
        out_ct = h

    t = time.time()
    out = out_ct.decrypt(backend)
    timings["decrypt"] = time.time() - t

    timings["total"] = sum(timings.values())
    return out, timings


# ──────────────────────────────────────────────────────────────────────
# Convenience: one-call helper for a model + phase
# ──────────────────────────────────────────────────────────────────────


def run_phase(
    phase: str,
    model_key: str,
    backend: CKKSBackend,
    x_plain: np.ndarray,
    *,
    layer_idx: int = 0,
    checkpoint_path: str | None = None,
    max_seq_len: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Dispatch helper used by ``experiments/run_protocol.py``.

    ``phase ∈ {"ffn", "attention", "layer", "model"}``.
    ``max_seq_len`` and ``n_jobs`` are forwarded to the relevant block function.
    """
    weights = load_model_weights(model_key, checkpoint_path=checkpoint_path)
    coeffs = load_coefficients(model_key)

    if phase == "model":
        return encrypt_inference(backend, x_plain, weights, coeffs,
                                 max_seq_len=max_seq_len, n_jobs=n_jobs)

    # O4 — truncate for sub-model phases too
    if max_seq_len is not None and x_plain.shape[0] > max_seq_len:
        x_plain = x_plain[:max_seq_len]

    layer = weights.layers[layer_idx]
    layer_coeffs = coeffs[layer_idx]
    timings_total: Dict[str, float] = {}

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings_total["encrypt"] = time.time() - t

    if phase == "ffn":
        out_ct, t_block = encrypt_ffn_block(backend, ct_x, layer, layer_coeffs, n_jobs=n_jobs)
    elif phase == "attention":
        out_ct, t_block = encrypt_attention_block(
            backend, ct_x, layer, layer_coeffs, weights.num_heads, n_jobs=n_jobs
        )
    elif phase == "layer":
        out_ct, t_block = encrypt_layer(
            backend, ct_x, layer, layer_coeffs, weights.num_heads, n_jobs=n_jobs
        )
    else:
        raise ValueError(f"unknown phase {phase!r}")
    timings_total.update(t_block)

    t = time.time()
    out = out_ct.decrypt(backend)
    timings_total["decrypt"] = time.time() - t
    timings_total["total"] = sum(timings_total.values())
    return out, timings_total
