#!/usr/bin/env python3
"""Stage-4 validation: end-to-end OpenFHE logits vs plaintext LPAN logits.

For each real SST-2 sample we run:
  • plaintext LPAN forward  : input_ids → logits          (PyTorch)
  • encrypted OpenFHE forward: token embeds → logits     (CKKS + bootstrap)
and report L1/L2/Linf/relative-L2 plus argmax agreement.

Usage
-----
    python experiments/validate_e2e_openfhe.py \
        --model tiny --task sst2 --num-samples 3 \
        --seq-len 8 --mult-depth 45 --ring-dim 16384

Writes ``results/encrypted_inference/<model>_<task>_e2e_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace as _dc_replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhe_thesis.config import (  # noqa: E402
    ENCRYPTED_INFERENCE_DIR,
    MODEL_REGISTRY,
    MULTI_MODEL_DIR,
    ensure_dirs,
)
from fhe_thesis.poly.approximation import (  # noqa: E402
    exp_func,
    gelu_func,
    weighted_minimax_approx,
)
from fhe_thesis.poly.chebyshev import chebyshev_to_power  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--task", default="sst2")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mult-depth", type=int, default=45)
    p.add_argument("--ring-dim", type=int, default=1 << 14)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--range-margin",
        type=float,
        default=0.30,
        help="Widen observed (min,max) by this fraction on each side before refitting polynomials.",
    )
    p.add_argument(
        "--refit",
        action="store_true",
        help="Refit GELU/Softmax polynomials on inference-set activation ranges.\n"
             "WARNING: this discards the trained polynomial coefficients and replaces\n"
             "them with vanilla Chebyshev approximations of gelu/exp on a wider interval.\n"
             "In practice this loses task accuracy — the trained polynomials are tuned\n"
             "end-to-end and are not generic approximations. Provided for diagnostic\n"
             "comparison only.",
    )
    p.add_argument(
        "--profile-samples",
        type=int,
        default=64,
        help="How many task samples to profile for activation ranges (used by --refit).",
    )
    return p.parse_args()


def _resolve_ckpt(model_key: str, override: str | None) -> Path:
    if override:
        return Path(override)
    cand = MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"
    if cand.exists():
        return cand
    cand2 = MULTI_MODEL_DIR / model_key / "stage4_range_aware" / "best_model"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"No LPAN checkpoint for {model_key}")


def _sync_intervals(layer_coeffs, bert_layer):
    """Overwrite hardcoded intervals with the trained polynomial intervals."""
    g = bert_layer.intermediate.intermediate_act_fn
    layer_coeffs["GELU"] = _dc_replace(
        layer_coeffs["GELU"], interval=(float(g.a), float(g.b))
    )
    ln = bert_layer.output.LayerNorm
    layer_coeffs["LN"] = _dc_replace(
        layer_coeffs["LN"], interval=(float(ln.a), float(ln.b))
    )
    if "Softmax" in layer_coeffs:
        sm = bert_layer.attention.self.poly_softmax
        layer_coeffs["Softmax"] = _dc_replace(
            layer_coeffs["Softmax"], interval=(float(sm.a), float(sm.b))
        )


def _profile_activation_ranges(model, tokenizer, ds, indices, seq_len, text_fn):
    """One forward pass that records GELU/Softmax input min/max per layer.

    Returns ``{layer_idx: {'GELU': (mn, mx), 'Softmax': (mn, mx)}}``.
    """
    import torch

    captured: dict[str, list[float]] = {}

    def make_hook(name: str):
        def _hook(_mod, inputs, _out):
            x = inputs[0].detach().float()
            mn, mx = float(x.min().item()), float(x.max().item())
            lst = captured.setdefault(name, [+1e30, -1e30])
            lst[0] = min(lst[0], mn)
            lst[1] = max(lst[1], mx)

        return _hook

    handles = []
    for li, blk in enumerate(model.bert.encoder.layer):
        handles.append(
            blk.intermediate.intermediate_act_fn.register_forward_hook(
                make_hook(f"L{li}_GELU")
            )
        )
        handles.append(
            blk.attention.self.poly_softmax.register_forward_hook(
                make_hook(f"L{li}_Softmax")
            )
        )

    with torch.no_grad():
        for idx in indices:
            t1, t2 = text_fn(ds[int(idx)])
            kwargs = dict(
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            enc = tokenizer(t1, t2, **kwargs) if t2 is not None else tokenizer(t1, **kwargs)
            _ = model(**enc).logits

    for h in handles:
        h.remove()

    out: dict[int, dict[str, tuple[float, float]]] = {}
    for li in range(len(model.bert.encoder.layer)):
        out[li] = {
            "GELU": tuple(captured.get(f"L{li}_GELU", (-1.0, 1.0))),
            "Softmax": tuple(captured.get(f"L{li}_Softmax", (-5.0, 5.0))),
        }
    return out


def _refit_widened(
    layer_coeffs, ranges_per_layer, layer_idx, margin: float
):
    """Refit GELU and Softmax polynomials on a widened interval.

    The trained polynomials approximate fixed canonical functions
    (gelu / exp), so refitting on a wider domain just yields a slightly
    different polynomial that remains accurate on the original range
    while extending validity to observed activations.

    LN is left untouched: its input is the *variance* (always positive
    and well-bounded) and the trained interval already covers it.
    """
    rng = ranges_per_layer.get(layer_idx, {})
    for op_name, func in (("GELU", gelu_func), ("Softmax", exp_func)):
        if op_name not in layer_coeffs or op_name not in rng:
            continue
        old = layer_coeffs[op_name]
        mn, mx = rng[op_name]
        span = max(mx - mn, 1e-6)
        a_new = mn - margin * span
        b_new = mx + margin * span
        # Uniform density (no profile data on widened tails).
        density = lambda x: np.ones_like(x, dtype=float)
        cheb_c, _ = weighted_minimax_approx(func, (a_new, b_new), old.degree, density)
        if old.per_head:
            num_heads = old.power_coeffs.shape[0]
            power = np.stack(
                [np.asarray(chebyshev_to_power(cheb_c), dtype=np.float64)
                 for _ in range(num_heads)]
            )
        else:
            power = np.asarray(chebyshev_to_power(cheb_c), dtype=np.float64)
        layer_coeffs[op_name] = _dc_replace(
            old, power_coeffs=power, interval=(float(a_new), float(b_new))
        )


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    hidden = cfg["hidden"]
    ckpt = _resolve_ckpt(args.model, args.checkpoint)
    print(
        f"[validate-e2e] model={args.model} ({cfg['short']}) hidden={hidden} "
        f"layers={cfg['layers']} heads={cfg['heads']} seq_len={args.seq_len}"
    )
    print(f"[validate-e2e] ckpt = {ckpt}")
    print(f"[validate-e2e] N = {args.num_samples} samples (task={args.task})")

    # ── heavy imports ──────────────────────────────────────────────
    import torch as _torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    from fhe_thesis.encryption.ops import enc_linear
    from fhe_thesis.encryption.packing import TokenPackedTensor
    from fhe_thesis.encryption.protocol import (
        encrypt_attention_block,
        encrypt_ffn_block,
        load_model_weights,
    )
    from fhe_thesis.models.lpan_loader import load_lpan_model

    # ── load model + weights + coefficients ─────────────────────────
    print("\n[1/4] loading LPAN model + extracted coefficients ...")
    model = load_lpan_model(args.model, ckpt, num_labels=2)
    weights = load_model_weights(args.model, checkpoint_path=str(ckpt))
    coeffs = load_coefficients(args.model, task=args.task)
    # Sync intervals with the trained polynomials so the encrypted side
    # uses identical normalisation.
    for li in range(len(weights.layers)):
        _sync_intervals(coeffs[li], model.bert.encoder.layer[li])
    print(f"      layers={len(weights.layers)}  cls_W={weights.cls_W is not None}")

    # ── range-widening refit (optional, off by default) ─────────────
    if args.refit:
        print(
            f"\n[1b/4] profiling activation ranges on {args.profile_samples} {args.task.upper()} samples for refit ..."
        )
        from datasets import load_dataset as _ld
        from transformers import AutoTokenizer as _AT

        _tok_prof = _AT.from_pretrained(cfg["name"])
        if args.task == "sst2":
            _ds_prof = _ld("glue", "sst2", split="validation")
            _text_fn = lambda r: (r["sentence"], None)
        elif args.task == "mrpc":
            _ds_prof = _ld("glue", "mrpc", split="validation")
            _text_fn = lambda r: (r["sentence1"], r["sentence2"])
        else:
            raise ValueError(args.task)
        _rng_prof = np.random.default_rng(args.seed + 1)
        _prof_idx = _rng_prof.choice(
            len(_ds_prof),
            size=min(args.profile_samples, len(_ds_prof)),
            replace=False,
        ).tolist()
        ranges = _profile_activation_ranges(
            model, _tok_prof, _ds_prof, _prof_idx, args.seq_len, _text_fn
        )
        print("      observed ranges (before refit):")
        for li in range(len(weights.layers)):
            g = ranges[li]["GELU"]
            s = ranges[li]["Softmax"]
            old_g = coeffs[li]["GELU"].interval
            old_s = coeffs[li]["Softmax"].interval if "Softmax" in coeffs[li] else (0, 0)
            print(
                f"        L{li}  GELU obs=[{g[0]:7.3f},{g[1]:7.3f}] "
                f"old=[{old_g[0]:7.3f},{old_g[1]:7.3f}]    "
                f"Softmax obs=[{s[0]:7.3f},{s[1]:7.3f}] "
                f"old=[{old_s[0]:7.3f},{old_s[1]:7.3f}]"
            )
        for li in range(len(weights.layers)):
            _refit_widened(coeffs[li], ranges, li, args.range_margin)
        print(
            f"      refit done with margin={args.range_margin:.0%} on each side."
        )

    # ── slot / backend setup ───────────────────────────────────────
    ffn_inter = 4 * hidden
    needed = max(hidden, ffn_inter, args.seq_len, args.seq_len * args.seq_len)
    num_slots = 1
    while num_slots < needed:
        num_slots <<= 1
    print(f"      num_slots = {num_slots}")

    print("\n[2/4] booting OpenFHE backend ...")
    import openfhe as ofhe

    t0 = time.perf_counter()
    backend = OpenFHEBackend(
        multiplicative_depth=args.mult_depth,
        ring_dim=args.ring_dim,
        scaling_mod_size=59,
        first_mod_size=60,
        enable_bootstrap=True,
        num_slots=num_slots,
        security_level=ofhe.SecurityLevel.HEStd_128_classic
        if args.ring_dim >= (1 << 16)
        else ofhe.SecurityLevel.HEStd_NotSet,
    )
    setup_time = time.perf_counter() - t0
    print(f"      setup wall = {setup_time:.1f}s")

    # ── sample real SST-2 inputs ───────────────────────────────────
    print(f"\n[3/4] sampling {args.num_samples} real {args.task.upper()} examples ...")
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    ds_split = "validation"
    if args.task == "sst2":
        ds = load_dataset("glue", "sst2", split=ds_split)
        text_field = "sentence"
    elif args.task == "mrpc":
        ds = load_dataset("glue", "mrpc", split=ds_split)
        text_field = None  # use sentence1+sentence2
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.num_samples, replace=False).tolist()

    # ── per-sample loop ────────────────────────────────────────────
    per_sample = []
    for s_idx, ds_idx in enumerate(indices):
        row = ds[int(ds_idx)]
        if text_field:
            enc = tok(
                row[text_field],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.seq_len,
            )
        else:
            enc = tok(
                row["sentence1"],
                row["sentence2"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.seq_len,
            )
        true_label = int(row["label"])

        # Plaintext LPAN forward (full model, polynomial activations).
        with _torch.no_grad():
            pt_out = model(**enc).logits[0].numpy().astype(np.float64)
        pt_pred = int(np.argmax(pt_out))

        # Get the input embeddings (what we'll feed into the encrypted forward).
        with _torch.no_grad():
            embeds = model.bert.embeddings(enc["input_ids"])[0].numpy().astype(np.float64)

        # ── encrypted forward ────────────────────────────────────
        print(f"\n  ── sample {s_idx + 1}/{args.num_samples} (ds_idx={ds_idx}, true={true_label}) ──")
        t_sample = time.perf_counter()
        h = TokenPackedTensor.encrypt(backend, embeds)

        for li, layer in enumerate(weights.layers):
            t_layer = time.perf_counter()
            h, _ = encrypt_attention_block(backend, h, layer, coeffs[li], weights.num_heads)
            new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)
            h, _ = encrypt_ffn_block(backend, h, layer, coeffs[li])
            new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)
            # Diagnostic: peek at decrypted CLS slot magnitude.
            try:
                cls_dec = backend.decrypt(h.cts[0])[: h.hidden_dim]
                cls_mag = float(np.max(np.abs(cls_dec)))
                cls_norm = float(np.linalg.norm(cls_dec))
            except Exception as e:
                cls_mag = float("nan")
                cls_norm = float("nan")
                print(f"     layer {li}: DECRYPT FAILED: {e}")
            print(
                f"     layer {li}: {time.perf_counter() - t_layer:6.1f}s  "
                f"level={backend.get_level(h.cts[0])}/{args.mult_depth}  "
                f"|CLS|max={cls_mag:.3f}  |CLS|2={cls_norm:.3f}"
            )

        # Classifier head (CLS token = position 0).
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
            try:
                pl_dec = backend.decrypt(cls.cts[0])[: cls.hidden_dim]
                print(f"     post-pooler |max|={float(np.max(np.abs(pl_dec))):.3f}  |L2|={float(np.linalg.norm(pl_dec)):.3f}")
            except Exception as e:
                print(f"     post-pooler DECRYPT FAILED: {e}")
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        # Try direct decrypt first; only bootstrap if it fails.
        try:
            enc_logits = np.asarray(out_ct.decrypt(backend))[0].astype(np.float64)
        except Exception as e:
            print(f"     direct cls decrypt failed ({e}); bootstrapping then retry")
            new_cts = [backend.bootstrap(ct) for ct in out_ct.cts]
            out_ct = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=out_ct.hidden_dim)
            enc_logits = np.asarray(out_ct.decrypt(backend))[0].astype(np.float64)
        sample_wall = time.perf_counter() - t_sample

        # ── metrics ──────────────────────────────────────────────
        diff = pt_out - enc_logits
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        linf = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(pt_out), 1e-12))
        enc_pred = int(np.argmax(enc_logits))
        argmax_agree = enc_pred == pt_pred

        per_sample.append(
            {
                "ds_idx": int(ds_idx),
                "true_label": true_label,
                "pt_logits": pt_out.tolist(),
                "enc_logits": enc_logits.tolist(),
                "pt_pred": pt_pred,
                "enc_pred": enc_pred,
                "argmax_agree": argmax_agree,
                "L1": l1,
                "L2": l2,
                "Linf": linf,
                "relative_L2": rel,
                "wall_s": sample_wall,
            }
        )
        print(
            f"     pt_logits={pt_out.round(4).tolist()}  enc_logits={enc_logits.round(4).tolist()}\n"
            f"     L1={l1:.4f}  L2={l2:.4f}  Linf={linf:.4f}  relL2={100 * rel:.2f}%  "
            f"argmax_match={argmax_agree}  ({sample_wall:.1f}s)"
        )

    # ── summary ────────────────────────────────────────────────────
    arr = lambda k: np.asarray([s[k] for s in per_sample], dtype=np.float64)
    summary = {
        "model": args.model,
        "task": args.task,
        "checkpoint": str(ckpt),
        "seq_len": args.seq_len,
        "num_samples": args.num_samples,
        "ring_dim": args.ring_dim,
        "mult_depth": args.mult_depth,
        "num_slots": num_slots,
        "setup_sec": setup_time,
        "metrics": {
            "mean_L1": float(np.mean(arr("L1"))),
            "mean_L2": float(np.mean(arr("L2"))),
            "mean_Linf": float(np.mean(arr("Linf"))),
            "mean_relative_L2": float(np.mean(arr("relative_L2"))),
            "mean_wall_s": float(np.mean(arr("wall_s"))),
            "argmax_agreement": float(
                np.mean([s["argmax_agree"] for s in per_sample])
            ),
        },
        "per_sample": per_sample,
    }

    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.task}_e2e_validation.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print("\n" + "=" * 60)
    print(f"  E2E validation: {cfg['short']} on {args.task.upper()}  (N={args.num_samples})")
    print("=" * 60)
    print(f"  mean L1            : {m['mean_L1']:.6f}")
    print(f"  mean L2            : {m['mean_L2']:.6f}")
    print(f"  mean Linf          : {m['mean_Linf']:.6f}")
    print(f"  mean relative L2   : {100 * m['mean_relative_L2']:.3f}%")
    print(f"  argmax agreement   : {100 * m['argmax_agreement']:.1f}%")
    print(f"  mean wall / sample : {m['mean_wall_s']:.1f}s")
    print(f"\n  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
