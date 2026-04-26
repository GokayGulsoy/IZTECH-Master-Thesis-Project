#!/usr/bin/env python3
"""Stage-4 range-aware fine-tune for FHE readiness.

The Stage-1/2/3 LPAN curriculum trains polynomial activations whose
plaintext forward pass clamps inputs to ``[a, b]`` before evaluating
the Chebyshev polynomial.  CKKS cannot clamp (no sign function), so
any input outside ``[a, b]`` blows up under the polynomial's
extrapolation tail.  This stage adds an explicit range-penalty term
to the loss so the network *learns* to keep activations inside their
intervals — making the (already-converged) plaintext clamp a no-op
and FHE inference numerically faithful.

Loss
----
    L = L_task + λ · Σ_{ℓ, op}  E_x [ ReLU(|x_ℓ,op| − r_ℓ,op)² ]

where for each polynomial submodule we use the standardised input the
plaintext clamp truncates (raw x for GELU, shifted scores for Softmax,
``var + eps`` for LayerNorm) and ``r`` is the half-width of ``[a, b]``
re-centred at 0.

Usage
-----
    python run_stage4_range_aware.py --model tiny --task sst2
    python run_stage4_range_aware.py --model tiny --task all
    python run_stage4_range_aware.py --model all --task all --lambda 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from fhe_thesis.config import MAX_INTERVALS, MODEL_REGISTRY, MULTI_MODEL_DIR
from fhe_thesis.models.activations import (
    PerHeadPolynomialSoftmax,
    PolynomialGELU,
    PolynomialLayerNorm,
    PolynomialSoftmax,
)
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.tasks import GLUE_TASKS, get_task
from fhe_thesis.training.trainer import (
    NaNSafeTrainer,
    compute_metrics_for_task,
    load_glue_dataset,
)


POLY_TYPES = (
    PolynomialGELU,
    PolynomialSoftmax,
    PerHeadPolynomialSoftmax,
    PolynomialLayerNorm,
)


def _ckpt_dir(model_key: str, task_name: str) -> Path:
    base = MULTI_MODEL_DIR / model_key
    if task_name != "sst2":
        base = base / task_name
    return base


def _interval_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The exact tensor the plaintext ``clamp`` operates on."""
    if isinstance(module, PolynomialGELU):
        return x
    if isinstance(module, (PolynomialSoftmax, PerHeadPolynomialSoftmax)):
        shifted = x - x.max(dim=-1, keepdim=True).values
        return shifted[shifted > -1e3]
    if isinstance(module, PolynomialLayerNorm):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return var + module.eps
    raise TypeError(f"unsupported polynomial module: {type(module)}")


def _module_op_name(module: torch.nn.Module) -> str:
    if isinstance(module, PolynomialGELU):
        return "GELU"
    if isinstance(module, (PolynomialSoftmax, PerHeadPolynomialSoftmax)):
        return "Softmax"
    if isinstance(module, PolynomialLayerNorm):
        return "LN"
    raise TypeError(type(module))


@torch.no_grad()
def _empirical_intervals(
    model: torch.nn.Module,
    tokenized_dataset,
    *,
    n_batches: int,
    batch_size: int,
    device: str,
    margin: float = 0.10,
) -> dict[str, tuple[float, float]]:
    """Profile the loaded LPAN model and compute tight ``[a, b]`` per module.

    ``tokenized_dataset`` is the HuggingFace Dataset returned from
    ``load_glue_dataset`` (already has ``input_ids``, ``attention_mask``,
    ``labels`` columns formatted as torch tensors).  Returns
    ``{qualified_name: (a, b)}`` clamped to ``MAX_INTERVALS``.
    """
    accum: dict[str, list[tuple[float, float]]] = {}
    targets: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, POLY_TYPES):
            targets.append((name, module))
            accum[name] = []

    handles = []
    for name, module in targets:
        def make_hook(n=name, m=module):
            def hook(_mod, inputs, _out):
                x_int = _interval_input(m, inputs[0])
                if x_int.numel() == 0:
                    return
                accum[n].append((float(x_int.min().item()), float(x_int.max().item())))
            return hook
        handles.append(module.register_forward_hook(make_hook()))

    n_take = min(n_batches * batch_size, len(tokenized_dataset))
    sub = tokenized_dataset.select(range(n_take))
    for i in range(0, n_take, batch_size):
        rows = sub[i : i + batch_size]
        batch = {
            "input_ids": rows["input_ids"].to(device),
            "attention_mask": rows["attention_mask"].to(device),
        }
        model(**batch)

    for h in handles:
        h.remove()

    intervals: dict[str, tuple[float, float]] = {}
    for name, module in targets:
        if not accum[name]:
            continue
        lo = min(p[0] for p in accum[name])
        hi = max(p[1] for p in accum[name])
        op = _module_op_name(module)
        width = hi - lo
        m_ = margin * width if width > 0 else max(abs(hi), abs(lo)) * margin
        a, b = lo - m_, hi + m_
        if op == "Softmax":
            b = min(b, 0.5)
        if op == "LN":
            a = max(0.01, a)
        lo_max, hi_max = MAX_INTERVALS[op]
        a = max(a, lo_max)
        b = min(b, hi_max)
        intervals[name] = (a, b)
    return intervals


def _attach_range_penalty(model: torch.nn.Module) -> tuple[list, dict[str, Any]]:
    """Hook every poly module to accumulate ``ReLU(|x_std| − 1)²``.

    We standardise into the polynomial's own ``[-1, 1]`` Chebyshev
    domain so a single ``λ`` works across operations with different
    interval widths.  The squared-relu of the over-range part is
    stashed into a shared list that the trainer's ``compute_loss``
    drains every step.
    """
    accum: dict[str, Any] = {"terms": [], "n_oor": 0, "n_total": 0}
    handles = []
    for _name, module in model.named_modules():
        if not isinstance(module, POLY_TYPES):
            continue

        def make_hook(mod_ref):
            def hook(_mod, inputs, _output):
                x = inputs[0]
                x_int = _interval_input(mod_ref, x)
                a, b = float(mod_ref.a), float(mod_ref.b)
                x_std = (2.0 * x_int - (a + b)) / (b - a)
                # Only the part outside [-1, 1] contributes.
                over = torch.relu(x_std.abs() - 1.0)
                # Mean-square so different ops contribute on the same scale.
                term = (over ** 2).mean()
                accum["terms"].append(term)
                with torch.no_grad():
                    accum["n_oor"] += int((over > 0).sum().item())
                    accum["n_total"] += x_std.numel()
            return hook

        handles.append(module.register_forward_hook(make_hook(module)))
    return handles, accum


class RangeAwareTrainer(NaNSafeTrainer):
    """``NaNSafeTrainer`` + range penalty + optional KD distillation.

    Loss
    ----
        L = α · L_CE + (1−α) · T² · KL(student/T ‖ teacher/T)
            + λ · Σ ReLU(|x_std| − 1)²

    The teacher is the *unmutated* LPAN-plaintext checkpoint loaded
    before Stage-4 widens intervals or attaches the range penalty.  Its
    soft labels encode the behaviour the plaintext clamp+polynomial
    achieves; distilling against it lets the student recover most of
    the accuracy lost when Stage-4 forces activations inside [a, b]
    naturally instead of via clamp.
    """

    def __init__(
        self,
        *args,
        range_accum: dict,
        range_lambda: float,
        teacher_model: torch.nn.Module | None = None,
        kd_alpha: float = 0.5,
        kd_temperature: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._range_accum = range_accum
        self._range_lambda = range_lambda
        self._teacher = teacher_model
        if self._teacher is not None:
            self._teacher.eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        self._kd_alpha = kd_alpha
        self._kd_T = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Hooks fire during the model forward pass below; clear first
        # so the accumulator only contains terms from *this* batch.
        self._range_accum["terms"].clear()
        outputs = model(**inputs)
        task_loss = outputs.loss

        if self._teacher is not None:
            # Lazily move teacher onto whatever device the trainer ends
            # up using (Trainer auto-detects CUDA regardless of --device).
            input_device = next(iter(inputs.values())).device
            if next(self._teacher.parameters()).device != input_device:
                self._teacher.to(input_device)
            with torch.no_grad():
                t_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                t_logits = self._teacher(**t_inputs).logits
            T = self._kd_T
            kd = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(outputs.logits / T, dim=-1),
                torch.nn.functional.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            base_loss = self._kd_alpha * task_loss + (1.0 - self._kd_alpha) * kd
        else:
            base_loss = task_loss

        terms = self._range_accum["terms"]
        if terms:
            penalty = torch.stack(terms).sum()
            loss = base_loss + self._range_lambda * penalty
        else:
            loss = base_loss
        return (loss, outputs) if return_outputs else loss


def _run_one(
    model_key: str,
    task_name: str,
    *,
    epochs: int,
    lr: float,
    range_lambda: float,
    seed: int,
    device: str,
    keep_intervals: bool = False,
    kd_alpha: float = 0.5,
    kd_temperature: float = 4.0,
    distill: bool = True,
    profile_on: str = "val",
    margin: float = 0.30,
) -> dict[str, Any]:
    cfg = MODEL_REGISTRY[model_key]
    base = _ckpt_dir(model_key, task_name)
    src_ckpt = base / "staged_lpan_final" / "best_model"
    if not src_ckpt.exists():
        return {"status": "missing", "checkpoint": str(src_ckpt)}

    out_dir = base / "stage4_range_aware"
    out_best = out_dir / "best_model"
    if out_best.exists():
        return {"status": "already_done", "output": str(out_best)}

    task = get_task(task_name)
    print(f"\n[stage4] {model_key} × {task_name}  λ={range_lambda}  "
          f"distill={distill}  src={src_ckpt}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the LPAN-plaintext teacher FIRST, before any interval mutation
    # on the student.  The teacher keeps the original Stage-3 intervals
    # (with the plaintext clamp active) so its logits encode the
    # behaviour we want to distill into the FHE-deployable student.
    teacher_model = None
    if distill:
        teacher_model = load_lpan_model(
            model_key, src_ckpt, num_labels=task.num_labels, device=device
        )

    model = load_lpan_model(
        model_key, src_ckpt, num_labels=task.num_labels, device=device
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    train_ds, eval_dict = load_glue_dataset(task, tokenizer, max_length=128)
    eval_ds = next(iter(eval_dict.values()))

    # ------------------------------------------------------------------
    # Empirical interval re-profiling.
    #
    # Stage-3 left activations drifting outside the base-BERT-profiled
    # ``[a, b]``.  We recompute tight intervals from the *trained model*
    # forward pass on real task data, add a 10% safety margin, and pin
    # them as ``module.a, module.b``.  The Stage-3 learnable
    # coefficients carry over and re-adapt during the brief fine-tune
    # below.  The frozen intervals get persisted so the audit and the
    # FHE encryption use the same ``[a, b]`` the model was trained on.
    # ------------------------------------------------------------------
    print("  Re-profiling activations from trained model on real data...")
    profile_ds = eval_ds if profile_on == "val" else train_ds
    print(f"    profile_on={profile_on!r}  margin={margin:.2f}  source={'validation' if profile_on=='val' else 'training'} split")
    intervals = _empirical_intervals(
        model,
        profile_ds,
        n_batches=16,
        batch_size=cfg["batch_size"],
        device=device,
        margin=margin,
    )
    if keep_intervals:
        # Keep the Stage-3 intervals exactly — the range penalty will
        # push activations *inward* to live inside them.  Avoids the
        # approximation-error penalty from widening the polynomial
        # domain.
        intervals = {
            name: (float(module.a), float(module.b))
            for name, module in model.named_modules()
            if isinstance(module, POLY_TYPES)
        }
    n_set = 0
    for name, module in model.named_modules():
        if isinstance(module, POLY_TYPES) and name in intervals:
            a, b = intervals[name]
            tag = "kept" if keep_intervals else "now"
            print(f"    {name:35s}  was=[{module.a:6.2f},{module.b:6.2f}]  {tag}=[{a:6.2f},{b:6.2f}]")
            module.a = float(a)
            module.b = float(b)
            n_set += 1
    print(f"  → updated {n_set} polynomial modules with empirical intervals")

    handles, accum = _attach_range_penalty(model)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=task.metric_for_best_model,
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
        fp16=False,
    )
    trainer = RangeAwareTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_for_task(task),
        range_accum=accum,
        range_lambda=range_lambda,
        teacher_model=teacher_model,
        kd_alpha=kd_alpha,
        kd_temperature=kd_temperature,
    )

    t0 = time.time()
    trainer.train()
    train_wall = time.time() - t0

    # Final evaluation + save best as `best_model/`.
    metrics = trainer.evaluate()
    out_best.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_best))
    tokenizer.save_pretrained(str(out_best))

    for h in handles:
        h.remove()

    # Re-profile after training to capture any small drift, widen if
    # needed, and persist the final intervals next to the checkpoint.
    model_device = str(next(model.parameters()).device)
    final_intervals = _empirical_intervals(
        model,
        profile_ds,
        n_batches=16,
        batch_size=cfg["batch_size"],
        device=model_device,
        margin=margin,
    )
    # Take the wider of pre-train and post-train per module so the
    # interval covers both training-time and inference-time activations.
    merged: dict[str, tuple[float, float]] = {}
    for name, (a0, b0) in intervals.items():
        a1, b1 = final_intervals.get(name, (a0, b0))
        merged[name] = (min(a0, a1), max(b0, b1))
    (out_best / "intervals.json").write_text(
        json.dumps({k: list(v) for k, v in merged.items()}, indent=2)
    )

    n_total = max(accum["n_total"], 1)
    final_oor = accum["n_oor"] / n_total
    summary = {
        "status": "ok",
        "src": str(src_ckpt),
        "output": str(out_best),
        "train_wall_s": train_wall,
        "epochs": epochs,
        "lr": lr,
        "range_lambda": range_lambda,
        "final_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "final_oor_frac_train": final_oor,
    }
    (out_dir / "stage4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [done] {model_key} × {task_name}  "
          f"wall={train_wall:.0f}s  metric={metrics.get(task.metric_for_best_model):.4f}  "
          f"final_oor_frac={100*final_oor:.4f}%")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-4 range-aware FHE-ready fine-tune")
    parser.add_argument(
        "--model", nargs="*", default=list(MODEL_REGISTRY),
        help="Which model(s) (default: all)",
    )
    parser.add_argument(
        "--task", nargs="*", default=list(GLUE_TASKS),
        help="Which GLUE task(s) (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Small LR — we are nudging an already-converged checkpoint.")
    parser.add_argument(
        "--lambda", dest="range_lambda", type=float, default=5e-2,
        help="Weight on the range-penalty term (default: 5e-2 with --keep-intervals).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--keep-intervals", action=argparse.BooleanOptionalAction, default=True,
        help="Keep Stage-3 intervals and push activations inward via the "
             "range penalty (default: True). Pass --no-keep-intervals to "
             "widen [a, b] to the empirical range instead.",
    )
    parser.add_argument(
        "--distill", action=argparse.BooleanOptionalAction, default=True,
        help="Use the unmutated LPAN-plaintext checkpoint as KD teacher "
             "(default: True). Pass --no-distill to disable.",
    )
    parser.add_argument(
        "--kd-alpha", type=float, default=0.5,
        help="Blend factor between task CE and KD loss: "
             "alpha · CE + (1 − alpha) · T² · KL.",
    )
    parser.add_argument(
        "--kd-temperature", type=float, default=4.0,
        help="Temperature for the KD soft labels.",
    )
    parser.add_argument(
        "--audit-summary", default="results/fhe_readiness/summary.json",
        help="If present, only run combos flagged ``needs_stage4`` in this audit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run all selected combos regardless of audit flags.",
    )
    parser.add_argument(
        "--profile-on",
        choices=["train", "val"],
        default="val",
        help="Which split to profile activations on for empirical interval fitting. "
             "'val' (default) is required for FHE inference correctness, since the FHE "
             "backend evaluates polynomials on validation activations whose distribution "
             "is wider than training activations.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.30,
        help="Safety margin to widen empirical [min, max] intervals on each side. "
             "Default 0.30 (30%%) — large enough to cover unseen test activations.",
    )
    args = parser.parse_args()

    audit_flagged: set[tuple[str, str]] = set()
    if not args.force:
        ap = Path(args.audit_summary)
        if ap.exists():
            try:
                summary = json.loads(ap.read_text())
                for c in summary.get("combos", []):
                    if c.get("needs_stage4"):
                        audit_flagged.add((c["model"], c["task"]))
                print(f"[stage4] audit at {ap} flags {len(audit_flagged)} combos")
            except Exception as e:
                print(f"[stage4] WARNING: cannot read audit {ap}: {e}", file=sys.stderr)

    results = []
    for task in args.task:
        for model in args.model:
            if not args.force and audit_flagged and (model, task) not in audit_flagged:
                print(f"[skip] {model} × {task} — audit says FHE-ready already")
                results.append({"model": model, "task": task, "status": "skipped_audit_clean"})
                continue
            r = _run_one(
                model, task,
                epochs=args.epochs, lr=args.lr,
                range_lambda=args.range_lambda,
                seed=args.seed, device=args.device,
                keep_intervals=args.keep_intervals,
                distill=args.distill,
                kd_alpha=args.kd_alpha,
                kd_temperature=args.kd_temperature,
                profile_on=args.profile_on,
                margin=args.margin,
            )
            r.update({"model": model, "task": task})
            results.append(r)

    out = Path("results/stage4_range_aware/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[stage4] summary → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
