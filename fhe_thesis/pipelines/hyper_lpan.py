"""HyPER-LPAN unified training pipeline.

Replaces the previous tangle of three scripts
(``finetune_lpan_staged.py`` / ``finetune_linear_mixing_progressive.py`` /
``finetune_hybrid_progressive.py``) with one resumable, YAML-driven pipeline.

Stages
------
* **A — LPAN baseline** (delegated to ``run_staged_lpan.py`` if absent)
* **B — Mid-layer Quad replacement** (default L4–L7)
* **C — Early-layer Linear-Mixing replacement** (default L0–L3)
* **D — Global fine-tune** (unfreeze entire encoder + classifier)

Each stage writes a ``.done`` marker to its output directory, so re-running
the pipeline is a no-op for completed stages — eliminating the previous
``--resume-skip-layers`` ergonomics nightmare.

The four stages share one frozen LPAN teacher and one student model; only
the student state evolves between stages.

Example
-------
::

    from fhe_thesis.pipelines import HyperLPANConfig, HyperLPANPipeline

    cfg = HyperLPANConfig.from_yaml("configs/hyper_lpan/sst2_base.yaml")
    pipeline = HyperLPANPipeline(cfg)
    results = pipeline.run()
    print(results["final_accuracy"])
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR
from fhe_thesis.models.hybrid_attention import (
    freeze_for_global_finetune,
    freeze_for_progressive_hybrid,
    summarize_attention_types,
)
from fhe_thesis.models.linear_mixing import replace_attention_with_linear_mixing
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.models.quad_attention import replace_attention_with_quad
from fhe_thesis.tasks import get_task
from fhe_thesis.training.checkpoints import (
    find_lpan_checkpoint,
    load_state_into,
    mark_stage_done,
    stage_done,
)
from fhe_thesis.training.trainer import (
    attn_distill_and_eval,
    compute_metrics_for_task,
    load_glue_dataset,
)

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class HyperLPANConfig:
    """Single source of truth for a HyPER-LPAN training run.

    All fields have sensible defaults matching the validated SST-2 base run
    (90.83% final, -0.45% from LPAN).  Override per-task via YAML.
    """

    # Task & model
    model: str = "base"
    task: str = "sst2"
    max_seq_len: int = 64
    seed: int = 42

    # Layer composition
    linear_mixing_layers: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3]
    )
    quad_attention_layers: List[int] = field(
        default_factory=lambda: [4, 5, 6, 7]
    )
    quad_num_heads: Optional[int] = None  # None → use model default

    # Stage A: LPAN baseline
    lpan_checkpoint: Optional[str] = None  # override; else canonical path
    auto_train_lpan: bool = False  # if True and missing, invoke run_staged_lpan.py

    # Stages B & C: progressive replacement
    epochs_per_layer: int = 4
    lr: float = 8e-5
    batch_size: int = 32
    gamma: float = 4.0
    gamma_decay: bool = True
    stage_order: str = "quad_first"  # quad_first | linear_first
    lr_schedule: str = "constant_with_warmup"

    # Stage D: global fine-tune
    final_epochs: int = 4
    global_gamma: float = 2.0
    global_lr_div: float = 3.0  # global_lr = lr / global_lr_div

    # Control flags
    skip_stage_b: bool = False
    skip_stage_c: bool = False
    skip_stage_d: bool = False

    # Output
    output_dir: Optional[str] = None  # default: MULTI_MODEL_DIR/<task>/<model>/hyper_lpan

    # ── Validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        overlap = set(self.linear_mixing_layers) & set(self.quad_attention_layers)
        if overlap:
            raise ValueError(
                f"Layers {sorted(overlap)} appear in both "
                f"linear_mixing_layers and quad_attention_layers"
            )
        if self.stage_order not in ("quad_first", "linear_first"):
            raise ValueError(
                f"stage_order must be 'quad_first' or 'linear_first', "
                f"got {self.stage_order!r}"
            )
        if self.model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {self.model!r}; valid: {sorted(MODEL_REGISTRY)}"
            )

    # ── (De)serialisation ────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HyperLPANConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load configs from YAML; install with "
                "`pip install pyyaml`"
            ) from exc
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return MODEL_REGISTRY[self.model]["layers"]

    @property
    def lpan_layers(self) -> List[int]:
        replaced = set(self.linear_mixing_layers) | set(self.quad_attention_layers)
        return sorted(set(range(self.num_layers)) - replaced)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return MULTI_MODEL_DIR / self.task / self.model / "hyper_lpan"


# ─── Pipeline ───────────────────────────────────────────────────────────────


class HyperLPANPipeline:
    """Orchestrates the four-stage HyPER-LPAN training trajectory.

    The pipeline is **resumable by default**: each stage that completes
    writes a ``.done`` file to its output directory and is silently skipped
    on subsequent invocations.  To force a re-run, delete the corresponding
    ``.done`` marker (or the entire stage directory).
    """

    def __init__(
        self,
        cfg: HyperLPANConfig,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_cfg = get_task(cfg.task)
        self.model_meta = MODEL_REGISTRY[cfg.model]
        self.output_dir = cfg.resolved_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State populated by run()
        self.lpan_acc: Optional[float] = None
        self.results: Dict[str, Any] = {
            "config": cfg.to_dict(),
            "model_short": self.model_meta["short"],
            "stages": {},
            "per_layer_results": [],
        }

    # ── Public entry point ───────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute (or resume) all stages and return the results dict."""
        set_seed(self.cfg.seed)
        self._print_banner()

        lpan_ckpt = self._stage_a_lpan()

        # Stages B & C order is determined by stage_order
        student, teacher, train_ds, eval_ds = self._setup_models_and_data(lpan_ckpt)

        self._stage_progressive(
            student, teacher, train_ds, eval_ds,
            stage_label="stage_b" if self.cfg.stage_order == "quad_first" else "stage_c",
            layer_type=("quad" if self.cfg.stage_order == "quad_first"
                        else "linear_mixing"),
            layer_indices=(self.cfg.quad_attention_layers
                           if self.cfg.stage_order == "quad_first"
                           else self.cfg.linear_mixing_layers),
            skip=self.cfg.skip_stage_b if self.cfg.stage_order == "quad_first"
                 else self.cfg.skip_stage_c,
        )
        self._stage_progressive(
            student, teacher, train_ds, eval_ds,
            stage_label="stage_c" if self.cfg.stage_order == "quad_first" else "stage_b",
            layer_type=("linear_mixing" if self.cfg.stage_order == "quad_first"
                        else "quad"),
            layer_indices=(self.cfg.linear_mixing_layers
                           if self.cfg.stage_order == "quad_first"
                           else self.cfg.quad_attention_layers),
            skip=self.cfg.skip_stage_c if self.cfg.stage_order == "quad_first"
                 else self.cfg.skip_stage_b,
        )

        final_acc = self._stage_d_global(student, teacher, train_ds, eval_ds)

        return self._finalise(student, final_acc)

    # ── Stage A: LPAN baseline ───────────────────────────────────────────

    def _stage_a_lpan(self) -> Path:
        """Resolve (or auto-train) the LPAN baseline checkpoint."""
        try:
            ckpt = find_lpan_checkpoint(
                self.cfg.model, self.cfg.task, override=self.cfg.lpan_checkpoint
            )
            print(f"\n[Stage A] LPAN checkpoint found: {ckpt}")
            self.results["stages"]["A"] = {"status": "found", "path": str(ckpt)}
            return ckpt
        except FileNotFoundError:
            if not self.cfg.auto_train_lpan:
                raise
            print(f"\n[Stage A] LPAN checkpoint missing — auto-training "
                  f"via run_staged_lpan.py ...")
            cmd = [
                sys.executable, "run_staged_lpan.py",
                "--model", self.cfg.model,
                "--task", self.cfg.task,
                "--seed", str(self.cfg.seed),
            ]
            subprocess.run(cmd, check=True)
            ckpt = find_lpan_checkpoint(
                self.cfg.model, self.cfg.task, override=self.cfg.lpan_checkpoint
            )
            self.results["stages"]["A"] = {"status": "trained", "path": str(ckpt)}
            return ckpt

    # ── Setup: load student, teacher, data; eval baseline ────────────────

    def _setup_models_and_data(
        self, lpan_ckpt: Path
    ) -> Tuple[torch.nn.Module, torch.nn.Module, Any, Any]:
        print("\n[Setup] Loading student & teacher models, tokenising data ...")
        # Student
        student = load_lpan_model(
            self.cfg.model, lpan_ckpt, device="cpu",
            profile_samples=200, degree=8,
        )
        # Teacher (frozen, on device)
        teacher = load_lpan_model(
            self.cfg.model, lpan_ckpt, device=self.device,
            profile_samples=200, degree=8,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Data
        tokenizer = AutoTokenizer.from_pretrained(self.model_meta["name"])
        train_ds, eval_dict = load_glue_dataset(
            self.task_cfg, tokenizer, max_length=self.cfg.max_seq_len,
        )
        # Pick first eval split for best-model selection (e.g. "validation")
        eval_ds = next(iter(eval_dict.values()))

        # Evaluate LPAN baseline once
        student.to(self.device)
        baseline_dir = self.output_dir / "_baseline_eval"
        baseline_trainer = Trainer(
            model=student,
            args=TrainingArguments(
                output_dir=str(baseline_dir),
                per_device_eval_batch_size=self.cfg.batch_size * 2,
                report_to="none", disable_tqdm=True,
            ),
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics_for_task(self.task_cfg),
        )
        baseline = baseline_trainer.evaluate()
        primary_metric = self.task_cfg.metric_for_best_model
        self.lpan_acc = float(baseline[primary_metric])
        print(f"  LPAN baseline {self.task_cfg.metric}: "
              f"{self.lpan_acc:.4f} ({self.lpan_acc:.2%})")
        shutil.rmtree(baseline_dir, ignore_errors=True)

        self.results["lpan_baseline"] = self.lpan_acc

        # Track replaced layers across all stages so freeze_for_progressive
        # sees the cumulative set
        self._replaced_so_far: List[int] = []

        return student, teacher, train_ds, eval_ds

    # ── Stages B / C: progressive layer replacement ──────────────────────

    def _stage_progressive(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        train_ds: Any,
        eval_ds: Any,
        stage_label: str,
        layer_type: str,
        layer_indices: List[int],
        skip: bool,
    ) -> None:
        if skip:
            print(f"\n[{stage_label}] SKIPPED via config")
            self.results["stages"][stage_label] = {"status": "skipped_config"}
            return
        if not layer_indices:
            print(f"\n[{stage_label}] SKIPPED (no layers specified)")
            self.results["stages"][stage_label] = {"status": "skipped_empty"}
            return

        stage_dir = self.output_dir / f"{stage_label}_{layer_type}"

        if stage_done(stage_dir):
            print(f"\n[{stage_label}] ✓ already complete (.done marker present)")
            # Replay: re-apply replacements + load final-layer checkpoint
            self._replay_stage(student, layer_type, layer_indices, stage_dir)
            self.results["stages"][stage_label] = {"status": "resumed"}
            return

        print(f"\n[{stage_label}] {layer_type} replacement, "
              f"{len(layer_indices)} layer(s) ...")
        layer_results = self._run_progressive_loop(
            student, teacher, train_ds, eval_ds,
            stage_dir=stage_dir,
            layer_type=layer_type,
            layer_indices=sorted(layer_indices),
        )
        self.results["per_layer_results"].extend(layer_results)
        self.results["stages"][stage_label] = {
            "status": "completed",
            "layers_trained": [r["layer"] for r in layer_results],
        }
        mark_stage_done(stage_dir)

    def _replay_stage(
        self,
        model: torch.nn.Module,
        layer_type: str,
        layer_indices: List[int],
        stage_dir: Path,
    ) -> None:
        """Re-architect ``model`` for a previously-completed stage and load
        its final per-layer checkpoint."""
        model.cpu()
        sorted_idx = sorted(layer_indices)
        for li in sorted_idx:
            self._replace_layer(model, li, layer_type)
            self._replaced_so_far.append(li)
        last_ckpt = stage_dir / f"layer_{sorted_idx[-1]}" / "best_model"
        if last_ckpt.exists():
            print(f"  Loading checkpoint: {last_ckpt}")
            load_state_into(model, last_ckpt, self.device)
        else:
            raise FileNotFoundError(
                f"Stage marked done but final checkpoint missing: {last_ckpt}"
            )

    def _run_progressive_loop(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        train_ds: Any,
        eval_ds: Any,
        stage_dir: Path,
        layer_type: str,
        layer_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """Replace each layer in turn and KD-fine-tune, with per-layer resume."""
        stage_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for li in layer_indices:
            layer_dir = stage_dir / f"layer_{li}"
            layer_done = layer_dir / ".done"

            student.cpu()
            self._replace_layer(student, li, layer_type)
            self._replaced_so_far.append(li)

            if layer_done.exists():
                # Load the layer's best checkpoint and skip retraining
                best = layer_dir / "best_model"
                print(f"  [layer {li}] ✓ resuming from {best}")
                load_state_into(student, best, self.device)
                # Need an eval to record accuracy
                acc = self._quick_eval(student, eval_ds)
                results.append({
                    "stage_dir": str(stage_dir.name),
                    "layer": li, "layer_type": layer_type,
                    "accuracy": acc, "drop_from_lpan": (self.lpan_acc - acc) * 100,
                    "status": "resumed",
                })
                continue

            lr_scale = 1.0 + 0.5 * (li / max(1, self.cfg.num_layers - 1))
            layer_lr = self.cfg.lr * lr_scale
            layer_gamma = self._compute_layer_gamma(li)

            # Co-adaptation: unfreeze attention + FFNs for ALL replaced layers
            trainable = freeze_for_progressive_hybrid(
                student,
                replaced_layers=self._replaced_so_far,
                unfreeze_all_replaced_ffns=True,
            )
            total = sum(p.numel() for p in student.parameters())
            print(f"\n  ── L{li} ({layer_type}) "
                  f"lr={layer_lr:.2e}  γ={layer_gamma:.2f}  "
                  f"trainable={trainable:,}/{total:,} "
                  f"({100*trainable/total:.1f}%) ──")

            student.to(self.device)
            result = attn_distill_and_eval(
                student_model=student,
                teacher_model=teacher,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                output_dir=str(layer_dir),
                epochs=self.cfg.epochs_per_layer,
                batch_size=self.cfg.batch_size,
                lr=layer_lr,
                label=f"L{li} {layer_type}",
                use_fp16=torch.cuda.is_available(),
                max_grad_norm=1.0,
                alpha=1.0, beta=0.0, gamma=layer_gamma,
                seed=self.cfg.seed,
                lr_scheduler_type=self.cfg.lr_schedule,
            )
            acc = float(result["accuracy"])
            drop = (self.lpan_acc - acc) * 100
            print(f"  → L{li} {layer_type}: acc={acc:.4f}  (drop {drop:+.2f}%)")

            # Cleanup intermediate trainer checkpoints; keep only best_model
            for item in layer_dir.iterdir():
                if item.is_dir() and item.name != "best_model":
                    shutil.rmtree(item, ignore_errors=True)
            layer_done.write_text("ok\n")

            results.append({
                "stage_dir": str(stage_dir.name),
                "layer": li, "layer_type": layer_type,
                "accuracy": acc, "drop_from_lpan": drop,
                "epochs": self.cfg.epochs_per_layer,
                "lr": layer_lr, "gamma": layer_gamma,
                "status": "trained",
            })

        return results

    # ── Stage D: global fine-tune ────────────────────────────────────────

    def _stage_d_global(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        train_ds: Any,
        eval_ds: Any,
    ) -> float:
        if self.cfg.skip_stage_d:
            print("\n[Stage D] SKIPPED via config")
            acc = self._quick_eval(student, eval_ds)
            self.results["stages"]["D"] = {"status": "skipped_config"}
            return acc

        stage_dir = self.output_dir / "stage_d_global"
        if stage_done(stage_dir):
            print(f"\n[Stage D] ✓ already complete; loading checkpoint")
            best = stage_dir / "best_model"
            load_state_into(student, best, self.device)
            acc = self._quick_eval(student, eval_ds)
            self.results["stages"]["D"] = {"status": "resumed", "accuracy": acc}
            return acc

        global_lr = self.cfg.lr / self.cfg.global_lr_div
        print(f"\n[Stage D] Global fine-tune: "
              f"epochs={self.cfg.final_epochs}  lr={global_lr:.2e}  "
              f"γ={self.cfg.global_gamma}")

        trainable = freeze_for_global_finetune(student)
        total = sum(p.numel() for p in student.parameters())
        print(f"  trainable={trainable:,}/{total:,} "
              f"({100*trainable/total:.1f}%)")

        result = attn_distill_and_eval(
            student_model=student,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=str(stage_dir),
            epochs=self.cfg.final_epochs,
            batch_size=self.cfg.batch_size,
            lr=global_lr,
            label="Global fine-tune",
            use_fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            alpha=1.0, beta=0.0, gamma=self.cfg.global_gamma,
            seed=self.cfg.seed,
            lr_scheduler_type=self.cfg.lr_schedule,
        )
        acc = float(result["accuracy"])
        print(f"  → Global FT: acc={acc:.4f}")
        for item in stage_dir.iterdir():
            if item.is_dir() and item.name != "best_model":
                shutil.rmtree(item, ignore_errors=True)
        mark_stage_done(stage_dir)
        self.results["stages"]["D"] = {"status": "completed", "accuracy": acc}
        return acc

    # ── Final save ───────────────────────────────────────────────────────

    def _finalise(
        self,
        student: torch.nn.Module,
        final_acc: float,
    ) -> Dict[str, Any]:
        final_dir = self.output_dir / "best_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        for p in student.parameters():
            p.data = p.data.contiguous()
        student.save_pretrained(str(final_dir))

        self.results["final_accuracy"] = final_acc
        self.results["accuracy_drop_from_lpan"] = (
            (self.lpan_acc - final_acc) * 100 if self.lpan_acc is not None else None
        )
        self.results["final_layer_types"] = summarize_attention_types(student)

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  HyPER-LPAN complete — final {self.task_cfg.metric}: "
              f"{final_acc:.4f}")
        print(f"  LPAN baseline: {self.lpan_acc:.4f}  "
              f"(drop: {self.results['accuracy_drop_from_lpan']:+.2f}%)")
        print(f"  Saved model: {final_dir}")
        print(f"  Saved results: {results_path}")
        print(f"{'='*70}\n")
        return self.results

    # ── Helpers ──────────────────────────────────────────────────────────

    def _replace_layer(self, model: torch.nn.Module, li: int, layer_type: str) -> None:
        if layer_type == "linear_mixing":
            replace_attention_with_linear_mixing(
                model, max_seq_len=self.cfg.max_seq_len,
                layer_indices=[li], num_heads=None,
            )
        elif layer_type == "quad":
            replace_attention_with_quad(
                model, layer_indices=[li],
                num_heads=self.cfg.quad_num_heads,
                init_from_original=True,
            )
        else:
            raise ValueError(f"Unknown layer_type {layer_type!r}")

    def _compute_layer_gamma(self, li: int) -> float:
        if not self.cfg.gamma_decay:
            return self.cfg.gamma
        third = self.cfg.num_layers // 3
        if li < third:
            return self.cfg.gamma
        elif li < 2 * third:
            return self.cfg.gamma * 0.5
        return self.cfg.gamma * 0.25

    def _quick_eval(self, model: torch.nn.Module, eval_ds: Any) -> float:
        tmp = self.output_dir / "_eval_tmp"
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(tmp),
                per_device_eval_batch_size=self.cfg.batch_size * 2,
                report_to="none", disable_tqdm=True,
            ),
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics_for_task(self.task_cfg),
        )
        out = trainer.evaluate()
        shutil.rmtree(tmp, ignore_errors=True)
        return float(out[self.task_cfg.metric_for_best_model])

    def _print_banner(self) -> None:
        c = self.cfg
        print(f"\n{'='*70}")
        print(f"  HyPER-LPAN Pipeline")
        print(f"{'='*70}")
        print(f"  Model: {self.model_meta['short']} ({c.model})  "
              f"Task: {c.task} ({self.task_cfg.description})")
        print(f"  Layer composition:")
        print(f"    Linear mixing : {c.linear_mixing_layers}")
        print(f"    2Quad attn    : {c.quad_attention_layers}")
        print(f"    LPAN (kept)   : {c.lpan_layers}")
        print(f"  max_seq_len={c.max_seq_len}  batch={c.batch_size}  "
              f"epochs/layer={c.epochs_per_layer}  final_epochs={c.final_epochs}")
        print(f"  lr={c.lr}  γ={c.gamma}  decay={c.gamma_decay}  "
              f"global_γ={c.global_gamma}  schedule={c.lr_schedule}")
        print(f"  stage_order={c.stage_order}  device={self.device}")
        print(f"  output_dir={self.output_dir}")
        print(f"{'='*70}")
