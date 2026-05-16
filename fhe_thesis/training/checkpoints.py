"""Checkpoint discovery, loading, and resume helpers.

Centralises the boilerplate used by the staged LPAN and Synthesizer-LPAN
training/export pipeline.

Public API
----------
- ``find_lpan_checkpoint(model, task, override=None)``
    Resolve the canonical LPAN checkpoint directory for a (model, task).

- ``load_state_into(model, ckpt_dir, device)``
    Load a HuggingFace-style checkpoint (``model.safetensors`` preferred,
    ``pytorch_model.bin`` fallback) directly into an already-architected
    model. Required when the model has custom polynomial or Synthesizer
    modules so ``AutoModel.from_pretrained`` cannot rebuild the architecture.

- ``stage_done(stage_dir)`` / ``mark_stage_done(stage_dir)``
    Resume markers: a stage with a ``.done`` file in its output directory
    is treated as already completed and is skipped on subsequent runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn

from fhe_thesis.config import MULTI_MODEL_DIR, SYNTHESIZER_LPAN_DIR

PathLike = Union[str, Path]


def _candidate_lpan_checkpoint_dirs(model: str, task: str) -> Iterable[Path]:
    """Yield candidate Stage-3 LPAN checkpoint directories in priority order.

    The repo currently contains both task-first and model-first historical
    layouts.  New code should not assume only one of them exists.
    """
    task = task.lower()
    model = model.lower()
    yield SYNTHESIZER_LPAN_DIR / task / model / "staged_lpan_s3_ln_kd" / "best_model"
    yield MULTI_MODEL_DIR / task / model / "staged_lpan_final" / "best_model"
    yield MULTI_MODEL_DIR / task / model / "staged_lpan_s3_ln_kd" / "best_model"
    yield MULTI_MODEL_DIR / model / task / "staged_lpan_final" / "best_model"
    yield MULTI_MODEL_DIR / model / task / "staged_lpan_s3_ln_kd" / "best_model"
    yield MULTI_MODEL_DIR / model / "staged_lpan_final" / "best_model"
    yield MULTI_MODEL_DIR / model / "staged_lpan_s3_ln_kd" / "best_model"


def load_checkpoint_state_dict(ckpt_dir: PathLike) -> Dict[str, torch.Tensor]:
    """Load a HuggingFace-style checkpoint directory into a CPU state dict."""
    ckpt_dir = Path(ckpt_dir)
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        import safetensors.torch

        return safetensors.torch.load_file(str(safetensors_path), device="cpu")

    bin_path = ckpt_dir / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir} "
            f"(looked for model.safetensors and pytorch_model.bin)"
        )
    return torch.load(str(bin_path), map_location="cpu")


def find_lpan_checkpoint(
    model: str,
    task: str,
    override: Optional[PathLike] = None,
) -> Path:
    """Resolve the LPAN ``best_model`` directory for a (model, task) pair.

    Parameters
    ----------
    model : str
        Model key (``tiny``/``mini``/``small``/``base``).
    task : str
        GLUE task key (``sst2``/``mrpc``/``qnli``/``rte``).
    override : str or Path, optional
        Explicit path that bypasses the canonical location.

    Returns
    -------
    Path
        Existing directory containing ``model.safetensors`` (or ``.bin``).

    Raises
    ------
    FileNotFoundError
        If the resolved directory does not exist.
    """
    if override is not None:
        path = Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"LPAN checkpoint override not found at {path}."
        )

    candidates = list(_candidate_lpan_checkpoint_dirs(model, task))
    for path in candidates:
        if path.exists():
            return path

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"LPAN checkpoint not found for model={model}, task={task}. Tried:\n"
        f"  - {tried}\n"
        f"Pass --lpan-checkpoint to override."
    )


def load_state_into(model: nn.Module, ckpt_dir: PathLike, device: str) -> None:
    """Load weights from ``ckpt_dir`` into ``model`` in-place (strict).

    The model must already have the correct architecture (all attention
    replacements applied) before calling this so state-dict keys match.
    """
    state_dict = load_checkpoint_state_dict(ckpt_dir)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)


def stage_done(stage_dir: PathLike) -> bool:
    """Return True iff ``stage_dir/.done`` exists (stage already completed)."""
    return (Path(stage_dir) / ".done").exists()


def mark_stage_done(stage_dir: PathLike) -> None:
    """Create ``stage_dir/.done`` to mark this stage as complete."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / ".done").write_text("ok\n")
