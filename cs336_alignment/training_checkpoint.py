"""Training checkpoints; independent of Modal and the GRPO implementation."""

import os
import pickle
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def load_training_checkpoint(directory: Path) -> dict[str, Any] | None:
    """Load the latest complete checkpoint on CPU, falling back after corruption.

    Only load trusted checkpoints: these files include pickled Python RNG state.
    """
    for path in sorted(directory.glob("checkpoint-*.pt"), reverse=True):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            print(f"Cannot read {path}: {error}; trying previous checkpoint", flush=True)
            continue
        if state.get("version") != 1:
            raise ValueError(f"Unsupported checkpoint version: {path}")
        print(f"Loaded checkpoint {path} (completed step {state['step']})", flush=True)
        return state
    if any(directory.glob("checkpoint-*.pt")):
        raise RuntimeError(f"No readable checkpoints in {directory}; refusing to restart silently")
    return None


def save_training_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cursor: int,
    train_examples: Sequence[Mapping[str, str]],
    rng: random.Random,
    config: dict[str, Any],
    run_id: str,
    final_metrics: dict[str, float] | None,
    commit: Callable[[], None] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint-{step:06d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    state = {
        "version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cursor": cursor,
        "train_examples": train_examples,
        "rng": rng.getstate(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": config,
        "run_id": run_id,
        "final_metrics": final_metrics,
    }
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    # Persist the new snapshot before deleting any older, known-good snapshot.
    if commit is not None:
        commit()
    old_paths = sorted(directory.glob("checkpoint-*.pt"))[:-2]
    for old_path in old_paths:
        old_path.unlink()
    if old_paths and commit is not None:
        commit()
    print(f"Saved checkpoint {path} (completed step {step})", flush=True)
    return path


def restore_training_checkpoint(
    state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> None:
    model.load_state_dict(state.pop("model"))
    optimizer.load_state_dict(state.pop("optimizer"))
    rng.setstate(state["rng"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
