"""
Checkpoint I/O shared by every training stage.

Saves are atomic: written to a temp file, then os.replace'd onto the final name. A watcher
syncing the directory therefore never copies a half-written file, and a crash mid-save
leaves the previous checkpoint intact.

Optimizer state travels with the weights. Adam's moments take hundreds of steps to rebuild,
so weights-only restarts spike the loss and discard the momentum the run had earned.
"""
import glob
import os
import re

import torch


def step_of(path: str) -> int:
    m = re.search(r"(\d+)(?=\.pt$)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def save_checkpoint(path: str, model, optimizer, step: int, scaler=None,
                    extra: dict | None = None) -> str:
    """Write a resumable checkpoint atomically. Returns the final path."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    # -- an inactive scaler has no state worth carrying; bf16 and fp32 runs never have one
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, model, optimizer=None, scaler=None, map_location: str = "cpu"):
    """
    Restore into an already-constructed model and optimizer. Returns (step, payload).

    Loading to CPU holds one copy of the state dict rather than two on the GPU; the
    optimizer then moves its own state to wherever its parameters already live.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    # -- checkpoints written before 'step' was recorded fall back to the filename
    step = ckpt.get("step")
    return (step if step is not None else step_of(path)), ckpt


def latest_checkpoint(directory: str, prefix: str) -> str | None:
    paths = glob.glob(os.path.join(directory, f"{prefix}_*.pt"))
    return max(paths, key=step_of) if paths else None


def resolve_resume(arg: str | None, directory: str, prefix: str) -> str | None:
    # -- 'auto' is the safe default for an unattended restart; an explicit path replays one step
    if arg is None:
        return None
    return latest_checkpoint(directory, prefix) if arg == "auto" else arg


def rotate_checkpoints(directory: str, prefix: str, keep: int) -> None:
    # -- keep <= 0 retains everything
    if keep <= 0:
        return
    paths = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.pt")), key=step_of)
    for old in paths[:-keep]:
        os.remove(old)
