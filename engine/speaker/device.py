"""Runtime device helpers for speaker embedding engines."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("Matrix_Speaker")


def resolve_speaker_device() -> str:
    """Resolve SPEAKER_DEVICE to a torch/modelscope device string."""
    requested = (os.environ.get("SPEAKER_DEVICE") or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        logger.warning("[Speaker] Invalid SPEAKER_DEVICE=%r, fallback to auto", requested)
        requested = "auto"
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        logger.debug("[Speaker] torch device probe failed: %s", exc)
    if requested == "cuda":
        logger.warning("[Speaker] SPEAKER_DEVICE=cuda requested but torch cuda is unavailable; fallback to cpu")
    return "cpu"


def move_tensor_to_device(tensor, device: str):
    """Move a torch tensor when possible; fake test tensors may not implement .to()."""
    if device == "cuda" and hasattr(tensor, "to"):
        return tensor.to(device)
    return tensor
