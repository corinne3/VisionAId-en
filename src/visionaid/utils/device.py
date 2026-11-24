"""Centralised selection of the torch device."""

from __future__ import annotations


def select_device(preference: str = "auto") -> str:
    """Returns 'mps' if available (or requested and available), otherwise 'cpu'."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if preference == "cpu":
        return "cpu"
    if preference in ("auto", "mps"):
        return "mps" if mps_ok else "cpu"
    return "cpu"
