"""Consistent logging for the whole project."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a logger configured only once (stderr output)."""
    root = logging.getLogger("visionaid")
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        root.addHandler(h)
        root.setLevel(level)
        root.propagate = False
    return logging.getLogger(f"visionaid.{name}")
