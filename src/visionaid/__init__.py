"""VisionAId: binaural sound guidance towards an object, for visually impaired people."""

import os

# An expired HF token present on the machine makes downloads of public models fail:
# no implicit token is sent (the models used are public).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import warnings as _warnings

# numpy 2.x + Accelerate (macOS arm64) emits spurious RuntimeWarnings on matmul.
_warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
