"""Shim of the `clip` module expected by Ultralytics for YOLOE text prompts.

Ultralytics only uses `clip.clip.tokenize` (CLIP BPE tokenizer, context 77) before
calling the MobileCLIP TorchScript text encoder. The `clip` package is only distributed via
GitHub; an equivalent tokenizer based on `transformers.CLIPTokenizer` is provided here.
"""

from __future__ import annotations

import sys
import types

import torch

_TOKENIZER = None
CONTEXT_LENGTH = 77


def tokenize(texts, context_length: int = CONTEXT_LENGTH, truncate: bool = True) -> torch.Tensor:
    """Identical to `clip.tokenize`: int32 ids padded with zeros."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import CLIPTokenizer

        _TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if isinstance(texts, str):
        texts = [texts]
    out = torch.zeros(len(texts), context_length, dtype=torch.int32)
    for i, t in enumerate(texts):
        ids = _TOKENIZER(t, add_special_tokens=True)["input_ids"]
        if len(ids) > context_length:
            if not truncate:
                raise RuntimeError(f"Text too long for context {context_length}: {t}")
            ids = ids[:context_length]
            ids[-1] = _TOKENIZER.eos_token_id
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return out


def install() -> None:
    """Registers `clip` in sys.modules if the real package is missing."""
    try:
        import clip  # noqa: F401

        return
    except ImportError:
        pass
    mod = types.ModuleType("clip")
    sub = types.ModuleType("clip.clip")
    sub.tokenize = tokenize
    mod.clip = sub
    mod.tokenize = tokenize
    sys.modules["clip"] = mod
    sys.modules["clip.clip"] = sub
