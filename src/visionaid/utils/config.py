"""Loading of the YAML configuration into dataclasses (immutable per section)."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def _to_dataclass(name: str, data: Any) -> Any:
    """Recursively converts a dict into an instance of a generated dataclass."""
    if not isinstance(data, dict):
        return data
    fields = []
    values = {}
    for key, val in data.items():
        conv = _to_dataclass(f"{name}_{key}", val)
        fields.append((str(key), type(conv), dataclasses.field(default=None)))
        values[str(key)] = conv
    cls = dataclasses.make_dataclass(name.title().replace("_", ""), fields, frozen=False)
    return cls(**values)


def to_dict(cfg: Any) -> Any:
    """Inverse of `_to_dataclass`."""
    if dataclasses.is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    return copy.deepcopy(cfg)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_raw(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Loads the raw YAML, applies the overrides (nested dict or dotted keys)."""
    with open(path or DEFAULT_CONFIG) as fh:
        raw = yaml.safe_load(fh) or {}
    _apply_overrides(raw, overrides)
    query = (raw.get("target") or {}).get("query")
    if query:  # fixed prompt: the target card overrides the config, explicit overrides take precedence
        from visionaid.perception.target import apply_target

        apply_target(raw, query)
        _apply_overrides(raw, {k: v for k, v in (overrides or {}).items() if k != "target.query"})
    return raw


def _apply_overrides(raw: dict, overrides: dict | None) -> None:
    for key, val in (overrides or {}).items():
        if "." in key:
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
        elif isinstance(val, dict) and isinstance(raw.get(key), dict):
            _deep_update(raw[key], val)
        else:
            raw[key] = val


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Any:
    """Loads the configuration and returns it as nested dataclasses."""
    return _to_dataclass("config", load_raw(path, overrides))


def resolve_path(p: str | Path) -> Path:
    """Path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def length_to_m(value, name: str = "longueur") -> float | None:
    """Length entered by the user → metres. Accepts m, cm or mm without an explicit unit:
    > 3 is read as cm ("150" → 1.50 m), > 300 as mm. Human heights leave no
    ambiguity."""
    if value is None:
        return None
    v = float(value)
    if v > 300:
        out = v / 1000.0
    elif v > 3:
        out = v / 100.0
    else:
        out = v
    if out != v:
        import logging

        logging.getLogger("visionaid.config").warning(
            "%s = %g interpreted as %g m (unit inferred)", name, v, out
        )
    return out
