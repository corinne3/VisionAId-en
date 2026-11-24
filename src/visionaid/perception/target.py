"""Fixed prompt → target card (`config/targets/<name>.yaml`), without a language model.

The phrase is normalised (lower case, accents, punctuation, stop words) then compared with the trigger
phrases of each card: exact match first, then inclusion of the object name. The chosen
card overrides the config (detection prompts, reference photos, colour model, threshold).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from visionaid.utils.config import _deep_update, resolve_path

STOPWORDS = {
    "ou", "est", "sont", "ma", "mon", "mes", "la", "le", "les", "l", "trouve", "trouver",
    "cherche", "chercher", "moi", "je", "veux", "s", "il", "te", "plait", "stp", "svp", "d",
    "de", "du", "des", "un", "une", "a",
}  # fmt: skip


class UnknownTarget(ValueError):
    """No card matches the phrase."""


@dataclass
class TargetSheet:
    """Target card loaded from YAML."""

    name: str
    label: str
    phrases: list[str]
    overrides: dict
    path: Path
    keys: set[str] = field(default_factory=set)
    spoken: str = ""  # name spoken in the announcements ("la gourde")
    plural: bool = False


def normalize(text: str) -> str:
    """Lower case, without accents or punctuation, single spaces."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def content_words(text: str) -> tuple[str, ...]:
    """Meaningful words (without stop words), approximately singular."""
    out = []
    for w in normalize(text).split():
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith(("s", "x")):
            w = w[:-1]
        out.append(w)
    return tuple(out)


def spoken_from_label(label: str) -> str:
    """« ma gourde » → « la gourde », « mes clés » → « les clés » (default spoken name)."""
    w = label.strip().split(" ", 1)
    art = {"ma": "la", "mon": "le", "mes": "les", "ta": "la", "ton": "le", "tes": "les"}
    if len(w) == 2 and w[0].lower() in art:
        return f"{art[w[0].lower()]} {w[1]}"
    return label


def load_sheets(directory: str | Path | None = None) -> list[TargetSheet]:
    """All the cards in the folder, sorted by name."""
    d = resolve_path(directory or "config/targets")
    sheets = []
    for p in sorted(Path(d).glob("*.yaml")):
        y = yaml.safe_load(p.read_text()) or {}
        name = str(y.get("name") or p.stem)
        phrases = [str(x) for x in y.get("phrases", [])] + [name]
        keys = {" ".join(content_words(x)) for x in phrases}
        keys.discard("")
        label = str(y.get("label", name))
        sheets.append(
            TargetSheet(
                name,
                label,
                phrases,
                y.get("overrides", {}),
                p,
                keys,
                str(y.get("spoken") or spoken_from_label(label)),
                bool(y.get("plural", False)),
            )
        )
    return sheets


def resolve_target(query: str, directory: str | Path | None = None) -> TargetSheet:
    """Card matching the phrase; `UnknownTarget` (with the list of targets) otherwise."""
    sheets = load_sheets(directory)
    key = " ".join(content_words(query))
    if key:
        exact = [s for s in sheets if key in s.keys]
        if len(exact) == 1:
            return exact[0]
        words = set(key.split())
        partial = [s for s in sheets if any(set(k.split()) <= words for k in s.keys)]
        if len(partial) == 1:
            return partial[0]
    known = ", ".join(f"« {s.label} »" for s in sheets) or "none"
    raise UnknownTarget(f"unknown target for « {query} ». Available targets: {known}.")


def apply_target(raw: dict, query: str | None) -> dict:
    """Applies the card matching `query` (or `raw['target']['query']`) to a raw config."""
    q = query if query is not None else (raw.get("target") or {}).get("query")
    if not q:
        return raw
    sheet = resolve_target(q, (raw.get("target") or {}).get("sheets_dir"))
    _deep_update(raw, yaml.safe_load(yaml.safe_dump(sheet.overrides)) or {})
    raw.setdefault("target", {})
    raw["target"]["query"] = q
    raw["target"]["name"] = sheet.name
    raw["target"]["label"] = sheet.label
    raw["target"]["spoken"] = sheet.spoken
    raw["target"]["plural"] = sheet.plural
    return raw


def load_config_for_target(query: str | None, path=None, overrides: dict | None = None):
    """Config with the target card applied (explicit `overrides` take precedence over the card)."""
    from visionaid.utils.config import load_config

    ov = dict(overrides or {})
    ov["target.query"] = query
    return load_config(path, ov)


# ------------------------------------------------------------------ commandes vocales
STOP_WORDS = {"stop", "arrete", "arreter", "arretes", "termine", "fini", "annule", "annuler"}
STOP_PHRASES = ("c est bon", "j ai trouve", "laisse tomber")


def parse_command(texts, directory: str | Path | None = None) -> tuple[str, TargetSheet | None]:
    """Recognised phrase(s) → (action, card). Action: `search`, `stop` or `unknown`.

    Several possible transcriptions (speech recognition alternatives): the first one
    that designates a card wins. « Stop », « arrête », « j'ai trouvé » (stop, I found it): end of the search.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [t for t in texts if t and t.strip()]
    for t in texts:
        try:
            return "search", resolve_target(t, directory)
        except UnknownTarget:
            pass
    for t in texts:
        n = normalize(t)
        if set(n.split()) & STOP_WORDS or any(p in n for p in STOP_PHRASES):
            return "stop", None
    return "unknown", None


def target_raw(raw: dict, query: str) -> dict:
    """Raw config for the target `query`, applied on the base config (without a card).

    `raw["_raw_base"]` (set by run_live) is the config without a target card: this avoids an
    override of the previous card (key detection threshold…) staying in place.
    """
    import copy

    base = raw.get("_raw_base") or raw
    out = copy.deepcopy({k: v for k, v in base.items() if k != "_raw_base"})
    for k, v in raw.items():  # runtime settings ("_phone", "_stub_models"…)
        if k.startswith("_") and k != "_raw_base":
            out[k] = v
    return apply_target(out, query)


def sheet_for_config(raw: dict) -> TargetSheet | None:
    """Card of the configured target (fixed prompt), or None."""
    q = (raw.get("target") or {}).get("query")
    if not q:
        return None
    try:
        return resolve_target(q, (raw.get("target") or {}).get("sheets_dir"))
    except UnknownTarget:
        return None


def known_targets_phrase(directory: str | Path | None = None) -> str:
    """« ta gourde ou tes clés » (your bottle or your keys): what the user can ask for."""
    art = {"ma": "ta", "mon": "ton", "mes": "tes"}
    names = []
    for s in load_sheets(directory):
        w = s.label.split(" ", 1)
        names.append(f"{art.get(w[0], w[0])} {w[1]}" if len(w) == 2 else s.label)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " ou " + names[-1]
