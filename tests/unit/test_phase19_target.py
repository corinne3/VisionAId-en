"""Phase 19: fixed prompt → target card, generic ReID (without colour, without photo)."""

from __future__ import annotations

import numpy as np
import pytest

from visionaid.perception.reid import TargetReID, TextOnlyReID, make_color_model
from visionaid.perception.target import (
    UnknownTarget,
    content_words,
    load_config_for_target,
    load_sheets,
    resolve_target,
)
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "q,name",
    [
        ("Où est ma gourde ?", "gourde"),
        ("ou est ma gourde", "gourde"),
        ("ma gourde", "gourde"),
        ("GOURDE", "gourde"),
        ("trouve ma gourde stp", "gourde"),
        ("Où sont mes clés ?", "cles"),
        ("où sont mes clefs", "cles"),
        ("mes clés", "cles"),
        ("clé", "cles"),
        ("le trousseau", "cles"),
    ],
)
def test_phrase_variants(q, name):
    assert resolve_target(q).name == name


@pytest.mark.parametrize("q", ["où est mon téléphone", "", "???", "ma gourde et mes clés"])
def test_unknown_or_ambiguous_phrase_lists_targets(q):
    with pytest.raises(UnknownTarget) as e:
        resolve_target(q)
    assert "ma gourde" in str(e.value) and "mes clés" in str(e.value)


def test_content_words():
    assert content_words("Où sont mes clés ?") == ("cle",)
    assert content_words("où est ma gourde") == ("gourde",)


def test_sheets_load():
    names = {s.name for s in load_sheets()}
    assert {"gourde", "cles"} <= names


def test_config_for_keys_target():
    cfg = load_config_for_target("où sont mes clés")
    assert cfg.target.name == "cles" and cfg.target.query == "où sont mes clés"
    assert "keys" in cfg.detector.text_prompts
    assert cfg.reid.color_model == "none" and cfg.reid.w_color == 0.0
    assert cfg.paths.ref_glob == "cles_*.jpg"
    assert cfg.target.height_m is None
    # explicit overrides take precedence over the card
    cfg2 = load_config(overrides={"target.query": "mes clés", "detector.conf": 0.3})
    assert cfg2.detector.conf == 0.3 and cfg2.target.name == "cles"


def test_default_config_is_gourde_and_unchanged():
    cfg = load_config()
    assert cfg.target.name == "gourde" and cfg.target.query is None
    assert cfg.reid.color_model == "orange_gray"
    g = load_config_for_target("ma gourde")
    assert list(g.detector.text_prompts) == list(cfg.detector.text_prompts)
    assert g.reid.threshold_file == cfg.reid.threshold_file


def test_reid_without_color_model():
    from visionaid.perception.reid import StubEmbedder

    cfg = load_config_for_target("mes clés")
    assert make_color_model(cfg.reid) is None
    reid = TargetReID(cfg, embedder=StubEmbedder(), backend="stub")
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)
    reid.build_gallery([(crop, np.ones((40, 30), bool))], n_aug=2)
    s = reid.score_crops([(crop, np.ones((40, 30), bool))])
    assert 0.0 <= s[0] <= 1.0 and reid.color_score(crop, np.ones((40, 30), bool)) == 1.0


def test_text_only_reid_uses_detector_confidence():
    cfg = load_config_for_target("mes clés")
    r = TextOnlyReID(cfg)
    assert r.threshold == pytest.approx(0.25)
    sc = r.score_detections(None, np.zeros((2, 4)), None, np.array([0.1, 0.6]))
    np.testing.assert_allclose(sc, [0.1, 0.6])
