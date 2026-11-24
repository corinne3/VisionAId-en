"""Phase 19 (data): key detection on the real `cles_piece` session (informative)."""

import pytest

from visionaid.apps.session_info import list_sessions
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.data, pytest.mark.slow]


@pytest.mark.timeout(1800)
def test_keys_detection_on_session():
    from visionaid.perception.pipeline import run_video_perception
    from visionaid.perception.target import load_config_for_target

    ss = [p for p in list_sessions(resolve_path(load_config().session.dir)) if "cles" in p.name]
    if not ss:
        pytest.skip("No \"cles_piece\" session (see docs/checkpoint_enregistrements.md)")
    cfg = load_config_for_target("où sont mes clés")
    refs = sorted(resolve_path(cfg.paths.ref_dir).glob(cfg.paths.ref_glob))
    dets, info = run_video_perception(cfg, ss[-1])
    confirmed = sum(d.target_idx >= 0 for d in dets) / max(len(dets), 1)
    seen_any = sum(len(d.boxes) > 0 for d in dets) / max(len(dets), 1)
    res = {
        "session": ss[-1].name,
        "mode": "photos" if refs else "text only",
        "n_refs": len(refs),
        "frames": len(dets),
        "confirmed_fraction": round(confirmed, 3),
        "any_detection_fraction": round(seen_any, 3),
    }
    update_report("keys", res)
    print(res)
    # informative: no blocking threshold (difficult object), only "the pipeline runs"
    assert len(dets) > 0
