"""Phase 16 (data): each real session in `data/sessions/` reads from start to end."""

import pytest

from visionaid.apps.session_info import list_sessions, summarize
from visionaid.io.session import SessionImu, SessionSource, load_session
from visionaid.utils.config import load_config, resolve_path

pytestmark = pytest.mark.data


def _sessions():
    return list_sessions(resolve_path(load_config().session.dir))


@pytest.mark.parametrize("sdir", _sessions() or [None], ids=lambda p: p.name if p else "aucune")
def test_real_session_reads_fully(sdir):
    if sdir is None:
        pytest.skip("No real session in data/sessions/ (see docs/checkpoint_enregistrements.md)")
    s = load_session(sdir)
    if not s.frames:  # recording started then stopped before any image
        pytest.skip(f"empty session (interrupted recording): {sdir.name}")
    n = sum(1 for _ in SessionSource(s, width=640))
    assert n == len(s.frames) > 0
    info = summarize(s)
    print(info)
    if s.meta.get("imu_available", True):
        assert len(SessionImu(s)) > 0, "session without IMU"
