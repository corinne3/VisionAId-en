"""Phase 17 (data): calibration on the real sessions `*calib_camera*` and `*calib_imu*`."""

import pytest

from visionaid.apps.session_info import list_sessions
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.data, pytest.mark.slow]


def _find(tag):
    ss = [p for p in list_sessions(resolve_path(load_config().session.dir)) if tag in p.name]
    if not ss:
        pytest.skip(f"No \"{tag}\" session (see docs/checkpoint_enregistrements.md)")
    return ss[-1]


def test_camera_calibration_real():
    from visionaid.apps.calibrate_phone import calibrate_camera_session

    res = calibrate_camera_session(load_config(), _find("calib_camera"))
    update_report("calibration", {"camera": res})
    assert res["rms_px"] < 0.5, res


def test_imu_calibration_real():
    from visionaid.apps.calibrate_phone import calibrate_imu_session

    res = calibrate_imu_session(load_config(), _find("calib_imu"))
    update_report("calibration", {"imu": res})
    assert res["correlation"] > 0.7, res
    assert res["units_ok"], res
