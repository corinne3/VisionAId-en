"""Phone calibration from recorded sessions (see docs/checkpoint_enregistrements.md).

Examples:
    uv run python -m visionaid.apps.calibrate_phone --camera data/sessions/…_calib_camera
    uv run python -m visionaid.apps.calibrate_phone --imu data/sessions/…_calib_imu
Results: config/calib/<phone.device>_camera.yaml and _imu.yaml (loaded automatically).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionaid.calib.camera import calibrate_images, square_size_from_notes
from visionaid.calib.device import camera_calib_path, imu_calib_path
from visionaid.calib.imu_cam import calibrate_imu_cam, visual_rates
from visionaid.io.session import SessionSource, load_session
from visionaid.utils.config import load_config
from visionaid.utils.logging import get_logger
from visionaid.utils.report import update_report

log = get_logger("calibrate_phone")


def calibrate_camera_session(
    cfg, session_dir, pattern=(9, 6), square_mm: float | None = None, out: Path | None = None
) -> dict:
    """Calibrates the camera on the `calib/` views of a session; writes the device YAML."""
    s = load_session(session_dir)
    files = s.calib_files
    if not files:
        raise FileNotFoundError(f"no calibration view in {s.dir / 'calib'}")
    sq = square_mm / 1000.0 if square_mm else square_size_from_notes(s.meta.get("notes", ""))
    if sq is None:
        log.warning(
            "Unknown square size (notes \"case = xx mm\"): 25 mm assumed; K stays right"
        )
        sq = 0.025
    cal = calibrate_images(files, pattern, sq)
    out = out or camera_calib_path(cfg)
    hfov_cfg = float(cfg.intrinsics.hfov_deg)
    extra = {
        "device": str(cfg.phone.device),
        "session": s.name,
        "ua": s.meta.get("ua"),
        "pattern": list(pattern),
        "square_m": sq,
        "hfov_config_deg": hfov_cfg,
        "rejected": cal.rejected,
    }
    cal.to_yaml(out, extra)
    summ = {
        "file": str(out),
        "rms_px": round(cal.rms_px, 3),
        "n_views": cal.n_views,
        "n_rejected": len(cal.rejected),
        "hfov_long_side_deg": round(cal.hfov_long_side_deg, 2),
        "hfov_config_deg": hfov_cfg,
        "size": [cal.width, cal.height],
        "dist": [round(float(x), 4) for x in cal.dist.ravel()[:5]],
    }
    if abs(cal.hfov_long_side_deg - hfov_cfg) > 8:
        log.warning(
            "Calibrated field of view %.1f° far from the default value %.1f°",
            cal.hfov_long_side_deg,
            hfov_cfg,
        )
    return summ


def calibrate_imu_session(cfg, session_dir, out: Path | None = None, every: int = 1) -> dict:
    """Calibrates IMU ↔ camera bias, offset and rotation; writes the device YAML."""
    s = load_session(session_dir)
    if len(s.imu_t) < 50:
        raise ValueError(f"session without usable IMU: {s.dir}")
    src = SessionSource.from_config(cfg, s.dir)
    imgs, ts = [], []
    for f in src:
        imgs.append(f.image)
        ts.append(f.t)
    K = src.intrinsics.K
    t0, t1, w = visual_rates(imgs, np.array(ts), K, every)
    cal = calibrate_imu_cam(
        s.imu_t, s.imu_gyro, s.imu_accel, t0, t1, w, np.asarray(cfg.body.R_cam_imu, float)
    )
    out = out or imu_calib_path(cfg)
    cal.to_yaml(out, {"device": str(cfg.phone.device), "session": s.name, "ua": s.meta.get("ua")})
    return {
        "file": str(out),
        "time_offset_ms": round(cal.time_offset_s * 1e3, 1),
        "gyro_bias_deg_s": [round(float(np.degrees(x)), 3) for x in cal.gyro_bias],
        "prior_angle_deg": round(cal.prior_angle_deg, 2),
        "gyro_scale": round(cal.gyro_scale, 4),
        "units_ok": cal.units_ok,
        "correlation": round(cal.correlation, 3),
        "n_pairs": cal.n_pairs,
        "residual_deg_s": round(cal.residual_deg_s, 2),
        "static_s": round(cal.static_s, 1),
        "accel_sign": cal.accel_sign,
        "accel_tilt_deg": round(cal.accel_tilt_deg, 1),
        "notes": cal.notes,
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", default=None, help="session with calibration views (checkerboard)")
    ap.add_argument("--imu", default=None, help="IMU calibration session (rotations)")
    ap.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    ap.add_argument("--square-mm", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    res = {}
    if args.camera:
        pat = tuple(int(x) for x in args.pattern.lower().split("x"))
        res["camera"] = calibrate_camera_session(cfg, args.camera, pat, args.square_mm)
    if args.imu:
        cfg = load_config()  # reload: the camera calibration may just have been written
        res["imu"] = calibrate_imu_session(cfg, args.imu)
    if not res:
        ap.print_help()
        return res
    update_report("calibration", res)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
