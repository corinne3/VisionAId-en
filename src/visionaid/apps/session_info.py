"""Summary of a phone session (duration, rates, gaps, clock) and viewing video.

Examples:
    uv run python -m visionaid.apps.session_info data/sessions/20261122-101500_gourde_piece
    uv run python -m visionaid.apps.session_info --all --video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionaid.io.session import Session, SessionImu, is_session_dir, load_session
from visionaid.utils.config import load_config, resolve_path


def summarize(s: Session) -> dict:
    """Statistics of a session."""
    t = np.array([f["t_phone"] for f in s.frames], float)
    dt = np.diff(t) if len(t) > 1 else np.zeros(0)
    gaps = int(np.sum(dt > 3.0 / max(s.fps, 1e-6))) if len(dt) else 0
    lat = np.array(
        [f["t_recv"] - f["t_phone"] for f in s.frames if f.get("t_recv") is not None], float
    )
    c = s.clock_offset()
    imu = SessionImu(s)
    cover = None
    if len(imu) and len(t):
        cover = {
            "imu_starts_before_first_frame_s": float(t[0] - s.imu_t[0]),
            "imu_ends_after_last_frame_s": float(s.imu_t[-1] - t[-1]),
        }
    out = {
        "session": s.name,
        "ua": s.meta.get("ua"),
        "target": s.meta.get("target"),
        "camera_height_m": s.meta.get("camera_height_m"),
        "n_frames": len(s.frames),
        "n_calib": len(s.calib_files),
        "duration_s": round(s.duration, 2),
        "fps_median": round(s.fps, 2),
        "fps_mean": round((len(t) - 1) / s.duration, 2) if s.duration > 0 else None,
        "gaps_over_3_periods": gaps,
        "max_gap_s": round(float(dt.max()), 3) if len(dt) else None,
        "imu_samples": len(imu),
        "imu_hz": round(imu.rate(), 1),
        "imu_coverage": cover,
        "clock_offset_s": c.offset if c else None,
        "clock_uncertainty_ms": round(c.uncertainty * 1e3, 2) if c else None,
        "frame_image_size": [s.frames[0]["w"], s.frames[0]["h"]] if s.frames else None,
    }
    if c is not None and len(lat):
        age = lat - c.offset  # reception (computer) − capture (brought to the computer clock)
        out["frame_age_ms_p50"] = round(float(np.percentile(age, 50)) * 1e3, 1)
        out["frame_age_ms_p95"] = round(float(np.percentile(age, 95)) * 1e3, 1)
    out["notes"] = s.meta.get("notes", "")
    return out


def write_video(s: Session, path: Path | None = None) -> Path:
    """Viewing `video.mp4` (median rate, unmodified images)."""
    import cv2

    path = path or (s.dir / "video.mp4")
    vw = None
    size = None
    for f in s.frames:
        img = cv2.imread(str(s.dir / f["file"]))
        if img is None:
            continue
        if vw is None:
            size = (img.shape[1], img.shape[0])
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), s.fps, size)
        if (img.shape[1], img.shape[0]) != size:  # resolution changed during the session
            img = cv2.resize(img, size)
        vw.write(img)
    if vw is not None:
        vw.release()
    return path


def list_sessions(root: str | Path) -> list[Path]:
    r = Path(root)
    return sorted(p for p in r.glob("*") if is_session_dir(p)) if r.exists() else []


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true", help="all the sessions in session.dir")
    ap.add_argument("--video", action="store_true", help="generates video.mp4")
    args = ap.parse_args(argv)
    cfg = load_config()
    dirs = [Path(p) for p in args.sessions]
    if args.all or not dirs:
        dirs += list_sessions(resolve_path(cfg.session.dir))
    out = []
    for d in dirs:
        s = load_session(d)
        info = summarize(s)
        if args.video:
            info["video"] = str(write_video(s))
        out.append(info)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    if not out:
        print(f"No session in {resolve_path(cfg.session.dir)}")
    return out


if __name__ == "__main__":
    main()
