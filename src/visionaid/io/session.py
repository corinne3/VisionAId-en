"""Recorded phone sessions: reading, replay (images + IMU) and synthetic generation.

Format described in `docs/protocol.md`. `SessionSource` is a `FrameSource` like
`VideoFileSource`: it replays at real-time pace with `ReplaySource`, and `SessionImu` provides the
inertial samples between two images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from visionaid.bus.messages import ClockSync, Frame, ImuBatch
from visionaid.io.intrinsics import Intrinsics, make_intrinsics


@dataclass
class Session:
    """Content of a session loaded in memory (except the images, read on demand)."""

    dir: Path
    meta: dict
    frames: list[dict]  # lines of frames.jsonl, sorted by seq
    imu_t: np.ndarray  # (N,)
    imu_gyro: np.ndarray  # (N,3) rad/s, device frame
    imu_accel: np.ndarray  # (N,3) m/s²
    clock: list[ClockSync] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1]["t_phone"] - self.frames[0]["t_phone"]

    @property
    def fps(self) -> float:
        """Median frame rate."""
        if len(self.frames) < 2:
            return 15.0
        dt = np.diff([f["t_phone"] for f in self.frames])
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 15.0

    @property
    def calib_files(self) -> list[Path]:
        return sorted((self.dir / "calib").glob("*.jpg"))

    @property
    def camera_height_m(self) -> float | None:
        """Camera height declared at recording, in metres (cm accepted)."""
        from visionaid.utils.config import length_to_m

        return length_to_m(self.meta.get("camera_height_m"), "session camera height")

    def clock_offset(self) -> ClockSync | None:
        """Best clock estimate of the session (minimal uncertainty)."""
        if not self.clock:
            return None
        return min(self.clock, key=lambda c: c.uncertainty)


def is_session_dir(p: str | Path) -> bool:
    p = Path(p)
    return p.is_dir() and (p / "frames.jsonl").exists() and (p / "meta.json").exists()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # truncated last line (abrupt stop)
            continue
    return out


def load_session(directory: str | Path) -> Session:
    """Loads the metadata, image index, IMU and clock of a session."""
    d = Path(directory)
    if not is_session_dir(d):
        raise FileNotFoundError(f"not a session: {d} (frames.jsonl / meta.json missing)")
    meta = json.loads((d / "meta.json").read_text())
    frames = sorted(
        (f for f in _jsonl(d / "frames.jsonl") if (d / f["file"]).exists()),
        key=lambda f: f["seq"],
    )
    imu = _jsonl(d / "imu.jsonl")
    if imu:
        imu.sort(key=lambda s: s["t"])
        t = np.array([s["t"] for s in imu], float)
        g = np.array([s["gyro"] for s in imu], float).reshape(-1, 3)
        a = np.array([s["accel"] for s in imu], float).reshape(-1, 3)
        keep = np.r_[True, np.diff(t) > 0]  # possible duplicates
        t, g, a = t[keep], g[keep], a[keep]
    else:
        t, g, a = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    clock = [
        ClockSync(c["t_recv"], c["offset"], c["uncertainty"], c.get("rtt", 0.0))
        for c in _jsonl(d / "clock.jsonl")
        if c.get("uncertainty") is not None and np.isfinite(c["uncertainty"])
    ]
    return Session(d, meta, frames, t, g, a, clock)


def session_hash(directory: str | Path, extra: str = "") -> str:
    """Fingerprint of a session (image and IMU index) for the disk caches."""
    d = Path(directory)
    h = hashlib.sha1()
    for name in ("frames.jsonl", "imu.jsonl", "meta.json"):
        p = d / name
        if p.exists():
            h.update(p.read_bytes())
    h.update(extra.encode())
    return h.hexdigest()[:12]


def media_hash(path: str | Path, extra: str = "") -> str:
    """Fingerprint of a video or a session."""
    from visionaid.utils.report import file_hash

    return session_hash(path, extra) if is_session_dir(path) else file_hash(path, extra)


class SessionSource:
    """`FrameSource` reading the images of a session (t = phone clock, idx = seq)."""

    def __init__(
        self,
        session: Session | str | Path,
        width: int | None = 640,
        cfg_intrinsics=None,
        max_frames: int | None = None,
        start_frame: int = 0,
        cfg=None,
    ) -> None:
        import cv2

        self.session = session if isinstance(session, Session) else load_session(session)
        if not self.session.frames:
            raise ValueError(f"session without images: {self.session.dir}")
        self._cv2 = cv2
        f0 = self.session.frames[0]
        w0, h0 = int(f0["w"]), int(f0["h"])
        if width and w0 > width:
            self.out_w, self.out_h = int(width), int(round(h0 * width / w0))
        else:
            self.out_w, self.out_h = w0, h0
        if cfg is not None:  # device calibration if available
            from visionaid.calib.device import phone_intrinsics

            self._intr = phone_intrinsics(cfg, self.out_w, self.out_h)
        else:
            self._intr = make_intrinsics(
                cfg_intrinsics if cfg_intrinsics is not None else _DefaultIntr(),
                self.out_w,
                self.out_h,
            )
        self.max_frames = max_frames
        self._i = start_frame
        self._start = start_frame
        self.frame_count = len(self.session.frames)

    @classmethod
    def from_config(cls, cfg, path: str | Path, **kw) -> SessionSource:
        from visionaid.utils.config import resolve_path

        return cls(resolve_path(path), width=cfg.video.width, cfg=cfg, **kw)

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intr

    @property
    def fps(self) -> float:
        return self.session.fps

    @property
    def duration(self) -> float:
        return self.session.duration

    def read(self) -> Frame | None:
        cv2 = self._cv2
        if self.max_frames is not None and self._i - self._start >= self.max_frames:
            return None
        while self._i < len(self.session.frames):
            rec = self.session.frames[self._i]
            self._i += 1
            img = cv2.imread(str(self.session.dir / rec["file"]), cv2.IMREAD_COLOR)
            if img is None:  # corrupted JPEG: ignored
                continue
            if img.shape[1] != self.out_w or img.shape[0] != self.out_h:
                img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            t = float(rec["t_phone"])
            return Frame(t, int(rec["seq"]), img, self._intr.K, t, rec.get("t_recv"))
        return None

    def close(self) -> None:
        pass

    def __iter__(self):
        while (f := self.read()) is not None:
            yield f


class _DefaultIntr:
    hfov_deg = 69.0
    calibration_file = None


class SessionImu:
    """Access to the IMU samples of a session by time window (phone clock)."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def __len__(self) -> int:
        return len(self.s.imu_t)

    def window(self, t0: float | None, t1: float) -> ImuBatch:
        """Samples with t0 < t ≤ t1 (t0 = None: from the start)."""
        t = self.s.imu_t
        i0 = 0 if t0 is None else int(np.searchsorted(t, t0, side="right"))
        i1 = int(np.searchsorted(t, t1, side="right"))
        return ImuBatch(
            t[i0:i1].copy(), self.s.imu_gyro[i0:i1].copy(), self.s.imu_accel[i0:i1].copy()
        )

    def rate(self) -> float:
        """Median rate (Hz)."""
        if len(self.s.imu_t) < 2:
            return 0.0
        dt = np.diff(self.s.imu_t)
        return float(1.0 / np.median(dt[dt > 0]))


def open_media(cfg, path: str | Path | None = None, **kw):
    """Image source for a video (file) or a session (folder)."""
    from visionaid.io.sources import VideoFileSource
    from visionaid.utils.config import resolve_path

    p = resolve_path(path or cfg.paths.video)
    if is_session_dir(p):
        return SessionSource.from_config(cfg, p, **kw)
    return VideoFileSource.from_config(cfg, path=p, **kw)


def write_synthetic_session(
    directory: str | Path,
    images: list[np.ndarray],
    fps: float = 15.0,
    imu_fn=None,
    imu_hz: float = 60.0,
    t0: float = 1_700_000_000.0,
    meta: dict | None = None,
    calib_images: list[np.ndarray] | None = None,
) -> Path:
    """Writes a synthetic session (tests, phases 16-18) without going through the server.

    `imu_fn(t_rel) -> (gyro rad/s (3,), accel m/s² (3,))` in the device frame.
    """
    import cv2

    from visionaid.phone.recorder import SessionRecorder

    rec = SessionRecorder(directory, {"protocol": 1, "ua": "synthetic", **(meta or {})})
    h, w = images[0].shape[:2]
    rec.update_meta(camera={"w": w, "h": h}, gyro_units="deg/s")
    for k, img in enumerate(calib_images or []):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec.frame(10_000 + k, t0, t0, img.shape[1], img.shape[0], buf.tobytes(), True)
    for i, img in enumerate(images):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t = t0 + i / fps
        rec.frame(i + 1, t, t + 0.03, img.shape[1], img.shape[0], buf.tobytes(), False)
    if imu_fn is not None:
        n = int((len(images) / fps + 0.2) * imu_hz)
        ts = t0 - 0.1 + np.arange(n) / imu_hz
        g = np.zeros((n, 3))
        a = np.zeros((n, 3))
        for k, t in enumerate(ts):
            g[k], a[k] = imu_fn(t - t0)
        rec.imu(ImuBatch(ts, g, a, None))
    rec.clock(ClockSync(t0, 0.0, 0.002, 0.004))
    rec.close()
    return Path(directory)
