"""Recording of a phone session to disk (format described in `docs/protocol.md`)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from visionaid.bus.messages import ClockSync, ImuBatch


def new_session_dir(root: str | Path, name: str) -> Path:
    """Folder `<root>/<YYYYMMDD-HHMMSS>_<name>` (numeric suffix if already taken)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "session"
    base = Path(root) / f"{stamp}_{safe}"
    p, k = base, 1
    while p.exists():
        k += 1
        p = base.with_name(f"{base.name}-{k}")
    return p


class SessionRecorder:
    """Writes raw JPEG images, IMU, clock and metadata as they come."""

    def __init__(self, directory: str | Path, meta: dict[str, Any] | None = None) -> None:
        self.dir = Path(directory)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "calib").mkdir(exist_ok=True)
        self.meta: dict[str, Any] = {"t_start_pc": time.time(), "notes": ""}
        self.meta.update(meta or {})
        self._frames = open(self.dir / "frames.jsonl", "a")  # noqa: SIM115
        self._imu = open(self.dir / "imu.jsonl", "a")  # noqa: SIM115
        self._clock = open(self.dir / "clock.jsonl", "a")  # noqa: SIM115
        self.n_frames = 0
        self.n_calib = 0
        self.n_imu = 0
        self.closed = False
        self._write_meta()

    def _write_meta(self) -> None:
        m = dict(self.meta)
        m.update(n_frames=self.n_frames, n_calib=self.n_calib, n_imu=self.n_imu)
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False))
        tmp.replace(self.dir / "meta.json")

    def update_meta(self, **kw: Any) -> None:
        """Completes the metadata (device, camera…) and rewrites it."""
        self.meta.update(kw)
        self._write_meta()

    def frame(
        self, seq: int, t_phone: float, t_recv: float, w: int, h: int, jpeg: bytes, calib: bool
    ) -> None:
        """Records an image (JPEG as received)."""
        if calib:
            (self.dir / "calib" / f"{seq:06d}.jpg").write_bytes(jpeg)
            self.n_calib += 1
            return
        name = f"frames/{seq:06d}.jpg"
        (self.dir / name).write_bytes(jpeg)
        self._frames.write(
            json.dumps(
                {"seq": seq, "t_phone": t_phone, "t_recv": t_recv, "w": w, "h": h, "file": name}
            )
            + "\n"
        )
        self.n_frames += 1
        if self.n_frames % 30 == 0:
            self.flush()

    def imu(self, batch: ImuBatch) -> None:
        """Records an IMU batch (gyro in rad/s)."""
        for i in range(len(batch)):
            self._imu.write(
                json.dumps(
                    {
                        "t": float(batch.t[i]),
                        "gyro": [float(x) for x in batch.gyro[i]],
                        "accel": [float(x) for x in batch.accel[i]],
                        "t_recv": batch.t_recv,
                    }
                )
                + "\n"
            )
        self.n_imu += len(batch)

    def clock(self, c: ClockSync) -> None:
        """Records an estimate of the clock offset."""
        self._clock.write(
            json.dumps(
                {"t_recv": c.t, "offset": c.offset, "uncertainty": c.uncertainty, "rtt": c.rtt}
            )
            + "\n"
        )

    def flush(self) -> None:
        for fh in (self._frames, self._imu, self._clock):
            fh.flush()

    def close(self) -> None:
        """Closes the files and writes the final metadata."""
        if self.closed:
            return
        self.closed = True
        self.meta["t_end_pc"] = time.time()
        for fh in (self._frames, self._imu, self._clock):
            fh.close()
        self._write_meta()
