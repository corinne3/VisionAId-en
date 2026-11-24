"""Speech synthesis on the computer side (macOS `say`): session replay and offline rendering.

Live, the sentences are spoken by the phone (Web Speech, `web/audio.js`), in the headset.
Here: `MacSpeaker` speaks them on the Mac speaker (replay of a session without a phone) and
`synth` renders them as samples for the WAV track of the videos.
"""

from __future__ import annotations

import hashlib
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from visionaid.utils.logging import get_logger

log = get_logger("tts")


def available() -> bool:
    return shutil.which("say") is not None


def synth(
    text: str, fs: int, voice: str = "Thomas", rate_wpm: int = 190, cache_dir: Path | None = None
) -> np.ndarray | None:
    """Sentence → mono float64 samples at `fs` (None if `say` is missing or fails)."""
    if not text or not available():
        return None
    import soundfile as sf

    key = hashlib.sha1(f"{voice}|{rate_wpm}|{fs}|{text}".encode()).hexdigest()[:16]
    cpath = Path(cache_dir) / f"{key}.wav" if cache_dir is not None else None
    if cpath is not None and cpath.exists():
        x, _ = sf.read(str(cpath), always_2d=True)
        return x.mean(axis=1)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.aiff"
        try:
            subprocess.run(
                ["say", "-v", voice, "-r", str(int(rate_wpm)), "-o", str(out), text],
                check=True,
                timeout=60,
                capture_output=True,
            )
            x, sr = sf.read(str(out), always_2d=True)
        except Exception as e:  # noqa: BLE001 - voice missing, format not read…
            log.warning("Speech synthesis impossible (%s): %s", voice, e)
            return None
    x = x.mean(axis=1)
    if sr != fs:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(fs))
        x = resample_poly(x, fs // g, sr // g)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x = x * (0.5 / peak)  # peak −6 dBFS, roughly the level of the guidance sounds
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(cpath), x, fs, subtype="PCM_16")
    return x


class MacSpeaker:
    """Speaks the sentences one by one on the Mac speaker (dedicated thread)."""

    def __init__(self, voice: str = "Thomas", rate_wpm: int = 190) -> None:
        self.voice = voice
        self.rate = int(rate_wpm)
        self.q: queue.Queue = queue.Queue()
        self.ok = available()
        self.n = 0
        if self.ok:
            threading.Thread(target=self._run, name="mac-tts", daemon=True).start()
        else:
            log.warning("`say` missing: sentences displayed only")

    def say(self, text: str) -> None:
        log.info("Voix : %s", text)
        self.n += 1
        if self.ok:
            self.q.put(text)

    def _run(self) -> None:
        while True:
            text = self.q.get()
            if text is None:
                return
            try:
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), text], timeout=60, check=False
                )
            except Exception as e:  # noqa: BLE001
                log.warning("say : %s", e)

    def close(self) -> None:
        self.q.put(None)
