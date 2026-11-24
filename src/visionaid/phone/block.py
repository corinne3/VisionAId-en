"""Bus source block fed by the phone: latest image + IMU accumulated since the previous call."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from visionaid.bus.messages import ClockSync, Frame, ImuBatch
from visionaid.phone.server import PhoneServer
from visionaid.utils.config import _to_dataclass


def merge_batches(batches: list[ImuBatch]) -> ImuBatch | None:
    """Concatenates IMU batches (sorted by time), or None if empty."""
    batches = [b for b in batches if len(b)]
    if not batches:
        return None
    t = np.concatenate([b.t for b in batches])
    o = np.argsort(t, kind="stable")
    return ImuBatch(
        t[o],
        np.concatenate([b.gyro for b in batches])[o],
        np.concatenate([b.accel for b in batches])[o],
        batches[-1].t_recv,
    )


class PhoneInbox:
    """Thread-safe buffer between the server (callbacks) and a synchronous consumer.

    Only keeps the **latest** image (no delay build-up), but all the IMU.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._imu: list[ImuBatch] = []
        self._clock: ClockSync | None = None
        self.n_frames_in = 0
        self.n_frames_out = 0
        self.n_overwritten = 0

    def put_frame(self, f: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self.n_overwritten += 1
            self._frame = f
            self.n_frames_in += 1
            self._cond.notify_all()

    def put_imu(self, b: ImuBatch) -> None:
        with self._cond:
            self._imu.append(b)

    def put_clock(self, c: ClockSync) -> None:
        with self._cond:
            self._clock = c

    def get(self, timeout: float = 0.1) -> tuple[Frame | None, ImuBatch | None, ClockSync | None]:
        """Waits for a new image (at most `timeout`); returns (image, IMU, clock)."""
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            f, self._frame = self._frame, None
            imu, self._imu = merge_batches(self._imu), []
            c, self._clock = self._clock, None
            if f is not None:
                self.n_frames_out += 1
            return f, imu, c


class CommandDesk:
    """User requests received from the phone → numbered `VoiceCommand`s.

    Text (Safari recognition, or typed): resolved immediately; recorded sound:
    transcribed by Whisper in a dedicated thread (the server does not wait).
    """

    def __init__(self, raw: dict, reply=None) -> None:
        c = raw.get("command") or {}
        self.sheets_dir = (raw.get("target") or {}).get("sheets_dir")
        self.stt_backend = str(c.get("stt", "whisper"))
        self.reply = reply  # callback (dict): what was understood, shown on the phone
        self._lock = threading.Lock()
        self.pending: list = []
        self.current = None
        self.id = 0
        self.seq = 0
        self.log: list[dict] = []
        self.stt = None
        if self.stt_backend == "whisper" and not raw.get("_no_whisper"):
            from visionaid.phone.stt import WhisperSTT

            self.stt = WhisperSTT(
                str(c.get("whisper_model", "openai/whisper-small")), str(c.get("language", "fr"))
            ).preload()

    def on_message(self, obj: dict) -> None:
        """Server callback (server thread)."""
        if obj.get("type") == "command_audio":
            import base64

            try:
                audio = base64.b64decode(obj.get("data", ""))
            except (ValueError, TypeError):
                return
            mime = str(obj.get("mime", "audio/mp4"))
            threading.Thread(
                target=self._transcribe, args=(audio, mime), name="stt", daemon=True
            ).start()
            return
        texts = [str(obj.get("text", ""))] + [str(a) for a in obj.get("alternatives", []) or []]
        self.submit(texts, source=str(obj.get("source", "phone")))

    def _transcribe(self, audio: bytes, mime: str) -> None:
        if self.stt is None:
            from visionaid.phone.stt import WhisperSTT

            self.stt = WhisperSTT()
        text = self.stt.transcribe(audio, mime)
        self.submit([text], source="whisper")

    def submit(self, texts: list[str], source: str = "") -> None:
        """Recognised phrase(s) → new request."""
        from visionaid.bus.messages import VoiceCommand
        from visionaid.perception.target import parse_command

        texts = [t.strip() for t in texts if t and t.strip()]
        action, sheet = parse_command(texts, self.sheets_dir)
        text = texts[0] if texts else ""
        with self._lock:
            self.seq += 1
            if action in ("search", "stop"):
                self.id += 1
            cmd = VoiceCommand(
                t=time.time(),
                id=self.id,
                action=action,
                text=text,
                query=text if sheet is not None else "",
                target=sheet.name if sheet is not None else "",
                spoken=sheet.spoken if sheet is not None else "",
                plural=bool(sheet.plural) if sheet is not None else False,
                seq=self.seq,
            )
            self.pending.append(cmd)
            self.log.append(
                {"t": cmd.t, "text": text, "action": action, "target": cmd.target, "source": source}
            )
        if self.reply is not None:
            self.reply({"type": "heard", "text": text, "action": action, "target": cmd.target})

    def pop(self):
        """Next request to publish (or None)."""
        with self._lock:
            if self.pending:
                self.current = self.pending.pop(0)
                return self.current
        return None


class PhoneSourceBlock:
    """Source of the real-time graph: starts the server and publishes `frame`, `imu`, `clock`,
    and `command` (requests spoken by the user, `stt` mode)."""

    def __init__(self, raw: dict, record: str | None = None) -> None:
        cfg = _to_dataclass("config", raw)
        self.inbox = PhoneInbox()
        rec = record if record is not None else raw.get("_phone_record")
        mode = str(raw.get("_command_mode") or (raw.get("command") or {}).get("mode", "prompt"))
        self.stt = mode == "stt"
        self.tts_phone = str(raw.get("_tts") or "phone") == "phone"
        self.desk = CommandDesk(raw, reply=self._reply) if self.stt else None
        c = raw.get("command") or {}
        self.repeat_s = float(c.get("repeat_s", 1.0))
        self._t_cmd = 0.0
        self._said: set[int] = set()
        self.server = PhoneServer(
            cfg,
            on_frame=self.inbox.put_frame,
            on_imu=self.inbox.put_imu,
            on_clock=self.inbox.put_clock,
            on_command=self.desk.on_message if self.desk is not None else None,
            record=rec,
            https_port=raw.get("_phone_https_port"),
            http_port=raw.get("_phone_http_port"),
            meta=raw.get("_phone_meta"),
        )
        t = raw.get("tts") or {}
        self.server.voice = {
            "mode": mode,
            "stt": str(c.get("stt", "whisper")),
            "max_record_s": float(c.get("max_record_s", 6.0)),
            "tts": self.tts_phone,
            "lang": str(t.get("lang", "fr-FR")),
            "rate": float(t.get("rate", 1.0)),
            "target": (raw.get("target") or {}).get("spoken") if mode == "prompt" else None,
        }
        self.server.start()
        self.t_start = time.time()
        self._last_sound = None

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        snd = inputs.get("sound")  # sound instruction from the guidance node → phone
        if snd is not None and snd is not self._last_sound:
            self._last_sound = snd
            self.server.send_sound(snd)
            sid = int(getattr(snd, "speech_id", 0) or 0)
            if sid and sid not in self._said and self.tts_phone:
                if self.server.send_speech(sid, snd.speech, bool(snd.speech_duck)):
                    self._said.add(sid)
        f, imu, c = self.inbox.get(0.02 if snd is not None else 0.05)
        out: dict[str, Any] = {}
        if self.desk is not None:
            cmd = self.desk.pop()
            now = time.time()
            if cmd is None and self.desk.current is not None and now - self._t_cmd >= self.repeat_s:
                cmd = self.desk.current  # republished: a node slow to start does not miss it
            if cmd is not None:
                self._t_cmd = now
                out["command"] = cmd
        if imu is not None:  # the IMU goes before the image it precedes
            out["imu"] = imu
        if c is not None:
            out["clock"] = c
        if f is not None:
            out["frame"] = f
        return out or None

    def stats(self) -> dict:
        s = self.server.summary()
        if self.desk is not None:
            s["commands"] = self.desk.log[-20:]
            if self.desk.stt is not None and self.desk.stt.times:
                s["stt_ms_p50"] = float(np.median(self.desk.stt.times) * 1e3)
        s.update(
            frames_in=self.inbox.n_frames_in,
            frames_out=self.inbox.n_frames_out,
            overwritten=self.inbox.n_overwritten,
        )
        return s

    def _reply(self, obj: dict) -> None:
        self.server.send_json(obj)

    def close(self) -> None:
        self.server.stop()
