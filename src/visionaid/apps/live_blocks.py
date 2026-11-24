"""Blocks of the "phone → path" graph (phase 20), importable by the bus subprocesses."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from visionaid.bus.messages import Path
from visionaid.utils.config import _to_dataclass, resolve_path


def _cfg(raw: dict):
    return _to_dataclass("config", raw)


# ---------------------------------------------------------------------------- source
def make_phone_source(raw: dict):
    """Phone server (live, or fake phone driven by the main process)."""
    from visionaid.phone.block import PhoneSourceBlock

    return PhoneSourceBlock(raw)


# ---------------------------------------------------------------------------- models (or stubs)
STUB_BOX = np.array([150, 300, 175, 360], float)


def stub_scene_depth(h: int, w: int, K: np.ndarray, pitch_deg: float = 20.0, cam_h: float = 1.4):
    """Depth of a floor seen by a tilted camera, wall at 5 m (tests without model)."""
    v, _u = np.mgrid[0:h, 0:w]
    y = (v - K[1, 2]) / K[1, 1]
    p = math.radians(pitch_deg)
    den = math.cos(p) * y + math.sin(p)
    floor = np.where(den > 1e-3, cam_h / np.maximum(den, 1e-3), np.inf)
    return np.minimum(floor, 5.0).astype(np.float32)


def make_live_perception(raw: dict):
    """Real perception, or stub (fixed box, text-only ReID) if `_stub_models`."""
    return LivePerceptionBlock(raw)


def stt_mode(raw: dict) -> bool:
    """Request spoken by the user (no target before the first request)."""
    return str(raw.get("_command_mode") or (raw.get("command") or {}).get("mode")) == "stt"


class LivePerceptionBlock:
    """Perception + target change at each request (`command` topic)."""

    def __init__(self, raw: dict) -> None:
        from visionaid.perception.pipeline import PerceptionBlock

        cfg = _cfg(raw)
        self.raw = raw
        active = not stt_mode(raw)
        self.cmd_id = 0
        self.n_switch = 0
        if not raw.get("_stub_models"):
            self.block = PerceptionBlock(cfg, active=active)
            if not active:
                self._preload()
            return
        from visionaid.perception.detector import StubDetector
        from visionaid.perception.reid import TextOnlyReID

        det = StubDetector(fn=lambda img, k: [(STUB_BOX, 0.9, "bottle")])
        det.backend, det.weights, det.device = "stub", "stub", "cpu"
        self.block = PerceptionBlock(
            cfg,
            detector=det,
            reid=TextOnlyReID(cfg) if active else None,
            active=active,
            reid_factory=lambda c, d: TextOnlyReID(c),
        )

    def _preload(self) -> None:
        """ReID gallery of each card prepared at startup (~10 s per card the first time):
        the target asked by voice is then taken in a few milliseconds."""
        from visionaid.perception.target import load_sheets, target_raw
        from visionaid.utils.logging import get_logger

        log = get_logger("perception")
        for sh in load_sheets((self.raw.get("target") or {}).get("sheets_dir")):
            t0 = time.time()
            try:
                self.block.set_target(_cfg(target_raw(self.raw, sh.phrases[0])), 0)
                log.info("Target \"%s\" ready (%.1f s)", sh.label, time.time() - t0)
            except Exception as e:  # noqa: BLE001 - card without a usable photo…
                log.warning("Target \"%s\" not prepared: %s", sh.label, e)
        self.block.clear_target(0)

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        cmd = inputs.get("command")
        if cmd is not None and cmd.id != self.cmd_id:
            self.cmd_id = int(cmd.id)
            self.n_switch += 1
            if cmd.action == "search" and cmd.query:
                from visionaid.perception.target import target_raw

                self.block.set_target(_cfg(target_raw(self.raw, cmd.query)), cmd.id)
            else:
                self.block.clear_target(cmd.id)
        return self.block.process(inputs)

    def stats(self) -> dict:
        return {"target_switches": self.n_switch, "target_id": self.block.target_id}


class LiveGeometryBlock:
    """Depth + pose (VIO if IMU); device intrinsics; stub depth if requested."""

    def __init__(self, raw: dict) -> None:
        from visionaid.perception.depth import DepthEstimator, StubDepthBackend
        from visionaid.pose.source import make_pose_source

        cfg = _cfg(raw)
        if raw.get("_stub_models"):
            self.depth = DepthEstimator(
                cfg,
                backend=StubDepthBackend(lambda h, w: stub_scene_depth(h, w, self._K(h, w))),
            )
        else:
            self.depth = DepthEstimator(cfg)
        self.cfg = cfg
        self.vo = make_pose_source(cfg, bool(raw.get("_imu", True)))
        self._Kc: dict = {}
        self.n = 0
        self.t_proc: list[float] = []

    def _K(self, h: int, w: int) -> np.ndarray:
        from visionaid.calib.device import phone_intrinsics

        if (w, h) not in self._Kc:
            self._Kc[(w, h)] = phone_intrinsics(self.cfg, w, h).K
        return self._Kc[(w, h)]

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        from visionaid.pose.source import merge_imu

        if hasattr(self.vo, "add_imu"):
            self.vo.add_imu(merge_imu(inputs.get("imu")))
        f = inputs.get("frame")
        if f is None:
            return None
        t0 = time.perf_counter()
        d = self.depth(f)
        if hasattr(self.vo, "correct_depth"):
            d = self.vo.correct_depth(f, d)
        p = self.vo.update(f, d)
        self.t_proc.append(time.perf_counter() - t0)
        self.n += 1
        return {"depth": d, "pose": p}

    def stats(self) -> dict:
        s = {"n": self.n}
        if self.t_proc:
            s["process_ms_p50"] = float(np.percentile(self.t_proc, 50) * 1e3)
        if hasattr(self.vo, "stats"):
            s.update(self.vo.stats())
        else:
            s["lost_fraction"] = self.vo.lost_fraction
        return s


def make_live_world(raw: dict):
    """World model with the device intrinsics (phase 17 calibration)."""
    from visionaid.apps.realtime_blocks import WorldBlock

    blk = WorldBlock(raw)
    blk.use_phone_intrinsics = True
    return blk


# ---------------------------------------------------------------------------- planification
class LivePlannerBlock:
    """Fixed-rate planner with a watchdog.

    Replans on a new costmap; if no fresh pose has arrived for `stale_s`
    (phone disconnected, pose lost), publishes an **invalid** path (once per second).
    """

    def __init__(self, raw: dict, stale_s: float = 1.5) -> None:
        from visionaid.planning.planner import Planner

        self.cfg = _cfg(raw)
        self.planner = Planner(self.cfg)
        self.cmd_id = 0
        self.stale_s = stale_s
        self._last_pose_key = None
        self._last_pose_wall = time.time()
        self._last_cm_key = None
        self._last_invalid = 0.0
        self.n_plans = 0
        self.n_invalid_stale = 0

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        from visionaid.planning.planner import Planner

        cmd = inputs.get("command")
        if cmd is not None and cmd.id != self.cmd_id:
            self.cmd_id = int(cmd.id)
            self.planner = Planner(self.cfg)  # new object: no more commitment to the old path
            self._last_cm_key = None
        now = time.time()
        pose = inputs.get("pose_world")
        if pose is not None:
            key = (pose.frame_idx, pose.t)
            if key != self._last_pose_key:
                self._last_pose_key = key
                self._last_pose_wall = now
        stale = now - self._last_pose_wall > self.stale_s or (
            pose is not None and pose.quality == "lost"
        )
        if stale:
            if now - self._last_invalid >= 1.0:
                self._last_invalid = now
                self.n_invalid_stale += 1
                t = pose.t if pose is not None else 0.0
                idx = pose.frame_idx if pose is not None else -1
                nan2 = np.full(2, np.nan)
                return {"path": Path(t, idx, np.zeros((0, 2)), nan2, nan2, False, 0.0, self.cmd_id)}
            return None
        cm, obj = inputs.get("costmap"), inputs.get("object")
        if obj is not None and getattr(obj, "target_id", 0) != self.cmd_id:
            obj = None  # object of a previous request
        if cm is None or obj is None or pose is None:
            return None
        ck = (cm.frame_idx, cm.t)
        if ck == self._last_cm_key:
            return None
        self._last_cm_key = ck
        self.n_plans += 1
        p = self.planner.plan(cm, pose.T_world_cam[:2, 3], obj)
        p.target_id = self.cmd_id
        return {"path": p}

    def stats(self) -> dict:
        return {"n_plans": self.n_plans, "n_invalid_stale": self.n_invalid_stale}


# ---------------------------------------------------------------------------- guidage sonore
class LiveGuidanceBlock:
    """Guidance instruction + sound instruction, at a fixed rate (`realtime.cue_hz`).

    Without a new pose for `stale_s` (phone disconnected, geometry blocked), the pose is
    considered lost: the user hears "stop" instead of a frozen guide.

    Requests (`command` topic): new search → new guidance (announcements included);
    "stop" → silence; misunderstood phrase → spoken reply, the current search continues.
    In `stt` mode, nothing plays before the first request (IDLE state).
    Sentences: carried by the sound instruction (phone) and/or spoken by the Mac (`_tts`).
    """

    def __init__(self, raw: dict, stale_s: float = 1.5) -> None:
        from visionaid.audio.director import SoundDirector
        from visionaid.guidance.cues import GuidanceEngine
        from visionaid.guidance.narration import SpeechQueue

        cfg = _cfg(raw)
        self.raw = raw
        self.cfg = cfg
        hold = float(getattr(getattr(cfg, "tts", None), "hold_s", 1.0) or 1.0)
        self.speech = SpeechQueue(hold)  # a single queue: unique numbers for the phone
        self.active = not stt_mode(raw)
        self.engine = GuidanceEngine(cfg, announce_start=self.active, speech=self.speech)
        self.director = SoundDirector(cfg.audio)
        self.stale_s = stale_s
        self._last_key = None
        self._last_wall = time.time()
        self.modes: dict[str, int] = {}
        self.n_stale = 0
        self.cmd_id = 0
        self.cmd_seq = 0
        self.landmarks = None
        self.commands: list[dict] = []
        out = str(raw.get("_tts") or "phone")
        self.mac = None
        if out == "mac":
            from visionaid.audio.tts import MacSpeaker

            tts = getattr(cfg, "tts", None)
            self.mac = MacSpeaker(
                str(getattr(tts, "mac_voice", "Thomas")), int(getattr(tts, "mac_rate_wpm", 190))
            )
        self._said: set[int] = set()

    def _on_command(self, cmd) -> None:
        from visionaid.guidance.cues import GuidanceEngine

        self.cmd_seq = int(cmd.seq)
        self.commands.append({"t": cmd.t, "action": cmd.action, "text": cmd.text, "id": cmd.id})
        if cmd.action == "unknown":
            from visionaid.perception.target import known_targets_phrase

            heard = f" « {cmd.text} »" if cmd.text else ""
            self.speech.say(
                f"Je n'ai pas compris{heard}. Tu peux chercher {known_targets_phrase()}."
            )
            return
        if cmd.id == self.cmd_id:
            return
        self.cmd_id = int(cmd.id)
        if cmd.action == "search":
            from visionaid.perception.target import target_raw

            cfg_t = _cfg(target_raw(self.raw, cmd.query)) if cmd.query else self.cfg
            self.engine = GuidanceEngine(
                cfg_t,
                spoken=cmd.spoken,
                plural=cmd.plural,
                announce_start=True,
                target_id=cmd.id,
                speech=self.speech,
            )
            self.engine.landmarks = self.landmarks
            self.active = True
        else:
            self.active = False
            self.speech.say("D'accord, j'arrête la recherche.")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        from dataclasses import replace

        from visionaid.bus.messages import GuidanceCue

        cmd = inputs.get("command")
        if cmd is not None and (cmd.seq != self.cmd_seq or cmd.id != self.cmd_id):
            self._on_command(cmd)
        lms = inputs.get("landmarks")
        if lms is not None:
            self.landmarks = lms
            self.engine.landmarks = lms
        pose = inputs.get("pose_world")
        if pose is None and not self.speech.queue and self.speech.cur is None:
            return None  # no pose yet: silence
        now = time.time()
        if pose is None or not self.active:
            t = pose.t if pose is not None else now
            text, sid, duck = self.speech.current(t)
            cue = GuidanceCue(
                t, -1 if pose is None else pose.frame_idx, "IDLE", speech=text, speech_id=sid
            )
            cue.speech_duck = duck
        else:
            key = (pose.frame_idx, pose.t)
            if key != self._last_key:
                self._last_key, self._last_wall = key, now
            elif now - self._last_wall > self.stale_s:
                pose = replace(pose, quality="lost")
                self.n_stale += 1
            cue = self.engine.update(
                pose.t,
                pose.frame_idx,
                pose,
                inputs.get("object"),
                inputs.get("path"),
                inputs.get("costmap"),
            )
        snd = self.director.command(cue)
        if snd.speech_id and snd.speech_id not in self._said:
            self._said.add(snd.speech_id)
            if self.mac is not None:
                self.mac.say(snd.speech)
        self.modes[snd.mode] = self.modes.get(snd.mode, 0) + 1
        return {"cue": cue, "sound": snd}

    def stats(self) -> dict:
        return {
            "modes": self.modes,
            "n_stale": self.n_stale,
            "messages": self.engine.messages[-50:],
            "speech": self.speech.history[-50:],
            "commands": self.commands[-20:],
        }

    def close(self) -> None:
        if self.mac is not None:
            self.mac.close()


# ---------------------------------------------------------------------------- puits
class PathSink:
    """Measurement: capture → published path latency, valid path rate, target state.

    Capture time on the computer side: `t_phone + offset` (synchronised clock, `clock` topic) live;
    `_t0_wall + t` in session replay (time brought back to the start, like video time).
    """

    def __init__(self, raw: dict) -> None:
        self.t0_wall = raw.get("_t0_wall")
        self.offset: float | None = None
        self.lat: list[float] = []
        self.valid_times: list[float] = []
        self.invalid_times: list[float] = []
        self.first_valid_wall: float | None = None
        self.t_start = time.time()
        self.obj_states: dict[str, int] = {}
        self.out = resolve_path(raw["paths"]["outputs"]) / "live_paths.jsonl"
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.out, "w")  # noqa: SIM115
        self.live = bool(raw.get("_live_clock", True))

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        c = inputs.get("clock")
        if c is not None and math.isfinite(c.uncertainty):
            self.offset = c.offset
        p = inputs.get("path")
        obj = inputs.get("object")
        if obj is not None:
            self.obj_states[obj.state] = self.obj_states.get(obj.state, 0) + 1
        if p is None:
            return None
        if p.frame_idx == getattr(self, "_last_idx", None) and p.valid == getattr(
            self, "_last_valid", None
        ):
            return None
        self._last_idx, self._last_valid = p.frame_idx, p.valid
        now = time.time()
        lat = None
        if p.frame_idx >= 0:
            if self.live and self.offset is not None:
                lat = now - (p.t + self.offset)
            elif self.t0_wall is not None and not self.live:
                lat = now - (self.t0_wall + p.t)
        if p.valid:
            self.valid_times.append(now)
            self.first_valid_wall = self.first_valid_wall or now
            if lat is not None and 0 <= lat < 30:
                self.lat.append(lat)
        else:
            self.invalid_times.append(now)
        self.fh.write(
            json.dumps(
                {
                    "wall": now,
                    "t": p.t,
                    "idx": p.frame_idx,
                    "valid": bool(p.valid),
                    "length": float(p.length),
                    "lat": lat,
                }
            )
            + "\n"
        )
        self.fh.flush()
        return None

    def stats(self) -> dict:
        lat = np.array(self.lat) if self.lat else np.zeros(0)
        vt = np.array(self.valid_times)
        rate = None
        if len(vt) >= 2:
            rate = float((len(vt) - 1) / max(vt[-1] - vt[0], 1e-6))
        return {
            "n_valid_paths": len(self.valid_times),
            "n_invalid_paths": len(self.invalid_times),
            "valid_path_hz": rate,
            "first_valid_after_s": (self.first_valid_wall - self.t_start)
            if self.first_valid_wall
            else None,
            "e2e_capture_to_path_ms_p50": float(np.percentile(lat, 50) * 1e3) if len(lat) else None,
            "e2e_capture_to_path_ms_p95": float(np.percentile(lat, 95) * 1e3) if len(lat) else None,
            "n_latency": int(len(lat)),
            "object_states": self.obj_states,
            "valid_wall_times": [round(x - self.t_start, 2) for x in self.valid_times][-400:],
            "invalid_wall_times": [round(x - self.t_start, 2) for x in self.invalid_times][-100:],
        }

    def close(self) -> None:
        self.fh.close()


# ---------------------------------------------------------------------------- visualisation
class _RerunFanout:
    """The `rerun` module, but with `log` / `set_time` / `send_blueprint` sent to several streams."""

    def __init__(self, rr, streams: list) -> None:
        self._rr = rr
        self._streams = streams

    def __getattr__(self, name: str):
        return getattr(self._rr, name)

    def log(self, *args, **kw) -> None:
        for s in self._streams:
            self._rr.log(*args, recording=s, **kw)

    def set_time(self, *args, **kw) -> None:
        for s in self._streams:
            self._rr.set_time(*args, recording=s, **kw)

    def send_blueprint(self, *args, **kw) -> None:
        for s in self._streams:
            self._rr.send_blueprint(*args, recording=s, **kw)

    def disconnect(self) -> None:
        for s in self._streams:
            try:
                s.flush()
                s.disconnect()
            except Exception:
                pass


class RerunLiveBlock:
    """Rerun visualisation: `live` (viewer started), `file` (.rrd) or both (`both`).

    Fixed layout: large 3D world (grid floor, obstacle map laid on the floor, point cloud
    coloured by height, trajectory, user position and heading, path,
    target), and on the right the annotated image, the 2D map and the state.
    """

    def __init__(self, raw: dict) -> None:
        import rerun as rr

        mode = str(raw.get("_rerun", "file"))
        out = resolve_path(raw["paths"]["outputs"]) / "live_session.rrd"
        out.parent.mkdir(parents=True, exist_ok=True)
        # one stream per destination: window (spawn) and/or file. Rerun 0.23 has no
        # `set_sinks`: the old "both" mode wrote no file, without saying so
        streams = []
        if mode in ("live", "both"):
            live = rr.RecordingStream("visionaid_live")
            live.spawn()
            streams.append(live)
        if mode in ("file", "both"):
            rec = rr.RecordingStream("visionaid_live")
            rec.save(str(out))
            streams.append(rec)
        self.rr = _RerunFanout(rr, streams)
        self.out = out
        self.target_label = str((raw.get("target") or {}).get("name") or "cible")
        self._blueprint()
        self.rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._floor_grid()
        self.traj: list[np.ndarray] = []
        self.n = 0
        self.last_cm = None
        self.last_path = None
        self.last_depth_idx = None

    def _blueprint(self) -> None:
        try:
            import rerun.blueprint as rrb

            bp = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="Monde 3D"),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="camera/image", name="Camera"),
                        rrb.Spatial2DView(origin="costmap", name="Obstacle map"),
                        rrb.TextLogView(origin="status", name="State"),
                        row_shares=[3, 3, 1],
                    ),
                    column_shares=[3, 2],
                ),
                collapse_panels=True,
            )
            self.rr.send_blueprint(bp)
        except Exception:  # Rerun version without blueprint: automatic layout
            pass

    def _floor_grid(self, half: float = 8.0, step: float = 1.0) -> None:
        """Floor grid (1 m) and axes, static."""
        rr = self.rr
        ticks = np.arange(-half, half + 1e-9, step)
        lines = [[[x, -half, 0.0], [x, half, 0.0]] for x in ticks]
        lines += [[[-half, y, 0.0], [half, y, 0.0]] for y in ticks]
        rr.log(
            "world/sol",
            rr.LineStrips3D(lines, colors=[[90, 90, 90]], radii=0.004),
            static=True,
        )
        rr.log(
            "world/origine",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 2,
                vectors=[[0.5, 0, 0], [0, 0.5, 0]],
                colors=[[200, 60, 60], [60, 200, 60]],
                labels=["x (initial heading)", "y"],
            ),
            static=True,
        )

    def _costmap_3d(self, cm) -> None:
        """Map laid on the floor: obstacles in dark red, free areas in light grey."""
        from visionaid.mapping.costmap import FREE, OCC

        rr = self.rr
        res = float(cm.resolution)
        for name, val, color, z, step in (
            ("obstacles", OCC, [170, 40, 40], 0.03, 1),
            ("libre", FREE, [200, 200, 200], 0.0, 2),
        ):
            ii, jj = np.nonzero(cm.state[::step, ::step] == val)
            if len(ii) == 0:
                rr.log(f"world/carte/{name}", rr.Clear(recursive=False))
                continue
            x = cm.origin[0] + (jj * step + 0.5) * res
            y = cm.origin[1] + (ii * step + 0.5) * res
            pts = np.c_[x, y, np.full(len(x), z)]
            if val == OCC:
                rr.log(
                    f"world/carte/{name}",
                    rr.Boxes3D(
                        centers=pts,
                        half_sizes=[[res / 2, res / 2, 0.03]],
                        colors=[color],
                        fill_mode="solid",
                    ),
                )
            else:
                rr.log(
                    f"world/carte/{name}", rr.Points3D(pts, colors=[color], radii=res * step / 2)
                )

    def _cloud(self, depth, T: np.ndarray, K: np.ndarray) -> None:
        """Point cloud of the latest depth, coloured by height (blue low → orange high)."""
        from visionaid.mapping.geometry import backproject, transform

        pc, _ = backproject(depth.depth, K, 8, None, 0.2, 5.0)
        if not len(pc):
            return
        pw = transform(T, pc)
        z = np.clip(pw[:, 2] / 2.0, 0.0, 1.0)[:, None]
        cols = (np.array([40, 110, 230]) * (1 - z) + np.array([255, 150, 30]) * z).astype(np.uint8)
        self.rr.log("world/nuage", self.rr.Points3D(pw, colors=cols, radii=0.012))

    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        rr = self.rr
        f = inputs.get("frame")
        if f is None:
            return None
        self.n += 1
        rr.set_time("t", duration=float(f.t))
        h, w = f.image.shape[:2]
        rr.log("camera/image", rr.Image(f.image[:, :, ::-1]).compress(jpeg_quality=70))
        d = inputs.get("detections")
        if d is not None and len(d.boxes):
            labels = [
                (d.labels[i] if i < len(d.labels) else "")
                + (f" {self.target_label.upper()}" if i == d.target_idx else "")
                for i in range(len(d.boxes))
            ]
            colors = [
                [255, 140, 0] if i == d.target_idx else [160, 160, 160] for i in range(len(d.boxes))
            ]
            rr.log(
                "camera/image/detections",
                rr.Boxes2D(
                    array=d.boxes, array_format=rr.Box2DFormat.XYXY, labels=labels, colors=colors
                ),
            )
        else:
            rr.log("camera/image/detections", rr.Clear(recursive=False))
        pose = inputs.get("pose_world")
        if pose is not None:
            T = pose.T_world_cam
            rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    image_from_camera=f.K,
                    width=w,
                    height=h,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.25,
                ),
            )
            self.traj.append(T[:3, 3].copy())
            if len(self.traj) > 1:
                tr = np.array(self.traj[-3000:])
                tr_floor = np.c_[tr[:, :2], np.full(len(tr), 0.01)]
                rr.log(
                    "world/trajectoire",
                    rr.LineStrips3D([tr_floor], colors=[[60, 120, 255]], radii=0.015),
                )
            # user: point on the floor + heading arrow (projected optical axis)
            fwd = T[:3, 2].copy()
            fwd[2] = 0.0
            nf = np.linalg.norm(fwd)
            fwd = fwd / nf * 0.6 if nf > 1e-6 else np.array([0.6, 0.0, 0.0])
            p0 = np.array([T[0, 3], T[1, 3], 0.02])
            rr.log(
                "world/vous",
                rr.Arrows3D(
                    origins=[p0],
                    vectors=[fwd],
                    colors=[[60, 120, 255]],
                    radii=0.025,
                    labels=["vous"],
                ),
            )
            dep = inputs.get("depth")
            if (
                dep is not None
                and dep.frame_idx == pose.frame_idx
                and dep.frame_idx != self.last_depth_idx
                and self.n % 3 == 0
            ):
                self.last_depth_idx = dep.frame_idx
                self._cloud(dep, T, f.K)
            rr.log("status/pose", rr.TextLog(f"pose : {pose.quality}"))
        cm = inputs.get("costmap")
        if cm is not None and cm is not self.last_cm and self.n % 5 == 0:
            from visionaid.mapping.costmap import FREE, OCC

            self.last_cm = cm
            img = np.full(cm.state.shape + (3,), 128, np.uint8)
            img[cm.state == FREE] = 255
            img[cm.state == OCC] = 0
            rr.log("costmap", rr.Image(img[::-1]))
            self._costmap_3d(cm)
        p = inputs.get("path")
        if p is not None and p is not self.last_path:
            self.last_path = p
            if p.valid and len(p.points) > 1:
                p3 = np.c_[p.points, np.full(len(p.points), 0.05)]
                rr.log("world/chemin", rr.LineStrips3D([p3], colors=[[0, 220, 0]], radii=0.03))
                rr.log(
                    "world/visee",
                    rr.Points3D([[*p.lookahead, 0.05]], colors=[[0, 255, 0]], radii=0.06),
                )
            else:
                rr.log("world/chemin", rr.Clear(recursive=False))
                rr.log("world/visee", rr.Clear(recursive=False))
        snd = inputs.get("sound")
        if snd is not None and snd is not getattr(self, "_last_sound", None):
            self._last_sound = snd
            rr.log(
                "status/son",
                rr.TextLog(
                    f"sound: {snd.mode} ({snd.state}), direction {np.degrees(snd.azimuth):+.0f}°, "
                    f"pan {snd.pan:+.2f}, one note every {snd.interval_s:.2f} s, "
                    f"obstacles {snd.obstacle_db:.0f} dB"
                ),
            )
        if snd is not None and snd.speech_id and snd.speech_id != getattr(self, "_last_sid", 0):
            self._last_sid = snd.speech_id
            rr.log("status/voix", rr.TextLog(f"voix : {snd.speech}"))
        lms = inputs.get("landmarks")
        if lms is not None and lms is not getattr(self, "_last_lms", None) and len(lms.labels):
            from visionaid.guidance.narration import noun

            self._last_lms = lms
            b = np.asarray(lms.boxes, float)
            top = np.asarray(lms.top_z, float)
            centers = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, top / 2]
            half = np.c_[(b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2, top / 2]
            rr.log(
                "world/meubles",
                rr.Boxes3D(
                    centers=centers,
                    half_sizes=half,
                    colors=[[140, 90, 200]],
                    labels=[noun(lab)[0] for lab in lms.labels],
                ),
            )
        obj = inputs.get("object")
        if obj is not None and np.isfinite(obj.position).all():
            pos = np.asarray(obj.position, float)
            rr.log(
                "world/cible",
                rr.Points3D(
                    [pos],
                    colors=[[255, 140, 0]],
                    radii=0.1,
                    labels=[f"{self.target_label} ({obj.state})"],
                ),
            )
            rr.log(  # vertical stick from the floor to the target: readable even from above
                "world/cible_tige",
                rr.LineStrips3D([[[pos[0], pos[1], 0.0], pos]], colors=[[255, 140, 0]], radii=0.01),
            )
        return None

    def close(self) -> None:
        try:
            self.rr.disconnect()
        except Exception:
            pass
