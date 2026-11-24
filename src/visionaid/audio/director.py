"""Choice of the sound to play from a guidance instruction.

All the sound logic is here, in one place: the phone (Web Audio) and the offline
rendering (WAV) only execute the `SoundCommand`s produced.

Two overlaid layers, so that the user hears at the same time where to go and what is in the way:

- **path** (sound 2): repeated note, placed by a stereo pan towards the direction of the path; the
  tempo speeds up when approaching the target. Object not found yet: search sound (sound 4);
- **obstacles** (sound 1): looping pad, centred, louder and louder when an obstacle in the
  frontal corridor gets closer (capped below the path, so as not to stress);
- **arrival** (sound 3): once, then silence;
- **tracking lost**: no more path, obstacle pad at medium volume ("stop").

Pan (like the panoramic knob of a mixing desk): pan = azimuth / 90°, bounded to ±1; at ±1 all
the sound is in a single ear. Path behind: everything on one side, the side of the shortest
U-turn, kept as long as the path does not clearly move to the other side (hysteresis around
180°, otherwise the sound would jump from one ear to the other).
"""

from __future__ import annotations

import math

import numpy as np

from visionaid.bus.messages import GuidanceCue, SoundCommand

SILENT, SEARCH, GUIDE, LOST, ARRIVED = "silent", "search", "guide", "lost", "arrived"
STATE_MODE = {
    "SEARCHING": SEARCH,
    "NAVIGATING": GUIDE,
    "APPROACHING": GUIDE,
    "OBSTACLE_ALERT": GUIDE,  # the path continues; the obstacle is the top layer
    "ARRIVED": ARRIVED,
    "TRACKING_LOST": LOST,
}
MUTE_DB = -120.0
SOUND_KEYS = ("guide", "search", "arrived", "obstacle")  # sounds from `audio.sounds`


def equal_power(pan):
    """Constant-power (left, right) gains; pan −1 → (1, 0), 0 → (0.707, 0.707)."""
    th = (np.clip(np.asarray(pan, float), -1.0, 1.0) + 1.0) * math.pi / 4
    return np.cos(th), np.sin(th)


class PanTracker:
    """Azimuth (rad, left +) → pan (−1 left … +1 right), with hysteresis behind."""

    def __init__(self, hold_deg: float = 150.0) -> None:
        self.hold = math.radians(float(hold_deg))
        self.side = 0.0  # side kept when the path is behind: +1 left, −1 right

    def __call__(self, az: float) -> float:
        az = math.atan2(math.sin(az), math.cos(az))
        if abs(az) <= math.pi / 2:
            self.side = 0.0
            return float(np.clip(-az / (math.pi / 2), -1.0, 1.0))
        s = 1.0 if az > 0 else -1.0
        if self.side == 0.0 or (s != self.side and abs(az) < self.hold):
            self.side = s  # entering behind, or path clearly moved to the other side
        return -self.side

    def reset(self) -> None:
        self.side = 0.0


class SoundDirector:
    """Converts successive `GuidanceCue`s into `SoundCommand`s."""

    def __init__(self, cfg_audio) -> None:
        a = cfg_audio
        self.far_s = float(a.guide_interval_far_s)
        self.near_s = float(a.guide_interval_near_s)
        self.accel_m = float(a.guide_accel_m)
        self.near_m = float(a.guide_near_m)
        self.search_s = float(a.search_interval_s)
        self.ob_far = float(a.obstacle_far_m)
        self.ob_near = float(a.obstacle_near_m)
        self.ob_min = float(a.obstacle_min_db)
        self.ob_max = float(a.obstacle_max_db)
        self.lost_db = float(a.lost_obstacle_db)
        self.cooldown = float(a.arrived_cooldown_s)
        self.pan = PanTracker(float(a.back_hold_deg))
        self.prev_mode: str | None = None
        self.t_arrived = -math.inf

    def interval(self, distance: float) -> float:
        """Spacing between two path notes: constant when far, then shorter and shorter
        below `guide_accel_m`, down to `guide_interval_near_s` at `guide_near_m`."""
        if not math.isfinite(distance):
            return self.far_s
        x = float(np.clip((distance - self.near_m) / max(self.accel_m - self.near_m, 1e-6), 0, 1))
        return self.near_s + (self.far_s - self.near_s) * x

    def obstacle_db(self, distance: float) -> float:
        """Gain of the obstacle pad: muted beyond `obstacle_far_m`, then linear in
        dB up to `obstacle_max_db` at `obstacle_near_m`."""
        if not math.isfinite(distance) or distance >= self.ob_far:
            return MUTE_DB
        x = float(np.clip((self.ob_far - distance) / max(self.ob_far - self.ob_near, 1e-6), 0, 1))
        return self.ob_min + (self.ob_max - self.ob_min) * x

    def command(self, cue: GuidanceCue) -> SoundCommand:
        mode = STATE_MODE.get(cue.state, SILENT)
        az = float(getattr(cue, "guide_azimuth", math.nan))
        if not math.isfinite(az):  # instruction without path direction (old recordings)
            az = float(cue.azimuth) if cue.state != "OBSTACLE_ALERT" else 0.0
        if mode in (SEARCH, GUIDE):
            pan = self.pan(az)
        else:
            pan, az = 0.0, 0.0
            self.pan.reset()
        if mode == GUIDE:
            ob = self.obstacle_db(float(getattr(cue, "obstacle_distance", math.inf)))
        elif mode == LOST:
            ob = self.lost_db
        else:
            ob = MUTE_DB
        event = ""
        if (
            mode == ARRIVED
            and self.prev_mode != ARRIVED
            and cue.t - self.t_arrived >= self.cooldown
        ):
            event = ARRIVED
            self.t_arrived = cue.t
        self.prev_mode = mode
        return SoundCommand(
            t=float(cue.t),
            frame_idx=int(cue.frame_idx),
            state=cue.state,
            mode=mode,
            azimuth=az,
            pan=float(pan),
            distance=float(cue.distance),
            interval_s=self.search_s if mode == SEARCH else self.interval(cue.distance),
            gain_db=0.0,
            obstacle_db=float(ob),
            event=event,
            speech=str(getattr(cue, "speech", "") or ""),
            speech_id=int(getattr(cue, "speech_id", 0) or 0),
            speech_duck=bool(getattr(cue, "speech_duck", False)),
        )
