"""Computation of the guidance instructions (azimuth/elevation relative to the head, distance, messages)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from visionaid.bus.messages import Costmap, GuidanceCue, ObjectEstimate, Path, Pose
from visionaid.guidance.state_machine import (
    APPROACHING,
    ARRIVED,
    NAVIGATING,
    OBSTACLE_ALERT,
    SEARCHING,
    TRACKING_LOST,
    GuidanceInputs,
    GuidanceStateMachine,
)
from visionaid.mapping.costmap import OCC

MESSAGES = {
    SEARCHING: "Tournez lentement sur place",
    NAVIGATING: "Suivez le son",
    APPROACHING: "Vous approchez, tendez la main",
    ARRIVED: "Vous êtes arrivé",
    OBSTACLE_ALERT: "Attention, obstacle devant",
    TRACKING_LOST: "Arrêtez-vous, repérage perdu",
}


def head_frame(T_world_head: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, horizontal forward, horizontal left) of the head in the world (z up)."""
    pos = T_world_head[:3, 3]
    fwd = T_world_head[:3, 2].copy()  # axe optique
    fwd[2] = 0.0
    if np.linalg.norm(fwd) < 1e-6:  # vertical gaze: take camera −y
        fwd = -T_world_head[:3, 1].copy()
        fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    return pos, fwd, left


def blocked_along(
    cm: Costmap | None, pos: np.ndarray, bearing: float, dist: float, half_width: float
) -> bool:
    """Is an obstacle of the map inside the body corridor (half width
    `half_width`) along the world direction `bearing`, over `dist` metres?"""
    if cm is None or dist <= 0:
        return False
    res = float(cm.resolution)
    c, s = math.cos(bearing), math.sin(bearing)
    along = np.arange(0.2, dist + 1e-9, res)  # not under the feet (map noise)
    side = np.arange(-half_width, half_width + 1e-9, res)
    a, b = np.meshgrid(along, side)
    x = pos[0] + a * c - b * s
    y = pos[1] + a * s + b * c
    j = np.floor((x - cm.origin[0]) / res).astype(int)
    i = np.floor((y - cm.origin[1]) / res).astype(int)
    h, w = cm.state.shape
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    return bool((cm.state[i[ok], j[ok]] == OCC).any())


def relative_direction(T_world_head: np.ndarray, p_world: np.ndarray) -> tuple[float, float, float]:
    """(azimuth left+, elevation up+, horizontal distance) of a world point."""
    pos, fwd, left = head_frame(T_world_head)
    p = np.asarray(p_world, float)
    if p.shape[0] == 2:  # floor point without height: at head height (zero elevation)
        p = np.array([p[0], p[1], pos[2]])
    v = p - pos
    hd = float(math.hypot(v[0], v[1]))
    az = math.atan2(float(v @ left), float(v @ fwd))
    el = math.atan2(float(v[2]), max(hd, 1e-6))
    return az, el, hd


def frontal_obstacle(
    cm: Costmap | None,
    T_world_head: np.ndarray,
    max_dist: float,
    cone_deg: float,
    half_width: float = np.inf,
) -> tuple[float, float]:
    """Distance and azimuth of the closest occupied obstacle in the frontal cone (inf if none).

    `half_width` restricts to the corridor swept by the body (|lateral offset| < half_width): an
    obstacle beside a clear path does not trigger an alert.
    """
    if cm is None:
        return math.inf, 0.0
    pos, fwd, left = head_frame(T_world_head)
    r = int(math.ceil(max_dist / cm.resolution)) + 1
    ci = int(math.floor((pos[1] - cm.origin[1]) / cm.resolution))
    cj = int(math.floor((pos[0] - cm.origin[0]) / cm.resolution))
    h, w = cm.state.shape
    a0, a1, b0, b1 = max(ci - r, 0), min(ci + r + 1, h), max(cj - r, 0), min(cj + r + 1, w)
    if a0 >= a1 or b0 >= b1:
        return math.inf, 0.0
    ii, jj = np.nonzero(cm.state[a0:a1, b0:b1] == OCC)
    if len(ii) == 0:
        return math.inf, 0.0
    x = cm.origin[0] + (jj + b0 + 0.5) * cm.resolution - pos[0]
    y = cm.origin[1] + (ii + a0 + 0.5) * cm.resolution - pos[1]
    d = np.hypot(x, y)
    lat = x * left[0] + y * left[1]
    az = np.arctan2(lat, x * fwd[0] + y * fwd[1])
    m = (np.abs(az) <= math.radians(cone_deg)) & (d <= max_dist) & (np.abs(lat) < half_width)
    if not m.any():
        return math.inf, 0.0
    k = int(np.argmin(np.where(m, d, np.inf)))
    return float(d[k]), float(az[k])


class GuidanceEngine:
    """Combines pose, object, path and costmap into a `GuidanceCue` instruction."""

    def __init__(
        self,
        cfg,
        spoken: str = "",
        plural: bool | None = None,
        announce_start: bool = True,
        target_id: int = 0,
        speech=None,
    ) -> None:
        g = cfg.guidance
        self.sm = GuidanceStateMachine(g)
        # head = torso (hackathon assumption): `body.T_cam_body` takes precedence over `guidance.T_cam_head`
        body = getattr(cfg, "body", None)
        T = getattr(body, "T_cam_body", None) if body is not None else None
        self.T_cam_head = np.asarray(T if T is not None else g.T_cam_head, float)
        self.cone = float(g.obstacle_cone_deg)
        self.obstacle_m = float(g.obstacle_m)
        self.corridor = float(g.obstacle_corridor_m)
        self.final_m = float(g.final_direct_m)
        self.arrived_floor = float(g.arrived_m) + float(g.hysteresis_m) + 0.01
        self.approach_lookahead = float(g.approach_lookahead_m)
        self.voice_interval = float(g.voice_min_interval_s)
        self.t_voice = -math.inf
        self.last_state = None
        self.sweep_rate = math.radians(float(cfg.sim.sweep_rate_dps))
        self.approach_m = float(g.approach_m)
        # range of the obstacle search: that of the alert, or that of the obstacle sound
        a = getattr(cfg, "audio", None)
        self.obstacle_range = max(
            self.obstacle_m + 0.3, float(getattr(a, "obstacle_far_m", 0.0) or 0.0)
        )
        self.messages: list[tuple[float, str]] = []
        # continuity of the sound direction: the direction to follow is smoothed in the
        # world frame (a path change glides instead of jumping from one ear to the other), then
        # brought back to the head: a body rotation is still rendered immediately
        self.dir_tau = float(getattr(g, "direction_tau_s", 0.0) or 0.0)
        self.dir_rate = math.radians(float(getattr(g, "direction_max_rate_dps", 0.0) or 0.0))
        self._bearing: float | None = None
        self.dir_safe_m = float(getattr(g, "direction_safe_m", 1.0) or 0.0)
        self.n_dir_snaps = 0
        self._t_bearing = -math.inf
        # voice announcements (object found, approach); furniture provided by the world model
        from visionaid.guidance.narration import Narrator

        tgt = getattr(cfg, "target", None)
        if plural is None:
            plural = bool(getattr(tgt, "plural", False)) if tgt is not None else False
        self.narrator = Narrator(cfg, spoken, plural, announce_start, speech)
        self.landmarks = None
        self.target_id = int(target_id)

    def smooth_bearing(self, t: float, bearing: float) -> float:
        """Smoothed world direction (low-pass `direction_tau_s`, then at most
        `direction_max_rate_dps`). Restarts from the raw direction after an interruption > 1 s."""
        if self._bearing is None or t - self._t_bearing > 1.0 or t < self._t_bearing:
            self._bearing = bearing
        else:
            dt = t - self._t_bearing
            d = math.atan2(math.sin(bearing - self._bearing), math.cos(bearing - self._bearing))
            if self.dir_tau > 0:
                d *= 1.0 - math.exp(-dt / self.dir_tau)
            if self.dir_rate > 0:
                d = float(np.clip(d, -self.dir_rate * dt, self.dir_rate * dt))
            self._bearing = math.atan2(math.sin(self._bearing + d), math.cos(self._bearing + d))
        self._t_bearing = t
        return self._bearing

    def voice(self, t: float, state: str) -> str:
        """Short message on state change, at most one every `voice_min_interval_s`."""
        if state == self.last_state:
            return ""
        if t - self.t_voice < self.voice_interval:
            return ""
        self.last_state = state
        self.t_voice = t
        msg = MESSAGES[state]
        self.messages.append((t, msg))
        return msg

    def update(
        self,
        t: float,
        frame_idx: int,
        pose: Pose | None,
        obj: ObjectEstimate | None,
        path: Path | None,
        cm: Costmap | None,
    ) -> GuidanceCue:
        if obj is not None and getattr(obj, "target_id", 0) != self.target_id:
            obj = None  # object of a previous request
        if path is not None and getattr(path, "target_id", 0) != self.target_id:
            path = None
        lost = pose is None or pose.quality == "lost"
        T_head = None if pose is None else pose.T_world_cam @ self.T_cam_head
        known = obj is not None and obj.state != "unknown" and np.isfinite(obj.position).all()
        dist = math.inf
        obst_d, obst_az = math.inf, 0.0
        if T_head is not None:
            if known:
                _, _, dist = relative_direction(T_head, obj.position)
                if (
                    obj.state == "candidate"
                ):  # probable target: never "arrived" before confirmation
                    dist = max(dist, self.arrived_floor)
            obst_d, obst_az = frontal_obstacle(
                cm, T_head, self.obstacle_range, self.cone, self.corridor
            )
        x = GuidanceInputs(t, lost, known, dist, obst_d, path is not None and path.valid)
        state = self.sm.step(x)
        az = el = 0.0
        alert = 0.0
        guide_az = math.nan
        if T_head is not None:
            if known:
                near = state in (APPROACHING, ARRIVED) or (
                    state == OBSTACLE_ALERT and dist < self.approach_m
                )
                guide_az, el = self._guide_direction(T_head, obj, path, near, state == ARRIVED)
                if state in (NAVIGATING, APPROACHING, ARRIVED, OBSTACLE_ALERT) and (
                    self.dir_tau > 0 or self.dir_rate > 0
                ):
                    pos, fwd, _ = head_frame(T_head)
                    yaw = math.atan2(fwd[1], fwd[0])
                    raw = yaw + guide_az
                    b = self.smooth_bearing(t, raw)
                    lag = abs(math.atan2(math.sin(b - raw), math.cos(b - raw)))
                    if (
                        lag > math.radians(5)
                        and blocked_along(cm, pos, b, self.dir_safe_m, self.corridor)
                        and not blocked_along(cm, pos, raw, self.dir_safe_m, self.corridor)
                    ):
                        # the smoothed direction leads onto an obstacle and the new one is clear
                        # (path replanned to avoid it): safety first, we take it
                        # immediately. If both are cluttered (furnished room), we smooth.
                        b = self._bearing = raw
                        self.n_dir_snaps += 1
                    guide_az = math.atan2(math.sin(b - yaw), math.cos(b - yaw))
                else:
                    self._bearing = None
            else:
                self._bearing = None
            if state in (NAVIGATING, APPROACHING, ARRIVED):
                az = guide_az
            elif state == OBSTACLE_ALERT:
                az = obst_az
                alert = (
                    float(np.clip(1.0 - obst_d / max(self.obstacle_m, 1e-6), 0.0, 1.0))
                    if math.isfinite(obst_d)
                    else 1.0
                )
            elif state == SEARCHING:
                az = math.radians(60.0) * math.sin(self.sweep_rate * t)  # slow sweep instruction
                guide_az = az
            if state != APPROACHING and state != ARRIVED:
                el = 0.0
        if state == TRACKING_LOST:
            alert = 1.0
        # obstacle to signal by sound: object known and not yet very close (the furniture
        # carrying the target is not an obstacle), same rule as the alert
        obst_report = obst_d if known and dist >= self.approach_m else math.inf
        self.narrator.update(t, state, T_head, obj, path, cm, self.landmarks, dist, obst_d)
        speech, speech_id, duck = self.narrator.current(t)
        return GuidanceCue(
            t,
            frame_idx,
            state,
            float(az),
            float(el),
            float(dist),
            alert,
            self.voice(t, state),
            float(guide_az),
            float(obst_report),
            speech,
            speech_id,
            duck,
        )

    def _guide_direction(
        self, T_head, obj, path, near: bool, direct: bool = False
    ) -> tuple[float, float]:
        """(azimuth, elevation) of the path to follow: path aiming point when far; near the target,
        end of the path (clear) while the goal is more than `final_direct_m` away, then the object
        (`direct`: the object straight away)."""
        if not near:
            target = (
                path.lookahead
                if (path is not None and path.valid and np.isfinite(path.lookahead).all())
                else obj.position[:2]
            )
            az, _, _ = relative_direction(T_head, target)
            return az, 0.0
        az, el, _ = relative_direction(T_head, obj.position)
        if not direct and path is not None and path.valid and len(path.points) > 1:
            from visionaid.planning.lookahead import lookahead_point

            if np.linalg.norm(path.goal - T_head[:2, 3]) > self.final_m:
                az, _, _ = relative_direction(
                    T_head, lookahead_point(path.points, self.approach_lookahead)
                )
        return az, el

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        pose = inputs.get("pose_world") or inputs.get("pose")
        if pose is None:
            return None
        return {
            "cue": self.update(
                pose.t,
                pose.frame_idx,
                pose,
                inputs.get("object"),
                inputs.get("path"),
                inputs.get("costmap"),
            )
        }
