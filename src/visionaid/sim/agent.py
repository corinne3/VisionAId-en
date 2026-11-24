"""Virtual user and closed simulation loop (real mapping, planning, guidance)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from visionaid.bus.messages import Costmap, GuidanceCue, Pose
from visionaid.guidance.cues import GuidanceEngine
from visionaid.mapping.costmap import OccupancyGrid
from visionaid.mapping.object_anchor import ObjectAnchor
from visionaid.planning.astar import astar
from visionaid.planning.planner import Planner
from visionaid.sim.world import SimSensor, SimWorld

COLLISION_RADIUS = 0.20  # half shoulder width: contact if the centre is < 20 cm from an obstacle
HEAD_HEIGHT = 1.4


def head_pose(x: float, y: float, heading: float) -> np.ndarray:
    """T_world_cam of a head at 1.4 m looking horizontally along `heading` (OpenCV)."""
    c, s = math.cos(heading), math.sin(heading)
    R = np.array(
        [[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]]
    )  # columns: x right, y down, z forward
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, HEAD_HEIGHT]
    return T


def optimal_length(world: SimWorld, body: float = 0.35, success_m: float = 0.5) -> float:
    """Length (m) of the shortest ground-truth path to the arrival area (< success_m)."""
    cc = np.where(world.dist >= body, 0.0, np.inf)
    s = world.cell(world.start[:2])
    if not world.inside(s) or not np.isfinite(cc[s]):
        return math.inf
    H, W = cc.shape
    ii, jj = np.nonzero(np.isfinite(cc))
    xy = np.stack(
        [world.origin[0] + (jj + 0.5) * world.res, world.origin[1] + (ii + 0.5) * world.res], 1
    )
    d = np.linalg.norm(xy - world.obj[:2], axis=1)
    cand = np.nonzero(d < success_m - world.res)[0]
    if len(cand) == 0:
        return math.inf
    best = math.inf
    for k in cand[np.argsort(d[cand])][:5]:
        p, L = astar(cc, s, (int(ii[k]), int(jj[k])))
        if p is not None:
            best = min(best, L * world.res)
    return best


@dataclass
class Agent:
    """Virtual user: reaction delay, angular noise, walking speed."""

    x: float
    y: float
    heading: float
    speed: float
    turn_rate: float
    sweep_rate: float
    reaction: float
    noise: float
    rng: np.random.Generator
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    _noise_val: float = 0.0
    _alert_dir: float = 0.0
    _noise_t: float = -1e9

    def heading_at(self, t: float) -> float:
        for tt, h in reversed(self.history):
            if tt <= t + 1e-9:
                return h
        return self.heading

    def step(
        self, t: float, dt: float, cue: GuidanceCue | None, world: SimWorld
    ) -> tuple[float, bool]:
        """Moves one step forward; returns (distance travelled, collision)."""
        self.history.append((t, self.heading))
        if t - self._noise_t >= 1.0:
            self._noise_val = self.rng.normal(0.0, self.noise)
            self._noise_t = t
        v = 0.0
        if cue is not None:
            st = cue.state
            if st in ("NAVIGATING", "APPROACHING"):
                target = self.heading_at(cue.t) + cue.azimuth + self._noise_val
                err = math.atan2(math.sin(target - self.heading), math.cos(target - self.heading))
                turn = float(np.clip(err, -self.turn_rate * dt, self.turn_rate * dt))
                self.heading += turn
                err -= turn
                if abs(err) < math.radians(30):  # turn first, then move forward
                    v = self.speed * math.cos(err) * (0.6 if st == "APPROACHING" else 1.0)
            elif st == "SEARCHING":
                self.heading += self.sweep_rate * dt
            elif st == "OBSTACLE_ALERT":
                # moves away from the alert sound; the direction is chosen when entering the alert then kept
                if self._alert_dir == 0.0:
                    a = cue.azimuth
                    self._alert_dir = (
                        -math.copysign(1.0, a)
                        if abs(a) > math.radians(5)
                        else float(self.rng.choice([-1.0, 1.0]))
                    )
                self.heading += self._alert_dir * 0.5 * self.turn_rate * dt
            if st != "OBSTACLE_ALERT":
                self._alert_dir = 0.0
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        if v <= 0:
            return 0.0, False
        nx = self.x + v * dt * math.cos(self.heading)
        ny = self.y + v * dt * math.sin(self.heading)
        if world.clearance(np.array([nx, ny])) < COLLISION_RADIUS:
            return 0.0, True  # contact: the user is blocked
        d = math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        return d, False


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    success: bool
    time_s: float
    collisions: int
    traveled_m: float
    optimal_m: float
    replans: int
    final_dist_m: float
    states: dict
    events_applied: int = 0
    frames: list | None = None
    cues: list | None = None  # emitted instructions (if record), for the sound rendering

    @property
    def ratio(self) -> float:
        return (
            self.traveled_m / self.optimal_m
            if self.optimal_m > 0 and math.isfinite(self.optimal_m)
            else math.inf
        )


def _place_box_ahead(
    world: SimWorld, agent: Agent, path_pts: np.ndarray | None, size: float, rng
) -> bool:
    """Sudden obstacle ~1.2 m ahead, on the planned path if possible."""
    cands = []
    if path_pts is not None and len(path_pts) > 1:
        from visionaid.planning.lookahead import lookahead_point

        for d in (1.4, 1.6, 1.8, 2.0):
            cands.append(lookahead_point(path_pts, d))
    for d in (1.4, 1.6):
        cands.append(
            np.array([agent.x + d * math.cos(agent.heading), agent.y + d * math.sin(agent.heading)])
        )
    for c in cands:
        # front face at ≥ 1 m: reaction distance (0.3 s + hold time at 0.6 m/s)
        if np.linalg.norm(c - [agent.x, agent.y]) < 1.3 or np.linalg.norm(c - world.obj[:2]) < 1.0:
            continue
        saved = world.occ.copy()
        world.add_box(c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2)
        tmp_start = world.start
        world.start = np.array([agent.x, agent.y, agent.heading])
        ok = (
            math.isfinite(optimal_length(world))
            and world.clearance(np.array([agent.x, agent.y])) >= COLLISION_RADIUS
        )
        world.start = tmp_start
        if ok:
            return True
        world.occ = saved
        world._dist = None
    return False


def run_episode(
    cfg, world: SimWorld, seed: int, record: bool = False, max_time: float | None = None
) -> EpisodeResult:
    """Simulates a full episode with the real mapping, planning and guidance modules."""
    sc = cfg.sim
    rng = np.random.default_rng(seed)
    dt = float(sc.dt)
    max_time = float(max_time or sc.max_time_s)
    grid = OccupancyGrid(cfg.costmap)
    span = max(world.occ.shape) * world.res
    grid.ensure(
        np.array([world.origin, world.origin + np.array(world.occ.shape[::-1]) * world.res]),
        margin=1.0,
    )
    _ = span
    anchor = ObjectAnchor(cfg.anchor)
    planner = Planner(cfg)
    engine = GuidanceEngine(cfg)
    sensor = SimSensor(sc, rng)
    agent = Agent(
        float(world.start[0]),
        float(world.start[1]),
        float(world.start[2]),
        float(sc.speed_mps),
        math.radians(float(sc.turn_rate_dps)),
        math.radians(float(sc.sweep_rate_dps)),
        float(sc.reaction_s),
        math.radians(float(sc.heading_noise_deg)),
        rng,
    )
    optimal = optimal_length(world, float(cfg.costmap.robot_radius_m))
    if world.known_object:
        anchor.update_world(world.obj + rng.normal(0, 0.05, 3), 0.0, 1.0, 2.0)
    cues: deque = deque()
    all_cues: list = []
    path = None
    cm_cost = None
    traveled, collisions, replans = 0.0, 0, 0
    in_contact = False
    states: dict[str, int] = {}
    frames = [] if record else None
    events = list(world.events)
    success = False
    n_events = 0
    t = 0.0
    k = 0
    while t < max_time:
        # events (sudden obstacle)
        for ev in list(events):
            if t >= ev[0] and ev[1] == "box_ahead":
                n_events += _place_box_ahead(
                    world,
                    agent,
                    None if path is None or not path.valid else path.points,
                    ev[2],
                    rng,
                )
                events.remove(ev)
        # simulated perception
        T = head_pose(agent.x, agent.y, agent.heading)
        pts = sensor.scan(world, agent.x, agent.y, agent.heading)
        grid.update(pts, np.array([agent.x, agent.y, HEAD_HEIGHT]), dt)
        carve = float(getattr(cfg.costmap, "carve_radius_m", 0.0) or 0.0)
        if carve > 0:  # same rule as the real pipeline: where the user walks, it is free
            grid.carve(np.array([agent.x, agent.y]), carve)
        if world.visible(
            np.array([agent.x, agent.y]),
            world.obj,
            agent.heading,
            float(sc.fov_deg),
            float(sc.range_m),
        ):
            d = float(np.linalg.norm(world.obj[:2] - [agent.x, agent.y]))
            anchor.update_world(world.obj + rng.normal(0, 0.03 + 0.01 * d, 3), t, 0.9, d)
        state = grid.state()
        if k % int(sc.replan_every) == 0 or cm_cost is None or cm_cost.shape != state.shape:
            cm_cost = grid.cost(state)
            cm = Costmap(t, k, state, cm_cost, grid.origin.copy(), grid.res)
            obj = anchor.estimate(t, k)
            if obj.state != "unknown":
                path = planner.plan(cm, np.array([agent.x, agent.y]), obj)
                replans += 1
        cm = Costmap(t, k, state, cm_cost, grid.origin.copy(), grid.res)
        obj = anchor.estimate(t, k)
        cue = engine.update(t, k, Pose(t, k, T, "ok"), obj, path, cm)
        states[cue.state] = states.get(cue.state, 0) + 1
        cues.append(cue)
        if record:
            all_cues.append(cue)
        delayed = None
        while cues and cues[0].t <= t - agent.reaction + 1e-9:
            delayed = cues.popleft()
        if delayed is not None:
            agent._last_cue = delayed
        dist_step, coll = agent.step(t, dt, getattr(agent, "_last_cue", None), world)
        traveled += dist_step
        if coll and not in_contact:
            collisions += 1
        in_contact = coll
        if record:
            frames.append(
                (
                    t,
                    agent.x,
                    agent.y,
                    agent.heading,
                    cue.state,
                    None if path is None or not path.valid else path.points.copy(),
                    state.copy(),
                    grid.origin.copy(),
                )
            )
        if np.linalg.norm(world.obj[:2] - [agent.x, agent.y]) < float(sc.success_m):
            success = True
            break
        t += dt
        k += 1
    final = float(np.linalg.norm(world.obj[:2] - [agent.x, agent.y]))
    return EpisodeResult(
        world.name,
        seed,
        success,
        t,
        collisions,
        traveled,
        optimal,
        replans,
        final,
        states,
        n_events,
        frames,
        list(all_cues) if record else None,
    )
