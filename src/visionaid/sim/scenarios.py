"""Scenario generators: furnished room, corridor with a door, sudden obstacle, real map."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from visionaid.sim.world import SimWorld

RES = 0.05
CLEAR_START = 0.45  # minimal clearance at the start


def _empty(w_m: float, h_m: float, margin: float = 0.5) -> SimWorld:
    W = int(round((w_m + 2 * margin) / RES))
    H = int(round((h_m + 2 * margin) / RES))
    occ = np.zeros((H, W), bool)
    world = SimWorld(occ, np.array([-margin, -margin]), RES, np.zeros(3), np.zeros(3))
    t = 0.1  # 10 cm walls
    world.add_box(-margin, -margin, w_m + margin, -0.0)
    world.add_box(-margin, h_m, w_m + margin, h_m + margin)
    world.add_box(-margin, -margin, 0.0, h_m + margin)
    world.add_box(w_m, -margin, w_m + margin, h_m + margin)
    _ = t
    return world


def _place_object_on(world: SimWorld, box, rng) -> None:
    """Object placed near the edge of the furniture (reachable by hand), 0.8 m high."""
    x0, y0, x1, y1 = box
    side = rng.integers(4)
    inset = 0.08
    if side == 0:
        p = (rng.uniform(x0 + 0.1, x1 - 0.1), y0 + inset)
    elif side == 1:
        p = (rng.uniform(x0 + 0.1, x1 - 0.1), y1 - inset)
    elif side == 2:
        p = (x0 + inset, rng.uniform(y0 + 0.1, y1 - 0.1))
    else:
        p = (x1 - inset, rng.uniform(y0 + 0.1, y1 - 0.1))
    world.obj = np.array([p[0], p[1], 0.8])
    world.obj_support = np.array(box)


def _free_start(world: SimWorld, rng, cond, tries: int = 2000) -> bool:
    H, W = world.occ.shape
    for _ in range(tries):
        x = world.origin[0] + rng.uniform(0, W * RES)
        y = world.origin[1] + rng.uniform(0, H * RES)
        xy = np.array([x, y])
        if world.clearance(xy) < CLEAR_START:
            continue
        if not cond(xy):
            continue
        world.start = np.array([x, y, rng.uniform(-math.pi, math.pi)])
        return True
    return False


def room(seed: int) -> SimWorld:
    """Rectangular room with random furniture; object on a piece of furniture, visible within 4 m of the start."""
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        w, h = rng.uniform(5.0, 7.0), rng.uniform(4.0, 6.0)
        world = _empty(w, h)
        world.name = "room"
        boxes = []
        for _ in range(rng.integers(3, 7)):
            bw, bh = rng.uniform(0.4, 1.4), rng.uniform(0.4, 1.0)
            x0, y0 = rng.uniform(0.3, w - bw - 0.3), rng.uniform(0.3, h - bh - 0.3)
            world.add_box(x0, y0, x0 + bw, y0 + bh)
            boxes.append((x0, y0, x0 + bw, y0 + bh))
        _place_object_on(world, boxes[rng.integers(len(boxes))], rng)
        ok = _free_start(
            world,
            rng,
            lambda xy: (
                1.5 < np.linalg.norm(xy - world.obj[:2]) < 4.0
                and world.visible(xy, world.obj, math.atan2(*(world.obj[:2] - xy)[::-1]), 70, 4.0)
            ),
        )
        if ok and _reachable(world):
            return world
    raise RuntimeError("room scenario not found")


def corridor(seed: int) -> SimWorld:
    """Two rooms connected by a door (0.9 m); the object is in the other room, already in memory."""
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        w, h = rng.uniform(7.0, 9.0), rng.uniform(3.0, 4.5)
        world = _empty(w, h)
        world.name = "corridor"
        xw = w / 2 + rng.uniform(-0.8, 0.8)
        door_y = rng.uniform(0.6, h - 1.5)
        world.add_box(xw, 0.0, xw + 0.1, door_y)
        world.add_box(xw, door_y + 0.9, xw + 0.1, h)
        boxes = []
        for _ in range(rng.integers(1, 4)):
            bw, bh = rng.uniform(0.4, 1.0), rng.uniform(0.4, 0.8)
            x0, y0 = rng.uniform(xw + 0.6, w - bw - 0.3), rng.uniform(0.3, h - bh - 0.3)
            if abs((y0 + bh / 2) - (door_y + 0.45)) < 0.9 and x0 < xw + 1.3:
                continue
            world.add_box(x0, y0, x0 + bw, y0 + bh)
            boxes.append((x0, y0, x0 + bw, y0 + bh))
        if not boxes:
            continue
        _place_object_on(world, boxes[rng.integers(len(boxes))], rng)
        world.known_object = True
        ok = _free_start(world, rng, lambda xy: xy[0] < xw - 0.8)
        if ok and _reachable(world):
            return world
    raise RuntimeError("corridor scenario not found")


def appearing_obstacle(seed: int) -> SimWorld:
    """Room + obstacle (0.6×0.6 m) that appears on the route after 4 s."""
    world = room(seed + 10_000)
    world.name = "appearing"
    world.events.append((4.0, "box_ahead", 0.6))
    return world


def real_map(seed: int, path: str | Path | None, body: float = 0.25) -> SimWorld | None:
    """Real map = costmap from the video (occupied cells), start and object from the run."""
    if path is None or not Path(path).exists():
        return None
    d = np.load(path)
    occ = d["state"] == 1
    world = SimWorld(
        occ.copy(), d["origin"].astype(float), float(d["res"]), d["obj"].astype(float), np.zeros(3)
    )
    world.name = "real"
    world.known_object = True
    rng = np.random.default_rng(seed)
    start = d["start"].astype(float)
    # start: clear cell closest to the beginning of the trajectory, random heading
    dist = world.dist
    ii, jj = np.nonzero(dist >= max(CLEAR_START, body + 0.1))
    if len(ii) == 0:
        return None
    xy = np.stack(
        [world.origin[0] + (jj + 0.5) * world.res, world.origin[1] + (ii + 0.5) * world.res], 1
    )
    k = int(np.argmin(np.linalg.norm(xy - start[:2], axis=1) + rng.uniform(0, 0.3, len(xy))))
    world.start = np.array([xy[k, 0], xy[k, 1], rng.uniform(-math.pi, math.pi)])
    if not _reachable(world):
        return None
    return world


BODY_RADIUS = 0.35  # = costmap.robot_radius_m: clearance required by the planner


def _reachable(world: SimWorld, body: float = BODY_RADIUS) -> bool:
    from visionaid.sim.agent import optimal_length

    return math.isfinite(optimal_length(world, body))


SCENARIOS = {"room": room, "corridor": corridor, "appearing": appearing_obstacle}
