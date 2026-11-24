"""High-level planner: costmap + user position + object → Path."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from visionaid.bus.messages import Costmap, ObjectEstimate, Path
from visionaid.mapping.costmap import FREE, OCC, UNKNOWN
from visionaid.planning.astar import astar
from visionaid.planning.dstar_lite import DStarLite
from visionaid.planning.grid import traversal_cost
from visionaid.planning.lookahead import lookahead_point, polyline_length, smooth_path


def cell_of(xy: np.ndarray, origin: np.ndarray, res: float) -> tuple[int, int]:
    return int(np.floor((xy[1] - origin[1]) / res)), int(np.floor((xy[0] - origin[0]) / res))


def world_of(ij: tuple[int, int], origin: np.ndarray, res: float) -> np.ndarray:
    return np.array([origin[0] + (ij[1] + 0.5) * res, origin[1] + (ij[0] + 0.5) * res])


def nearest_passable(
    cc: np.ndarray,
    ij: tuple[int, int],
    radius_cells: int,
    prefer: np.ndarray | None = None,
    toward: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Traversable cell closest to ij (within a radius), preferring `prefer` (e.g. free).

    At equal distance (within one cell), ties are broken by proximity to `toward` (the user).
    """
    h, w = cc.shape
    i0, j0 = ij
    r = radius_cells
    a0, a1 = max(i0 - r, 0), min(i0 + r + 1, h)
    b0, b1 = max(j0 - r, 0), min(j0 + r + 1, w)
    if a0 >= a1 or b0 >= b1:
        return None
    sub = np.isfinite(cc[a0:a1, b0:b1])
    for mask in ([sub & prefer[a0:a1, b0:b1]] if prefer is not None else []) + [sub]:
        if not mask.any():
            continue
        ii, jj = np.nonzero(mask)
        d = np.sqrt((ii + a0 - i0) ** 2 + (jj + b0 - j0) ** 2)
        if d.min() > r:
            continue
        cand = d <= d.min() + 1.0
        if toward is not None:
            du = (ii + a0 - toward[0]) ** 2 + (jj + b0 - toward[1]) ** 2
            k = int(np.argmin(np.where(cand, du, np.inf)))
        else:
            k = int(np.argmin(d))
        return int(ii[k] + a0), int(jj[k] + b0)
    return None


class Planner:
    """Chooses the goal (free cell closest to the object), plans (A* or D* Lite), smooths."""

    def __init__(self, cfg) -> None:
        p = cfg.planning
        self.algorithm = str(p.algorithm)
        self.unknown_cost = float(p.unknown_cost)
        self.allow_unknown = bool(p.allow_unknown)
        # no known path to the object: path up to the edge of the known area
        self.frontier_fallback = bool(getattr(p, "frontier_fallback", True))
        self.n_frontier = 0
        self.lookahead_m = float(p.lookahead_m)
        self.search_r = float(p.target_search_radius_m)
        self.body_clearance = float(cfg.guidance.obstacle_corridor_m)
        self.dstar: DStarLite | None = None
        self._dkey = None
        self.n_plans = 0
        self.n_fail = 0
        # stability (phase 25): commitment to the previous path, aiming at the object at the goal
        self.commit_ratio = float(getattr(p, "commit_ratio", 0.0) or 0.0)
        self.commit_margin = float(getattr(p, "commit_margin_m", 0.0) or 0.0)
        self.near_goal = float(getattr(p, "near_goal_m", 0.0) or 0.0)
        self.prev_pts: np.ndarray | None = None
        self.prev_goal: np.ndarray | None = None
        self.n_kept = 0
        # approaching the object (phase 27): "within arm's reach" goal that is quickest to reach,
        # reduced margin near the object (going between two chairs to grab the bottle)
        self.reach = float(getattr(p, "reach_m", 0.0) or 0.0)
        self.zone = float(getattr(p, "approach_zone_m", 0.0) or 0.0)
        self.zone_clear = float(getattr(p, "approach_clearance_m", 0.0) or 0.0)
        self.zone_penalty = float(getattr(p, "approach_penalty", 3.0))
        self.goal_keep = float(getattr(p, "goal_keep_ratio", 0.15))
        self.n_reach_goal = 0
        self.detour_ratio = float(getattr(p, "approach_detour_ratio", 1.5))
        self.detour_slack = float(getattr(p, "approach_detour_slack_m", 1.0))
        self.n_relaxed = 0

    def plan(self, cm: Costmap, user_xy: np.ndarray, obj: ObjectEstimate) -> Path:
        """Path to the object through **known** areas (seen as free) only.

        Without `allow_unknown`, a never-seen area is never proposed, even if shorter. If
        no known path reaches the object (map still partial), the path stops at the
        edge of the known area, on the best estimated route (`frontier_fallback`): the user goes
        there, the map grows, and the full path appears.
        """
        t, idx = cm.t, cm.frame_idx
        empty = Path(t, idx, np.zeros((0, 2)), np.full(2, np.nan), np.full(2, np.nan), False, 0.0)
        if obj is None or not np.isfinite(obj.position).all():
            return empty
        res, origin = cm.resolution, cm.origin
        cc_known = traversal_cost(cm.cost, cm.state, self.unknown_cost, self.allow_unknown)
        h, w = cc_known.shape
        s_ij = cell_of(user_xy, origin, res)
        o_ij = cell_of(obj.position[:2], origin, res)
        if not (0 <= s_ij[0] < h and 0 <= s_ij[1] < w):
            return empty
        o_ij = (int(np.clip(o_ij[0], 0, h - 1)), int(np.clip(o_ij[1], 0, w - 1)))
        found = self._search(cm, cc_known, s_ij, o_ij, obj, self.allow_unknown)
        frontier = False
        if found is None and not self.allow_unknown and self.frontier_fallback:
            cc_unk = traversal_cost(cm.cost, cm.state, self.unknown_cost, True)
            found = self._search(cm, cc_unk, s_ij, o_ij, obj, True)
            if found is not None:
                goal, cells, start, _ = found
                cells = self._cut_at_unknown(cm.state, cells)
                if cells is None:
                    found = None
                else:
                    found = (cells[-1], cells, start, cc_known)
                    frontier = True
                    self.n_frontier += 1
        if found is None:
            self.n_fail += 1
            return empty
        goal, cells, start, cc = found
        self.n_plans += 1
        cells = smooth_path(cc, cells)
        pts = np.array([world_of(c, origin, res) for c in cells])
        pts = (
            np.vstack([user_xy[None, :2], pts[1:]])
            if len(pts) > 1
            else np.vstack([user_xy[None, :2], pts])
        )
        goal_xy = world_of(goal, origin, res)
        kept = self._keep_previous(cc, origin, res, user_xy[:2], goal_xy, polyline_length(pts))
        if kept is not None:
            pts = kept
            goal_xy = kept[-1].copy()  # the published goal is the end of the path actually followed
            self.n_kept += 1
        self.prev_pts, self.prev_goal = pts.copy(), goal_xy.copy()
        length = polyline_length(pts)
        if not frontier and self.near_goal > 0 and length < self.near_goal:
            # at the goal: the direction of a point a few centimetres away no longer makes sense, we aim
            # at the object itself (stable), as guidance does at the end of the approach
            la = np.asarray(obj.position[:2], float).copy()
        else:
            la = self._lookahead_los(cm, pts, s_ij, start, origin, res)
        return Path(t, idx, pts, la, goal_xy, True, length)

    def _search(self, cm, cc, s_ij, o_ij, obj, allow_unknown: bool):
        """Goal and path cells on the cost grid `cc`; (goal, cells, start, cc)
        or None."""
        res, origin = cm.resolution, cm.origin
        d_obj = None
        strict = np.isfinite(cc)  # traversable with the normal margin
        cc_strict = cc
        if self.zone > 0 and self.zone_clear > 0:
            cc, d_obj = self._relax_near_object(cm, cc, obj, res, origin, allow_unknown)
        goal = nearest_passable(cc, o_ij, int(self.search_r / res), cm.state == FREE, toward=s_ij)
        if goal is None:
            return None
        start = s_ij
        if not np.isfinite(cc[start]):
            # the user is in an inflated zone (noise, nearby furniture) or not yet
            # mapped: start from the closest traversable cell
            start = nearest_passable(cc, s_ij, int(1.0 / res))
            if start is None:
                return None
        cells = None
        if self.reach > 0:
            reach = self._reach_goal(cc, start, obj, res, origin, d_obj, strict)
            if reach is not None and cc is not cc_strict and np.isfinite(cc_strict[start]):
                # the reduced margin is only used if it avoids a real detour: otherwise we keep the
                # path with the normal margin (going around a sudden obstacle, etc.)
                normal = self._reach_goal(cc_strict, start, obj, res, origin, d_obj, strict)
                if normal is not None:
                    ln = len(normal[1]) * res
                    lr = len(reach[1]) * res
                    if ln <= self.detour_ratio * lr + self.detour_slack:
                        reach = normal
                        cc = cc_strict
                    else:
                        self.n_relaxed += 1
            if reach is not None:
                goal, cells = reach
                self.n_reach_goal += 1
        if cells is None:
            if self.algorithm == "dstar" and not (allow_unknown and not self.allow_unknown):
                key = (cc.shape, tuple(np.round(origin, 6)), goal)
                if self.dstar is None or key != self._dkey:
                    self.dstar = DStarLite(cc, goal)
                    self._dkey = key
                else:
                    self.dstar.set_costs(cc)
                cells, _cost = self.dstar.plan(start)
            else:
                cells, _cost = astar(cc, start, goal)
        if cells is None:
            return None
        return goal, cells, start, cc

    def _cut_at_unknown(self, state, cells):
        """Cuts the path at the first unknown cell (after the starting cells); None
        if nothing useful remains (< 2 known cells)."""
        k = 0
        while k < len(cells) and state[cells[k]] == UNKNOWN:  # start not mapped yet
            k += 1
        e = k
        while e < len(cells) and state[cells[e]] != UNKNOWN:
            e += 1
        out = cells[:e]
        return out if e - k >= 2 else None

    def _relax_near_object(self, cm, cc, obj, res, origin, allow_unknown: bool = True):
        """Near the object (≤ `approach_zone_m`), the margin goes from `robot_radius_m` to
        `approach_clearance_m` (half body width). These cells stay more expensive
        (`approach_penalty`): they are only used if this really shortens the route."""
        h, w = cc.shape
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        xs = origin[0] + (jj + 0.5) * res
        ys = origin[1] + (ii + 0.5) * res
        d_obj = np.hypot(xs - obj.position[0], ys - obj.position[1])
        clear = distance_to_obstacles(cm.state, res)
        relax = (
            ~np.isfinite(cc) & (cm.state != OCC) & (clear >= self.zone_clear) & (d_obj <= self.zone)
        )
        if not allow_unknown:  # the reduced margin never makes an unknown area traversable
            relax &= cm.state != UNKNOWN
        if relax.any():
            cc = cc.copy()
            cc[relax] = self.zone_penalty + np.where(cm.state[relax] < 0, self.unknown_cost, 0.0)
        return cc, d_obj

    def _reach_goal(self, cc, start, obj, res, origin, d_obj, strict=None):
        """Goal "within arm's reach": among the traversable cells less than `reach_m` from
        the object (you grab a bottle on a table from the edge, no need to go
        around it), the one with the cheapest route from the user. Hysteresis:
        the previous goal is kept if it stays within +15% of the best one.
        Returns (goal, path) or None (fallback to the "closest cell" goal)."""
        from visionaid.planning.astar import _astar_kernel

        h, w = cc.shape
        if d_obj is None:
            jj, ii = np.meshgrid(np.arange(w), np.arange(h))
            d_obj = np.hypot(
                origin[0] + (jj + 0.5) * res - obj.position[0],
                origin[1] + (ii + 0.5) * res - obj.position[1],
            )
        cand = (d_obj <= self.reach) & np.isfinite(cc)
        if not cand.any():
            return None
        flat = np.ascontiguousarray(cc, np.float64).ravel()
        s = start[0] * w + start[1]
        g, parent = _astar_kernel(flat, h, w, s, -1, False)  # costs from the user
        g = g.reshape(h, w)
        # the goal keeps the normal margin if such a reachable cell exists within arm's
        # reach; it only enters the reduced-margin band otherwise (object in the middle of a
        # table). Without this preference, the simulator agent ended up stuck to the furniture.
        if strict is not None:
            pref = cand & strict & np.isfinite(g)
            if pref.any():
                cand = pref
        score = np.where(cand, g + 3.0 * d_obj / res, np.inf)  # at equal cost, near the object
        k = int(np.argmin(score))
        if not np.isfinite(score.flat[k]):
            return None
        goal = (k // w, k % w)
        if self.prev_goal is not None:
            pg = cell_of(self.prev_goal, origin, res)
            if 0 <= pg[0] < h and 0 <= pg[1] < w and np.isfinite(score[pg]):
                if score[pg] <= score.flat[k] * (1.0 + self.goal_keep) + 2.0:
                    goal = pg
        cells = []
        u = goal[0] * w + goal[1]
        while u != -1:
            cells.append((int(u // w), int(u % w)))
            u = int(parent[u])
        cells.reverse()
        return goal, cells

    def _keep_previous(self, cc, origin, res, user_xy, goal_xy, new_len) -> np.ndarray | None:
        """Previous path, shortened from the user, if it stays practicable and not
        clearly longer than the new one (avoids alternating between two almost equivalent
        detours on a noisy map). None otherwise."""
        self.keep_why = "no previous path"
        if self.commit_ratio <= 0 or self.prev_pts is None or len(self.prev_pts) < 2:
            return None
        self.keep_why = "goal moved"
        if np.linalg.norm(goal_xy - self.prev_goal) > 0.15:
            # goal moved (even slightly): new path. Otherwise the kept path would end at
            # the old goal while guidance compares the position with the new one (agent stuck
            # on approach in the simulator, corridor: 94% → 67% success)
            return None
        prev = self.prev_pts
        # point of the previous path closest to the user (projection onto the segments)
        best, best_k, best_p = np.inf, 0, prev[0]
        for k in range(len(prev) - 1):
            a, b = prev[k], prev[k + 1]
            ab = b - a
            u = float(np.clip(np.dot(user_xy - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
            p = a + u * ab
            d = float(np.linalg.norm(user_xy - p))
            if d < best:
                best, best_k, best_p = d, k, p
        self.keep_why = "off the path"
        if best > 0.5:
            return None  # the user has moved away from the previous path
        rest = np.vstack([user_xy[None], best_p[None], prev[best_k + 1 :]])
        rest = rest[np.r_[True, np.linalg.norm(np.diff(rest, axis=0), axis=1) > 1e-6]]
        h, w = cc.shape
        self.keep_why = "blocked"
        for a, b in zip(rest[:-1], rest[1:], strict=True):  # practicable on the new map
            n = max(int(np.linalg.norm(b - a) / (0.5 * res)), 1)
            for s_ in np.linspace(0.0, 1.0, n + 1):
                i, j = cell_of(a + s_ * (b - a), origin, res)
                if not (0 <= i < h and 0 <= j < w) or not np.isfinite(cc[i, j]):
                    return None
        self.keep_why = "too long"
        if polyline_length(rest) > new_len * (1.0 + self.commit_ratio) + self.commit_margin:
            return None
        self.keep_why = "kept"
        return rest

    def _lookahead_los(self, cm: Costmap, pts, user_ij, start, origin, res) -> np.ndarray:
        """Farthest look-ahead point (≤ lookahead_m) in the user's line of sight.

        Line of sight is tested from the real position on a grid where only the cells
        less than `body_clearance_m` from an obstacle are blocking: the given direction never leads
        straight onto an obstacle, even when the user has drifted off the path. Without
        this constraint, the chord towards a point located after a turn cuts the corner.
        """
        from visionaid.planning.lookahead import line_of_sight

        clear = distance_to_obstacles(cm.state, res)
        body = np.where(clear >= self.body_clearance, 0.0, np.inf)
        if not self.allow_unknown:  # the aim does not cut through a never-seen area either
            body[cm.state == UNKNOWN] = np.inf
        h, w = body.shape
        src = user_ij if np.isfinite(body[user_ij]) else start
        d = self.lookahead_m
        while d > 0.15:
            p = lookahead_point(pts, d)
            ij = cell_of(p, origin, res)
            if 0 <= ij[0] < h and 0 <= ij[1] < w and line_of_sight(body, src, ij):
                return p
            d -= 0.1
        # no safe aim: first go back to the clear starting cell
        return world_of(start, origin, res) if start != user_ij else lookahead_point(pts, 0.3)

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        cm, obj, pose = inputs.get("costmap"), inputs.get("object"), inputs.get("pose_world")
        if cm is None or obj is None or pose is None:
            return None
        return {"path": self.plan(cm, pose.T_world_cam[:2, 3], obj)}


def distance_to_obstacles(state: np.ndarray, res: float) -> np.ndarray:
    from visionaid.mapping.costmap import OCC

    occ = state == OCC
    return distance_transform_edt(~occ) * res if occ.any() else np.full(state.shape, np.inf)
