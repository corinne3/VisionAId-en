"""Simulator evaluation: N seeds per scenario (in parallel), statistics, videos."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from visionaid.utils.config import load_config, resolve_path, to_dict
from visionaid.utils.logging import get_logger

log = get_logger("sim")
REAL_BODY_RADIUS = 0.25


def _one(args):
    name, seed, cfg_raw, real_map_path = args
    from visionaid.sim.agent import run_episode
    from visionaid.sim.scenarios import SCENARIOS, real_map
    from visionaid.utils.config import _to_dataclass

    if name == "real":  # cluttered real map: reduced body radius (see DECISIONS.md)
        import copy

        cfg_raw = copy.deepcopy(cfg_raw)
        cfg_raw["costmap"]["robot_radius_m"] = REAL_BODY_RADIUS
    cfg = _to_dataclass("config", cfg_raw)
    try:
        world = real_map(seed, real_map_path) if name == "real" else SCENARIOS[name](seed)
        if world is None:
            return None
        r = run_episode(cfg, world, seed)
        return {
            "scenario": name,
            "seed": seed,
            "success": r.success,
            "collisions": r.collisions,
            "time_s": r.time_s,
            "traveled_m": r.traveled_m,
            "optimal_m": r.optimal_m,
            "ratio": r.ratio,
            "final_dist_m": r.final_dist_m,
            "states": r.states,
            "events_applied": r.events_applied,
        }
    except Exception as e:  # noqa: BLE001
        return {"scenario": name, "seed": seed, "error": repr(e)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == name and "error" not in r]
        if not rs:
            continue
        ratios = [r["ratio"] for r in rs if r["success"] and math.isfinite(r["ratio"])]
        out[name] = {
            "n": len(rs),
            "errors": sum(1 for r in rows if r["scenario"] == name and "error" in r),
            "success_rate": float(np.mean([r["success"] for r in rs])),
            "collision_free_rate": float(np.mean([r["collisions"] == 0 for r in rs])),
            "ratio_median": float(np.median(ratios)) if ratios else None,
            "time_median_s": float(
                np.median([r["time_s"] for r in rs if r["success"]] or [np.nan])
            ),
            "failed_seeds": [r["seed"] for r in rs if not r["success"]],
            "collision_seeds": [r["seed"] for r in rs if r["collisions"] > 0],
            "events_applied_rate": float(np.mean([r.get("events_applied", 0) > 0 for r in rs])),
            "success_rate_with_event": (
                float(np.mean([r["success"] for r in rs if r.get("events_applied", 0) > 0]))
                if any(r.get("events_applied", 0) > 0 for r in rs)
                else None
            ),
        }
    return out


def evaluate(
    cfg=None,
    n_seeds: int | None = None,
    scenarios: tuple[str, ...] = ("room", "corridor", "appearing", "real"),
    workers: int | None = None,
    real_map_path=None,
) -> tuple[dict, list[dict]]:
    cfg = cfg or load_config()
    n = int(n_seeds or int(os.environ.get("VISIONAID_SIM_SEEDS", cfg.sim.n_seeds)))
    raw = to_dict(cfg)
    real_map_path = str(real_map_path or resolve_path(cfg.paths.outputs) / "real_map.npz")
    jobs = [(s, k, raw, real_map_path) for s in scenarios for k in range(n)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    # compile the numba kernels once (disk cache) before starting the workers:
    # a concurrent compilation in all the workers made them crash
    from visionaid.audio.binaural import spatialize
    from visionaid.planning.astar import warmup as wa
    from visionaid.planning.dstar_lite import warmup as wd

    wa()
    wd()
    spatialize(np.zeros(16), 0.0)
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(workers) as pool:
        futs = [(j, pool.apply_async(_one, (j,))) for j in jobs]
        for j, f in futs:
            try:
                r = f.get(timeout=600)
            except Exception as e:  # noqa: BLE001 - dead or stuck worker: recorded as an error
                r = {"scenario": j[0], "seed": j[1], "error": repr(e)}
            if r is not None:
                rows.append(r)
    return summarize(rows), rows


def render_examples(
    cfg, out_dir: Path, picks: list[tuple[str, int]], real_map_path=None
) -> list[str]:
    from visionaid.sim.agent import run_episode
    from visionaid.sim.render import render_episode
    from visionaid.sim.scenarios import SCENARIOS, real_map

    paths = []
    for name, seed in picks:
        world = real_map(seed, real_map_path) if name == "real" else SCENARIOS[name](seed)
        if world is None:
            continue
        r = run_episode(cfg, world, seed, record=True)
        p = render_episode(world, r.frames, out_dir / f"{name}_{seed}.mp4")
        paths.append(str(p))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args(argv)
    summ, _ = evaluate(n_seeds=args.seeds)
    for k, v in summ.items():
        log.info("%s : %s", k, v)
    return summ


if __name__ == "__main__":
    main()
