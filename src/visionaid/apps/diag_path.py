"""Path diagnostic on a real session: when guidance starts, validity, failure causes,
start inside an inflated zone, stability of the aiming direction.

Examples:
    uv run python -m visionaid.apps.diag_path data/sessions/20261121-140325_gourde_piece
    uv run python -m visionaid.apps.diag_path --all --out outputs/diag
Results: a JSON summary per session, plus the frame-by-frame detail (`<session>_frames.json`),
used to plot the before/after comparison (`--compare`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visionaid.mapping.costmap import FREE, OCC, UNKNOWN
from visionaid.planning import planner as P
from visionaid.planning.grid import traversal_cost


def _wrap_deg(a: float) -> float:
    return abs((a + 180.0) % 360.0 - 180.0)


def unknown_length(cm, pts: np.ndarray, skip_m: float = 0.3, step: float = 0.05) -> float:
    """Length (m) of the path in unknown cells (excluding the first `skip_m` metres,
    under the user's feet)."""
    if len(pts) < 2:
        return 0.0
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.r_[0.0, np.cumsum(ln)]
    s = np.arange(skip_m, cum[-1], step)
    if not len(s):
        return 0.0
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    j = np.floor((x - cm.origin[0]) / cm.resolution).astype(int)
    i = np.floor((y - cm.origin[1]) / cm.resolution).astype(int)
    h, w = cm.state.shape
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    unk = np.ones(len(s), bool)  # outside the map = unknown
    unk[ok] = cm.state[i[ok], j[ok]] == UNKNOWN
    return float(unk.sum() * step)


def guide_bearing_deg(cue, pose) -> float | None:
    """Direction of the guidance sound in the world frame (°), or None outside guidance."""
    import math

    from visionaid.guidance.cues import head_frame

    if pose is None or cue.state not in ("NAVIGATING", "APPROACHING", "OBSTACLE_ALERT"):
        return None
    if not math.isfinite(cue.guide_azimuth):
        return None
    _, fwd, _ = head_frame(pose.T_world_cam)
    return float(np.degrees(math.atan2(fwd[1], fwd[0]) + cue.guide_azimuth))


def diag_session(session_dir, cfg=None) -> tuple[dict, list[dict]]:
    """Replays the session in the offline pipeline and measures path quality."""
    from visionaid.io.session import SessionImu, SessionSource, load_session
    from visionaid.perception.depth import video_depth_estimator
    from visionaid.perception.pipeline import run_video_perception
    from visionaid.perception.target import load_config_for_target
    from visionaid.pipeline import OfflinePipeline

    s = load_session(session_dir)
    cfg = cfg or load_config_for_target(s.meta.get("target") or "où est ma gourde")
    if s.camera_height_m:
        cfg.body.camera_height_m = s.camera_height_m
    dets, _ = run_video_perception(cfg, s.dir)
    by = {d.frame_idx: d for d in dets}
    pipe = OfflinePipeline(cfg, None, video_depth_estimator(cfg, s.dir), imu_source=SessionImu(s))
    plans: list[dict] = []
    orig = pipe.planner.plan

    def wrapped(cm, user_xy, obj):
        path = orig(cm, user_xy, obj)
        pl = pipe.planner
        cc = traversal_cost(cm.cost, cm.state, pl.unknown_cost, pl.allow_unknown)
        s_ij = P.cell_of(user_xy, cm.origin, cm.resolution)
        h, w = cc.shape
        inside = 0 <= s_ij[0] < h and 0 <= s_ij[1] < w
        known = obj is not None and np.isfinite(obj.position).all()
        rec = {
            "t": float(cm.t),
            "valid": bool(path.valid),
            "length": float(path.length),
            "start_blocked": bool(inside and not np.isfinite(cc[s_ij])),
            "user": [float(x) for x in user_xy[:2]],
            "la": [float(x) for x in path.lookahead] if path.valid else None,
            "points": path.points.tolist() if path.valid else [],
            "unknown_m": unknown_length(cm, path.points) if path.valid else 0.0,
            "keep": getattr(pl, "keep_why", None),
            "goal": [float(x) for x in path.goal] if path.valid else None,
            "obj": [float(x) for x in obj.position[:2]] if known else None,
            "occ": float((cm.state == OCC).mean()),
            "free": float((cm.state == FREE).mean()),
        }
        if not path.valid:
            if not known:
                rec["why"] = "object unknown"
            elif not inside:
                rec["why"] = "user outside the map"
            else:
                rec["why"] = "no path (map cut off or start blocked)"
        plans.append(rec)
        return path

    pipe.planner.plan = wrapped
    frames: list[dict] = []
    t0 = None
    for f in SessionSource.from_config(cfg, s.dir):
        r = pipe.step(f, by.get(f.frame_idx))
        t0 = f.t if t0 is None else t0
        d = r["dets"]
        frames.append(
            {
                "t": float(f.t - t0),
                "state": r["cue"].state,
                "confirmed": bool(d is not None and d.target_idx >= 0),
                "obj_state": r["object"].state,
                "guide_deg": guide_bearing_deg(r["cue"], r["pose_world"]),
                "user": [float(x) for x in r["pose_world"].T_world_cam[:2, 3]]
                if r["pose_world"] is not None
                else None,
            }
        )
    for p in plans:
        p["t"] -= t0 or 0.0
    summ = summarize(s.name, frames, plans)
    first_cand = next((f["t"] for f in frames if f["obj_state"] == "candidate"), None)
    summ["first_candidate_s"] = round(first_cand, 1) if first_cand is not None else None
    summ["candidate_frac"] = round(
        sum(f["obj_state"] == "candidate" for f in frames) / max(len(frames), 1), 3
    )
    summ["candidates_wrong"] = int(getattr(pipe.world, "n_cand_wrong", 0))
    return summ, {"frames": frames, "plans": plans}


def summarize(name: str, frames: list[dict], plans: list[dict]) -> dict:
    """Aggregated indicators of a session."""
    valid = [p for p in plans if p["valid"]]
    why: dict[str, int] = {}
    for p in plans:
        if not p["valid"]:
            why[p["why"]] = why.get(p["why"], 0) + 1
    jumps = []
    for a, b in zip(valid, valid[1:], strict=False):
        if b["t"] - a["t"] > 0.5:
            continue
        da = np.degrees(np.arctan2(a["la"][1] - a["user"][1], a["la"][0] - a["user"][0]))
        db = np.degrees(np.arctan2(b["la"][1] - b["user"][1], b["la"][0] - b["user"][0]))
        jumps.append(_wrap_deg(db - da))
    states = [f["state"] for f in frames]
    first_valid = valid[0]["t"] if valid else None
    first_known = next((f["t"] for f in frames if f["obj_state"] != "unknown"), None)
    return {
        "session": name,
        "frames": len(frames),
        "duration_s": round(frames[-1]["t"], 1) if frames else 0,
        "first_target_known_s": round(first_known, 1) if first_known is not None else None,
        "first_valid_path_s": round(first_valid, 1) if first_valid is not None else None,
        "searching_frac": round(states.count("SEARCHING") / max(len(states), 1), 3),
        "confirmed_frac": round(np.mean([f["confirmed"] for f in frames]), 3) if frames else 0,
        "plans": len(plans),
        "valid_frac_when_known": round(
            len(valid) / max(sum(p["obj"] is not None for p in plans), 1), 3
        ),
        "invalid_why": why,
        "start_blocked_frac": round(float(np.mean([p["start_blocked"] for p in plans])), 3)
        if plans
        else None,
        "aim_jump_deg_med": round(float(np.median(jumps)), 1) if jumps else None,
        "aim_jump_deg_p90": round(float(np.percentile(jumps, 90)), 1) if jumps else None,
        "aim_jumps_over_30": int(np.sum(np.array(jumps) > 30)) if jumps else 0,
        # paths going through never-seen areas (> 20 cm away from the user's feet)
        "unknown_path_frac": round(
            float(np.mean([p.get("unknown_m", 0.0) > 0.2 for p in valid])), 3
        )
        if valid
        else None,
        "unknown_m_max": round(max((p.get("unknown_m", 0.0) for p in valid), default=0.0), 2),
        **_sound_jumps(frames),
        "states": {k: states.count(k) for k in sorted(set(states))},
        "final_state": states[-1] if states else None,
    }


def _sound_jumps(frames: list[dict]) -> dict:
    """Jumps of the guidance sound direction (world frame) from one frame to the next."""
    d = []
    for a, b in zip(frames, frames[1:], strict=False):
        if a.get("guide_deg") is None or b.get("guide_deg") is None or b["t"] - a["t"] > 0.5:
            continue
        d.append(_wrap_deg(b["guide_deg"] - a["guide_deg"]))
    return {
        "sound_jump_deg_p90": round(float(np.percentile(d, 90)), 1) if d else None,
        "sound_jumps_over_20": int(np.sum(np.array(d) > 20)) if d else 0,
        "sound_jumps_over_45": int(np.sum(np.array(d) > 45)) if d else 0,
    }


# colours: validated reference palette (categorical, fixed order) — before = 1, after = 2
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _aim_series(detail: dict) -> tuple[np.ndarray, np.ndarray]:
    """Time and difference (°) between the aiming direction and the object direction."""
    ts, ds = [], []
    for p in detail["plans"]:
        if not p["valid"] or p["obj"] is None:
            continue
        a = np.degrees(np.arctan2(p["la"][1] - p["user"][1], p["la"][0] - p["user"][0]))
        o = np.degrees(np.arctan2(p["obj"][1] - p["user"][1], p["obj"][0] - p["user"][0]))
        ts.append(p["t"])
        ds.append(((a - o + 180.0) % 360.0) - 180.0)
    return np.array(ts), np.array(ds)


def compare(
    before: Path,
    after: Path,
    out_png: Path,
    sim: tuple | None = None,
    title: str = "Path to the object — before / after",
) -> Path:
    """Before/after image: for each session, when a valid path exists, and the error of the
    aiming direction; table of indicators (and of the simulator if provided)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(p.name[: -len("_summary.json")] for p in Path(after).glob("*_summary.json"))
    names = [n for n in names if (Path(before) / f"{n}_summary.json").exists()]
    n = len(names)
    fig = plt.figure(figsize=(13, 3.6 * n + 2.6), facecolor=SURFACE)
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [0.75], width_ratios=[1, 1.25])
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2})
    rows = []
    for k, name in enumerate(names):
        sb = json.loads((Path(before) / f"{name}_summary.json").read_text())
        sa = json.loads((Path(after) / f"{name}_summary.json").read_text())
        db = json.loads((Path(before) / f"{name}_frames.json").read_text())
        da = json.loads((Path(after) / f"{name}_frames.json").read_text())
        short = name.split("_", 1)[1] if "_" in name else name
        # 1) timeline: target known and valid path
        ax = fig.add_subplot(gs[k, 0])
        dur = max(sb["duration_s"], sa["duration_s"])
        for row, (det, col, lab) in enumerate(((da, C_AFTER, "after"), (db, C_BEFORE, "before"))):
            fr = det["frames"]
            for f0, f1 in zip(fr, fr[1:], strict=False):
                if f0["state"] == "SEARCHING":
                    ax.broken_barh([(f0["t"], f1["t"] - f0["t"])], (row - 0.32, 0.64), color=GRID)
            for p0, p1 in zip(det["plans"], det["plans"][1:], strict=False):
                if p0["valid"]:
                    w = min(p1["t"] - p0["t"], 0.5)
                    ax.broken_barh([(p0["t"], w)], (row - 0.32, 0.64), color=col)
            fv = (sa if lab == "after" else sb)["first_valid_path_s"]
            if fv is not None:
                ax.annotate(
                    f"1st path: {fv:.1f} s",
                    (fv, row + 0.36),
                    fontsize=9,
                    color=INK,
                    va="bottom",
                )
        ax.set_yticks([0, 1], ["after", "before"])
        ax.set_ylim(-0.6, 1.75)
        ax.set_xlim(0, dur)
        ax.set_xlabel("time (s)")
        ax.set_title(
            f"{short} — valid path (colour) / searching (grey)", loc="left", fontsize=11
        )
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.set_facecolor(SURFACE)
        # 2) aiming error / object direction
        ax2 = fig.add_subplot(gs[k, 1])
        for det, col, lab in ((db, C_BEFORE, "before"), (da, C_AFTER, "after")):
            t, d = _aim_series(det)
            if len(t):
                ax2.plot(t, d, color=col, lw=2, marker="o", ms=4, label=lab)
        ax2.axhline(0, color=INK2, lw=1)
        ax2.set_ylim(-180, 180)
        ax2.set_yticks([-180, -90, 0, 90, 180])
        ax2.set_xlim(0, dur)
        ax2.grid(axis="y", color=GRID, lw=1)
        ax2.set_xlabel("temps (s)")
        ax2.set_ylabel("aim − object direction (°)")
        ax2.set_title(
            f"{short} — direction given (a detour is normal)", loc="left", fontsize=11
        )
        ax2.legend(frameon=False, loc="upper left", ncol=2)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        ax2.set_facecolor(SURFACE)

        def fmt(v, pct=False):
            if v is None:
                return "—"
            return f"{100 * v:.0f} %" if pct else f"{v}"

        rows += [
            [
                short,
                "1st valid path (s)",
                fmt(sb["first_valid_path_s"]),
                fmt(sa["first_valid_path_s"]),
            ],
            [
                short,
                "time searching",
                fmt(sb["searching_frac"], True),
                fmt(sa["searching_frac"], True),
            ],
            [
                short,
                "valid paths (target known)",
                fmt(sb["valid_frac_when_known"], True),
                fmt(sa["valid_frac_when_known"], True),
            ],
            [
                short,
                "start blocked",
                fmt(sb["start_blocked_frac"], True),
                fmt(sa["start_blocked_frac"], True),
            ],
            [
                short,
                "median aim jump (°)",
                fmt(sb["aim_jump_deg_med"]),
                fmt(sa["aim_jump_deg_med"]),
            ],
            [
                short,
                "aim jumps > 30°",
                fmt(sb["aim_jumps_over_30"]),
                fmt(sa["aim_jumps_over_30"]),
            ],
        ]
    if sim is not None:
        s0, s1 = sim
        for sc, lab in (
            ("room", "room"),
            ("corridor", "corridor"),
            ("appearing", "sudden obstacle"),
        ):
            rows.append(
                [
                    "simulator",
                    f"{lab}: success / collision-free",
                    f"{100 * s0[sc]['success_rate']:.0f} / {100 * s0[sc]['collision_free_rate']:.0f} %",
                    f"{100 * s1[sc]['success_rate']:.0f} / {100 * s1[sc]['collision_free_rate']:.0f} %",
                ]
            )
    axt = fig.add_subplot(gs[n, :])
    axt.axis("off")
    tab = axt.table(
        cellText=rows,
        colLabels=["", "indicator", "before", "after"],
        loc="center",
        cellLoc="left",
        colWidths=[0.16, 0.42, 0.18, 0.18],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(color=INK2, weight="bold")
        elif c == 2:
            cell.set_text_props(color=INK)
        elif c == 3:
            cell.set_text_props(color=INK, weight="bold")
    fig.suptitle(
        title,
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def main(argv=None) -> list[dict]:
    from visionaid.apps.session_info import list_sessions
    from visionaid.utils.config import load_config, resolve_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true", help="all navigation sessions")
    ap.add_argument("--out", default="outputs/diag")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("AVANT", "APRES"),
        help="folders of two diagnostics to compare → before/after image (no new computation)",
    )
    ap.add_argument(
        "--sim", nargs=2, metavar=("SIM_AVANT", "SIM_APRES"), help="simulator results (JSON)"
    )
    ap.add_argument("--png", default="outputs/diag/comparaison.png")
    ap.add_argument("--title", default="Path to the object — before / after")
    args = ap.parse_args(argv)
    if args.compare:
        sim = None
        if args.sim:
            sim = tuple(json.loads(Path(p).read_text()) for p in args.sim)
        png = compare(
            Path(args.compare[0]), Path(args.compare[1]), resolve_path(args.png), sim, args.title
        )
        print(f"Comparaison : {png}")
        return []
    dirs = [Path(p) for p in args.sessions]
    if args.all or not dirs:
        dirs = [
            p
            for p in list_sessions(resolve_path(load_config().session.dir))
            if "calib" not in p.name and "rotations" not in p.name
        ]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for d in dirs:
        summ, detail = diag_session(d)
        (out / f"{summ['session']}_frames.json").write_text(json.dumps(detail))
        (out / f"{summ['session']}_summary.json").write_text(
            json.dumps(summ, indent=2, ensure_ascii=False)
        )
        print(json.dumps(summ, ensure_ascii=False))
        res.append(summ)
    return res


if __name__ == "__main__":
    main()
