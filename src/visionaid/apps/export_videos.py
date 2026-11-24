"""Export of a session as three MP4 videos (H.264, readable by PowerPoint and Keynote).

- `camera.mp4`  : camera image, detections (target in orange), guidance state;
  (the three videos carry the sound guidance track, also written to `guidage.wav`)
- `carte.mp4`   : top-down obstacle map, trajectory, position and heading, path, target;
- `monde3d.mp4` : 3D scene (grid floor, obstacles, point cloud, trajectory, path, target).

The session is replayed in the offline pipeline (same modules as live), frame by frame.

Example:
    uv run python -m visionaid.apps.export_videos data/sessions/20261121-182903_gourde_direct2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from visionaid.mapping.costmap import FREE, OCC
from visionaid.utils.logging import get_logger

log = get_logger("export")

# colours (BGR for OpenCV) — same roles as in the Rerun view
C_USER = (214, 120, 42)  # bleu #2a78d6
C_TARGET = (52, 104, 235)  # orange #eb6834
C_PATH = (40, 170, 40)  # vert
C_OCC = (40, 40, 170)  # rouge sombre
C_INFL = (210, 190, 235)  # light pink: safety margin
C_UNKNOWN = (128, 128, 128)
STATE_FR = {
    "SEARCHING": "SEARCHING",
    "NAVIGATING": "NAVIGATING",
    "APPROACHING": "APPROACHING",
    "ARRIVED": "ARRIVED",
    "OBSTACLE_ALERT": "OBSTACLE ALERT",
    "TRACKING_LOST": "TRACKING LOST",
}


def replay(session_dir, cfg=None, cloud_every: int = 2, with_audio: bool = False):
    """Replays the session; returns, per frame, what is needed for the three videos, and the rate.

    With `with_audio`, also returns the stereo track of the sound guidance (same sound director as
    the phone: `audio.director`) and its sample rate."""
    from visionaid.io.session import SessionImu, SessionSource, load_session
    from visionaid.mapping.geometry import backproject, transform
    from visionaid.perception.depth import video_depth_estimator
    from visionaid.perception.pipeline import run_video_perception, unpack_masks
    from visionaid.perception.target import load_config_for_target
    from visionaid.pipeline import OfflinePipeline

    s = load_session(session_dir)
    cfg = cfg or load_config_for_target(s.meta.get("target") or "où est ma gourde")
    if s.camera_height_m:
        cfg.body.camera_height_m = s.camera_height_m
    dets, _ = run_video_perception(cfg, s.dir)
    by = {d.frame_idx: d for d in dets}
    pipe = OfflinePipeline(cfg, None, video_depth_estimator(cfg, s.dir), imu_source=SessionImu(s))
    src = SessionSource.from_config(cfg, s.dir)
    out, t0, last_cloud = [], None, None
    for k, f in enumerate(src):
        r = pipe.step(f, by.get(f.frame_idx))
        t0 = f.t if t0 is None else t0
        d = r["dets"]
        pw = r["pose_world"]
        cm = r["costmap"]
        if pw is not None and r["depth"] is not None and k % cloud_every == 0:
            pc, uv = backproject(r["depth"].depth, f.K, 8, None, 0.2, 5.0)
            if len(pc):
                last_cloud = transform(pw.T_world_cam, pc).astype(np.float32)
        masks = unpack_masks(d, f.image.shape[1]) if d is not None and d.masks is not None else None
        obj = r["object"]
        p = r["path"]
        out.append(
            {
                "t": float(f.t - t0),
                "image": f.image,
                "boxes": None if d is None else d.boxes.copy(),
                "scores": None if d is None or d.reid_scores is None else d.reid_scores.copy(),
                "target_idx": -1 if d is None else d.target_idx,
                "cand_idx": -1 if d is None else getattr(d, "candidate_idx", -1),
                "target_mask": masks[d.target_idx]
                if masks is not None and d is not None and d.target_idx >= 0
                else None,
                "T": None if pw is None else pw.T_world_cam.copy(),
                "cm": None
                if cm is None
                else (cm.state.copy(), ~np.isfinite(cm.cost), cm.origin.copy(), cm.resolution),
                "obj": obj.position.copy() if np.isfinite(obj.position).all() else None,
                "obj_state": obj.state,
                "path": p.points.copy() if p is not None and p.valid else None,
                "la": p.lookahead.copy() if p is not None and p.valid else None,
                "state": r["cue"].state,
                "dist": float(r["cue"].distance),
                "cloud": last_cloud,
            }
        )
    if not with_audio:
        return out, float(src.fps)
    from visionaid.audio.renderer import AudioRenderer
    from visionaid.utils.config import resolve_path

    ar = AudioRenderer(cfg.audio, getattr(cfg, "tts", None), resolve_path(cfg.paths.cache) / "tts")
    t_first = pipe.records[0].t if pipe.records else 0.0
    duration = len(out) / float(src.fps)
    stereo = ar.render(pipe.cues, t_first, duration)
    return out, float(src.fps), stereo, ar


# ---------------------------------------------------------------------------- camera
def render_camera(fr: dict, height: int = 720) -> np.ndarray:
    img = fr["image"].copy()
    if fr["target_mask"] is not None:
        over = img.copy()
        over[fr["target_mask"]] = C_TARGET
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    if fr["boxes"] is not None:
        for i, b in enumerate(fr["boxes"]):
            x0, y0, x1, y1 = (int(v) for v in b)
            if i == fr["target_idx"]:
                col, lab, th = C_TARGET, "GOURDE", 3
            elif i == fr["cand_idx"]:
                col, lab, th = C_TARGET, "probable bottle", 2
            else:
                col, lab, th = (200, 200, 200), "", 1
            cv2.rectangle(img, (x0, y0), (x1, y1), col, th)
            if lab:
                sc = fr["scores"][i] if fr["scores"] is not None else 0
                cv2.putText(
                    img, f"{lab} {sc:.2f}", (x0, max(14, y0 - 6)), 0, 0.5, col, 2, cv2.LINE_AA
                )
    s = img.shape[0] / height
    img = cv2.resize(img, (int(round(img.shape[1] / s)) // 2 * 2, height))
    _banner(img, fr)
    return img


def _ascii(t: str) -> str:
    """OpenCV does not display accents: É → E, etc."""
    import unicodedata

    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def _banner(img: np.ndarray, fr: dict) -> None:
    txt = _ascii(STATE_FR.get(fr["state"], fr["state"]))
    if np.isfinite(fr["dist"]):
        txt += f"   {fr['dist']:.1f} m"
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(img, txt, (12, 28), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img, f"{fr['t']:.1f} s", (img.shape[1] - 90, 28), 0, 0.7, (200, 200, 200), 2, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------- 2D map
def map_extent(frames: list[dict], margin: float = 0.6) -> tuple[float, float, float, float]:
    """Extent (x0, x1, y0, y1) of the known and walked areas, over the whole session."""
    xs, ys = [], []
    for fr in frames:
        if fr["T"] is not None:
            xs.append(fr["T"][0, 3])
            ys.append(fr["T"][1, 3])
        if fr["obj"] is not None:
            xs.append(fr["obj"][0])
            ys.append(fr["obj"][1])
    last = next((fr["cm"] for fr in reversed(frames) if fr["cm"] is not None), None)
    if last is not None:
        st, _, origin, res = last
        ii, jj = np.nonzero(st != -1)
        if len(ii):
            xs += [origin[0] + jj.min() * res, origin[0] + (jj.max() + 1) * res]
            ys += [origin[1] + ii.min() * res, origin[1] + (ii.max() + 1) * res]
    if not xs:
        return -3, 3, -3, 3
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    side = max(x1 - x0, y1 - y0)  # square: same scale in x and y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def activity_extent(frames: list[dict], margin: float = 2.0) -> tuple[float, float, float, float]:
    """Tighter extent for 3D: trajectory and target (+ margin), not the whole map."""
    xs = [fr["T"][0, 3] for fr in frames if fr["T"] is not None]
    ys = [fr["T"][1, 3] for fr in frames if fr["T"] is not None]
    for fr in frames:
        if fr["obj"] is not None:
            xs.append(fr["obj"][0])
            ys.append(fr["obj"][1])
    if not xs:
        return map_extent(frames)
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def render_map(fr: dict, traj: np.ndarray, ext, size: int = 720) -> np.ndarray:
    x0, x1, y0, y1 = ext
    sc = size / (x1 - x0)
    img = np.full((size, size, 3), C_UNKNOWN, np.uint8)

    def px(x, y):
        return int(round((x - x0) * sc)), int(round((y1 - y) * sc))

    if fr["cm"] is not None:
        st, blocked, origin, res = fr["cm"]
        h, w = st.shape
        # resampling of the grid within the fixed extent
        gx = x0 + (np.arange(size) + 0.5) / sc
        gy = y1 - (np.arange(size) + 0.5) / sc
        jj = np.floor((gx - origin[0]) / res).astype(int)
        ii = np.floor((gy - origin[1]) / res).astype(int)
        okj = (jj >= 0) & (jj < w)
        oki = (ii >= 0) & (ii < h)
        sub = np.full((size, size), -1, np.int8)
        bl = np.zeros((size, size), bool)
        sub[np.ix_(oki, okj)] = st[np.ix_(ii[oki], jj[okj])]
        bl[np.ix_(oki, okj)] = blocked[np.ix_(ii[oki], jj[okj])]
        img[sub == FREE] = (255, 255, 255)
        img[bl & (sub != OCC)] = C_INFL
        img[sub == OCC] = C_OCC
    # quadrillage 1 m
    for gx in np.arange(np.ceil(x0), x1, 1.0):
        cv2.line(img, px(gx, y0), px(gx, y1), (170, 170, 170), 1)
    for gy in np.arange(np.ceil(y0), y1, 1.0):
        cv2.line(img, px(x0, gy), px(x1, gy), (170, 170, 170), 1)
    if len(traj) > 1:
        cv2.polylines(
            img, [np.array([px(*p) for p in traj], np.int32)], False, C_USER, 2, cv2.LINE_AA
        )
    if fr["path"] is not None and len(fr["path"]) > 1:
        cv2.polylines(
            img, [np.array([px(*p) for p in fr["path"]], np.int32)], False, C_PATH, 4, cv2.LINE_AA
        )
        if fr["la"] is not None and np.isfinite(fr["la"]).all():
            cv2.circle(img, px(*fr["la"]), 7, C_PATH, -1, cv2.LINE_AA)
    if fr["obj"] is not None:
        c = px(fr["obj"][0], fr["obj"][1])
        cv2.circle(img, c, 11, C_TARGET, -1, cv2.LINE_AA)
        cv2.circle(img, c, 11, (255, 255, 255), 2, cv2.LINE_AA)
    if fr["T"] is not None:
        T = fr["T"]
        p = px(T[0, 3], T[1, 3])
        fwd = T[:2, 2] / max(np.linalg.norm(T[:2, 2]), 1e-6)
        tip = px(T[0, 3] + 0.5 * fwd[0], T[1, 3] + 0.5 * fwd[1])
        cv2.arrowedLine(img, p, tip, C_USER, 4, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(img, p, 9, C_USER, -1, cv2.LINE_AA)
    _banner(img, fr)
    # legend
    y = size - 16
    for col, lab in (
        (C_USER, "vous"),
        (C_PATH, "chemin"),
        (C_TARGET, "gourde"),
        (C_OCC, "obstacle"),
        (C_INFL, "marge"),
    ):
        cv2.rectangle(img, (12, y - 12), (28, y + 2), col, -1)
        cv2.putText(img, lab, (34, y), 0, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y -= 22
    cv2.putText(img, "quadrillage : 1 m", (size - 170, size - 14), 0, 0.5, (40, 40, 40), 1)
    return img


# ---------------------------------------------------------------------------- 3D world
class World3DRenderer:
    """Matplotlib rendering (off screen) of a fixed 3D view, frame by frame."""

    def __init__(self, ext, size=(1280, 720), elev: float = 38.0, azim: float = -125.0) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.ext = ext
        self.size = size
        self.elev, self.azim = elev, azim
        self.fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor="#fcfcfb")
        self.ax = self.fig.add_axes([-0.05, -0.08, 1.1, 1.08], projection="3d")

    def render(self, fr: dict, traj: np.ndarray) -> np.ndarray:
        ax = self.ax
        ax.cla()
        x0, x1, y0, y1 = self.ext
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(0, 2.0)
        ax.set_box_aspect((x1 - x0, y1 - y0, 2.0), zoom=1.3)
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_axis_off()
        for gx in np.arange(np.ceil(x0), x1, 1.0):
            ax.plot([gx, gx], [y0, y1], [0, 0], color="#d8d7d2", lw=0.6)
        for gy in np.arange(np.ceil(y0), y1, 1.0):
            ax.plot([x0, x1], [gy, gy], [0, 0], color="#d8d7d2", lw=0.6)
        if fr["cm"] is not None:
            st, _, origin, res = fr["cm"]
            ii, jj = np.nonzero(st == OCC)
            if len(ii):
                xs = origin[0] + (jj + 0.5) * res
                ys = origin[1] + (ii + 0.5) * res
                m = (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
                ax.scatter(
                    xs[m],
                    ys[m],
                    np.full(m.sum(), 0.04),
                    s=4,
                    c="#aa2828",
                    marker="s",
                    depthshade=False,
                )
        if fr["cloud"] is not None:
            pc = fr["cloud"]
            m = (
                (pc[:, 0] > x0)
                & (pc[:, 0] < x1)
                & (pc[:, 1] > y0)
                & (pc[:, 1] < y1)
                & (pc[:, 2] < 2.0)
            )
            pc = pc[m]
            z = np.clip(pc[:, 2] / 2.0, 0, 1)[:, None]
            cols = np.array([42, 120, 214]) / 255 * (1 - z) + np.array([235, 104, 52]) / 255 * z
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1.2, c=cols, depthshade=False, alpha=0.8)
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], np.full(len(traj), 0.01), color="#2a78d6", lw=2.5)
        if fr["path"] is not None and len(fr["path"]) > 1:
            P = fr["path"]
            ax.plot(P[:, 0], P[:, 1], np.full(len(P), 0.05), color="#28aa28", lw=4)
        if fr["obj"] is not None:
            o = fr["obj"]
            ax.plot([o[0], o[0]], [o[1], o[1]], [0, o[2]], color="#eb6834", lw=2)
            ax.scatter(
                [o[0]], [o[1]], [o[2]], s=160, c="#eb6834", edgecolors="white", depthshade=False
            )
        if fr["T"] is not None:
            T = fr["T"]
            fwd = T[:2, 2] / max(np.linalg.norm(T[:2, 2]), 1e-6)
            ax.quiver(T[0, 3], T[1, 3], 0.02, fwd[0], fwd[1], 0, length=0.6, color="#2a78d6", lw=3)
            ax.plot(
                [T[0, 3], T[0, 3]],
                [T[1, 3], T[1, 3]],
                [0, T[2, 3]],
                color="#2a78d6",
                lw=1.5,
                ls=":",
            )
            ax.scatter([T[0, 3]], [T[1, 3]], [T[2, 3]], s=60, c="#2a78d6", depthshade=False)
        txt = STATE_FR.get(fr["state"], fr["state"])
        if np.isfinite(fr["dist"]):
            txt += f"   {fr['dist']:.1f} m"
        self.fig.suptitle(f"{txt}      {fr['t']:.1f} s", x=0.02, ha="left", fontsize=15)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------- video
class Mp4Writer:
    """Writes images to MP4 H.264 (yuv420p) via ffmpeg, OpenCV fallback otherwise."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = Path(path)
        self.fps = fps
        self.proc = None
        self.cv = None

    def write(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if w % 2 or h % 2:
            img = img[: h - h % 2, : w - w % 2]
            h, w = img.shape[:2]
        if self.proc is None and self.cv is None:
            if shutil.which("ffmpeg"):
                self.proc = subprocess.Popen(
                    [
                        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{w}x{h}", "-r", f"{self.fps:.3f}", "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", str(self.path),
                    ],
                    stdin=subprocess.PIPE,
                )  # fmt: skip
            else:
                self.cv = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                )
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=300)
        if self.cv is not None:
            self.cv.release()


def export(session_dir, out_dir, which=("camera", "carte", "monde3d")) -> dict[str, Path]:
    frames, fps, stereo, ar = replay(session_dir, with_audio=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = map_extent(frames)
    writers = {w: Mp4Writer(out / f"{w}.mp4", fps) for w in which}
    r3d = World3DRenderer(activity_extent(frames)) if "monde3d" in which else None
    traj: list[np.ndarray] = []
    for k, fr in enumerate(frames):
        if fr["T"] is not None:
            traj.append(fr["T"][:2, 3].copy())
        tr = np.array(traj) if traj else np.zeros((0, 2))
        if "camera" in writers:
            writers["camera"].write(render_camera(fr))
        if "carte" in writers:
            writers["carte"].write(render_map(fr, tr, ext))
        if r3d is not None:
            writers["monde3d"].write(r3d.render(fr, tr))
        if k % 50 == 0:
            log.info("Export: frame %d / %d", k, len(frames))
    for w in writers.values():
        w.close()
    # sound guidance: WAV track, then added to each video (the sound of the full chain)
    wav = out / "guidage.wav"
    ar.write_wav(wav, stereo)
    res = {}
    for w in which:
        v = out / f"{w}.mp4"
        res[w] = v
        if shutil.which("ffmpeg"):
            tmp = out / f"_{w}_son.mp4"
            ok = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(v), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(tmp),
                ],
                timeout=300,
            )  # fmt: skip
            if ok.returncode == 0:
                tmp.replace(v)
    res["guidage"] = wav
    return res


def main(argv=None) -> dict:
    from visionaid.utils.config import resolve_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("session")
    ap.add_argument(
        "--out", default=None, help="output folder (default: outputs/videos/<session>)"
    )
    ap.add_argument("--only", nargs="+", choices=["camera", "carte", "monde3d"], default=None)
    args = ap.parse_args(argv)
    name = Path(args.session).name
    out = resolve_path(args.out or f"outputs/videos/{name}")
    res = export(args.session, out, tuple(args.only or ("camera", "carte", "monde3d")))
    for k, v in res.items():
        print(f"{k:8s} {v}")
    return res


if __name__ == "__main__":
    main()
