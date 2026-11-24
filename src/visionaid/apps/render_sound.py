"""Sound guidance of a recorded session: WAV track (and camera video with the sound).

The session is replayed once in the offline pipeline (same modules as live); the
guidance instructions are kept in `consignes.json`. Later runs only redo
the audio, in a few seconds: handy for tuning the sounds (`config/default.yaml`, section
`audio`) and listening again right away. `--recompute` replays the vision (after a guidance change).

Outputs in `outputs/son/<session>/`: `guidage.wav` (with headphones), `consignes.json`, and with
`--video` `camera_son.mp4` (camera image, sound played written at the bottom, sound track).

    uv run python -m visionaid.apps.render_sound data/sessions/20261121-182903_gourde_direct2 --video
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from visionaid.bus.messages import GuidanceCue
from visionaid.utils.logging import get_logger

log = get_logger("render_sound")
CACHE = "consignes.json"


def compute_cues(session_dir) -> dict:
    """Replays the session (vision, map, path, guidance); timestamped instructions + duration."""
    from visionaid.io.session import SessionImu, SessionSource, load_session
    from visionaid.perception.depth import video_depth_estimator
    from visionaid.perception.pipeline import run_video_perception
    from visionaid.perception.target import load_config_for_target
    from visionaid.pipeline import OfflinePipeline

    s = load_session(session_dir)
    cfg = load_config_for_target(s.meta.get("target") or "où est ma gourde")
    if s.camera_height_m:
        cfg.body.camera_height_m = s.camera_height_m
    dets, _ = run_video_perception(cfg, s.dir)
    by = {d.frame_idx: d for d in dets}
    pipe = OfflinePipeline(cfg, None, video_depth_estimator(cfg, s.dir), imu_source=SessionImu(s))
    src = SessionSource.from_config(cfg, s.dir)
    n = 0
    for f in src:
        pipe.step(f, by.get(f.frame_idx))
        n += 1
        if n % 100 == 0:
            log.info("Replay: frame %d", n)
    t0 = pipe.records[0].t if pipe.records else 0.0
    return {
        "session": str(Path(session_dir).resolve().name),
        "t0": float(t0),
        "fps": float(src.fps),
        "n_frames": n,
        "duration": n / float(src.fps),
        "cues": [dataclasses.asdict(c) for c in pipe.cues],
    }


def load_or_compute(session_dir, out: Path, recompute: bool = False) -> dict:
    p = out / CACHE
    if p.exists() and not recompute:
        log.info("Instructions reused from %s (--recompute to replay the vision)", p)
        return json.loads(p.read_text())
    data = compute_cues(session_dir)
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return data


def caption(cmd) -> str:
    """Text of the sound played (ASCII: OpenCV font)."""
    if cmd is None:
        return "Sound: silence"
    if cmd.mode in ("guide", "search"):
        p = cmd.pan
        side = (
            "centre" if abs(p) < 0.05 else f"{abs(p) * 100:.0f}% {'left' if p < 0 else 'right'}"
        )
        txt = "path" if cmd.mode == "guide" else "search"
        s = f"Sound: {txt}, pan {side}, one note / {cmd.interval_s:.2f} s"
        if cmd.obstacle_db > -100:
            s += f"  | obstacle {cmd.obstacle_db:.0f} dB"
        return s
    return {"lost": "Sound: tracking lost", "arrived": "Sound: arrived"}.get(
        cmd.mode, "Sound: silence"
    )


def write_video(session_dir, out: Path, data: dict, cmds, wav: Path, height: int = 720) -> Path:
    from visionaid.apps.export_videos import Mp4Writer
    from visionaid.io.session import SessionSource
    from visionaid.perception.target import load_config_for_target

    cfg = load_config_for_target("où est ma gourde")
    src = SessionSource.from_config(cfg, session_dir)
    ts = [c.t for c in cmds]
    silent = out / "_camera.mp4"
    w = Mp4Writer(silent, data["fps"])
    for f in src:
        k = bisect.bisect_right(ts, f.t) - 1
        cmd = cmds[k] if k >= 0 else None
        img = f.image
        s = img.shape[0] / height
        img = cv2.resize(img, (int(round(img.shape[1] / s)) // 2 * 2, height))
        cv2.rectangle(img, (0, height - 40), (img.shape[1], height), (0, 0, 0), -1)
        state = cmd.state if cmd is not None else ""
        cv2.putText(img, f"{state}  {caption(cmd)}", (10, height - 13), 0, 0.6, (255, 255, 255),
                    1, cv2.LINE_AA)  # fmt: skip
        w.write(img)
    w.close()
    video = out / "camera_son.mp4"
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg missing: silent video, separate sound (%s)", wav)
        silent.replace(video)
        return video
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav),
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(video),
        ],
        check=True,
        timeout=600,
    )  # fmt: skip
    silent.unlink(missing_ok=True)
    return video


def main(argv=None) -> dict:
    from visionaid.audio.renderer import AudioRenderer
    from visionaid.utils.config import load_config, resolve_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("session", help="session folder (data/sessions/<date>_<name>)")
    ap.add_argument(
        "--out", default=None, help="output folder (default: outputs/son/<session>)"
    )
    ap.add_argument("--recompute", action="store_true", help="replays the vision even if cached")
    ap.add_argument("--video", action="store_true", help="also writes camera_son.mp4")
    args = ap.parse_args(argv)
    name = Path(args.session).resolve().name
    out = resolve_path(args.out or f"outputs/son/{name}")
    data = load_or_compute(args.session, out, args.recompute)
    cues = [GuidanceCue(**c) for c in data["cues"]]
    cfg = load_config()
    ar = AudioRenderer(cfg.audio, getattr(cfg, "tts", None), resolve_path(cfg.paths.cache) / "tts")
    stereo = ar.render(cues, data["t0"], data["duration"])
    wav = out / "guidage.wav"
    ar.write_wav(wav, stereo)
    cmds = ar.commands(cues)
    modes: dict[str, int] = {}
    for c in cmds:
        modes[c.mode] = modes.get(c.mode, 0) + 1
    n_ob = sum(c.obstacle_db > -100 for c in cmds)
    res = {
        "guidage": wav,
        "modes": modes,
        "obstacle_s": n_ob / max(len(cmds), 1) * data["duration"],
    }
    if args.video:
        res["video"] = write_video(args.session, out, data, cmds, wav)
    print(f"Duration {data['duration']:.1f} s, instructions: {modes}")
    print(f"Obstacle audible for ~{res['obstacle_s']:.1f} s")
    for k in ("guidage", "video"):
        if k in res:
            print(f"{k:8s} {res[k]}")
    return res


if __name__ == "__main__":
    main()
