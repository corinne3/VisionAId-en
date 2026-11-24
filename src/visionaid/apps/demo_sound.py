"""Sound demo: the project's simulated scenario rendered as a video with the guidance sound.

For each scenario (`sim.scenarios`: room, corridor, sudden obstacle), the simulated agent follows
the instructions; we write the video (ground truth on the left, built map on the right, sound played
at the bottom) and the stereo track produced by `AudioRenderer`, then put them together.

Outputs: `outputs/demo_sons/<scenario>_<seed>.mp4` (video + sound) and `.wav`.
Listen with headphones. For the real apartment video, use `run_offline` (which writes
`annotated_audio.mp4` with the same sounds): it needs the vision models.

    uv run python -m visionaid.apps.demo_sound --scenario room:0 corridor:1 appearing:0
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from visionaid.audio.renderer import AudioRenderer
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.logging import get_logger

log = get_logger("demo_sound")

MODE_FR = {
    "search": "search",
    "guide": "path",
    "lost": "tracking lost: stop",
    "arrived": "arrived",
    "silent": "silence",
}


def _side(az: float) -> str:
    d = round(math.degrees(az))
    if d == 0:
        return "ahead"
    if abs(d) >= 179:
        return "behind"
    return f"{abs(d)} deg {'left' if d > 0 else 'right'}"


def caption(cmd) -> str:
    """Text of the sound played (ASCII: OpenCV font)."""
    m = MODE_FR.get(cmd.mode, cmd.mode)
    if cmd.mode == "search":
        return f"Sound: {m}, one note every {cmd.interval_s:.2f} s"
    if cmd.mode == "guide":
        dist = f", goal at {cmd.distance:.1f} m" if math.isfinite(cmd.distance) else ""
        ob = ", OBSTACLE" if cmd.obstacle_db > -100 else ""
        return (
            f"Sound: {m} - direction {_side(cmd.azimuth)}{dist}, "
            f"one note every {cmd.interval_s:.2f} s{ob}"
        )
    return f"Sound: {m}"


def render_scenario(cfg, name: str, seed: int, out_dir: Path, fps: int = 10) -> dict:
    from visionaid.sim.agent import run_episode
    from visionaid.sim.render import render_episode
    from visionaid.sim.scenarios import SCENARIOS

    world = SCENARIOS[name](seed)
    r = run_episode(cfg, world, seed, record=True)
    cues = list(r.cues)
    if r.success and cues and cues[-1].state != "ARRIVED":
        # the simulation stops at 0.5 m (success): we emit the arrival instruction that would follow
        last = cues[-1]
        cues.append(replace(last, t=last.t + 0.1, frame_idx=last.frame_idx + 1, state="ARRIVED"))
        r.frames.append((last.t + 0.1, *r.frames[-1][1:4], "ARRIVED", *r.frames[-1][5:]))
    ar = AudioRenderer(cfg.audio)
    cmds = ar.commands(cues)
    stem = out_dir / f"{name}_{seed}"
    video = render_episode(
        world,
        r.frames,
        stem.with_suffix(".silent.mp4"),
        fps=fps,
        captions=[caption(c) for c in cmds],
    )
    duration = len(r.frames) / fps + 2.0  # 2 more seconds: the arrival sound finishes
    wav = stem.with_suffix(".wav")
    ar.write_wav(wav, ar.render(cues, cues[0].t if cues else 0.0, duration))
    out = stem.with_suffix(".mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav),
                "-filter_complex", "[0:v]tpad=stop_mode=clone:stop_duration=2[v]",
                "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out),
            ],
            check=True,
            timeout=600,
        )  # fmt: skip
        video.unlink(missing_ok=True)
    else:
        log.warning("ffmpeg missing: silent video and separate sound (%s)", wav)
        out = video
    modes: dict[str, int] = {}
    for c in cmds:
        modes[c.mode] = modes.get(c.mode, 0) + 1
    return {
        "scenario": name,
        "seed": seed,
        "success": bool(r.success),
        "time_s": round(r.time_s, 1),
        "collisions": r.collisions,
        "modes": modes,
        "video": str(out),
        "wav": str(wav),
    }


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scenario",
        nargs="+",
        default=["room:0", "corridor:1", "appearing:0"],
        help="scenario[:seed]; default: search then guidance, obstacle, sudden obstacle",
    )
    ap.add_argument("--out", default=None, help="output folder (default outputs/demo_sons)")
    a = ap.parse_args(argv)
    cfg = load_config()
    out = Path(a.out) if a.out else resolve_path(cfg.paths.outputs) / "demo_sons"
    out.mkdir(parents=True, exist_ok=True)
    picks = [(p.split(":")[0], int(p.split(":")[1]) if ":" in p else 0) for p in a.scenario]
    res = [render_scenario(cfg, name, seed, out) for name, seed in picks]
    for r in res:
        print(
            f"{r['scenario']:10s} success={r['success']} t={r['time_s']} s "
            f"collisions={r['collisions']} → {r['video']}"
        )
    return res


if __name__ == "__main__":
    main()
