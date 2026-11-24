import numpy as np
import pytest

from visionaid.io.sources import VideoFileSource
from visionaid.mapping.geometry import inv_T, rot_angle_deg
from visionaid.perception.depth import video_depth_estimator
from visionaid.pose.source import run_vo_video
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.slow, pytest.mark.data]


@pytest.fixture(scope="module")
def vo_run(video_path, cfg):
    return run_vo_video(cfg)


def test_no_jumps(vo_run, cfg):
    poses, _ = vo_run
    worst_t, worst_r = 0.0, 0.0
    for a, b in zip(poses[:-1], poses[1:], strict=True):
        if a.quality == "lost" or b.quality == "lost":
            continue
        rel = inv_T(a.T_world_cam) @ b.T_world_cam
        worst_t = max(worst_t, float(np.linalg.norm(rel[:3, 3])))
        worst_r = max(worst_r, rot_angle_deg(rel[:3, :3]))
    update_report("vo", {"max_step_m": worst_t, "max_step_deg": worst_r})
    assert worst_t < 0.5 and worst_r < 30.0


def test_lost_fraction(vo_run):
    poses, vo = vo_run
    lost = sum(p.quality == "lost" for p in poses) / len(poses)
    P = np.array([p.T_world_cam[:3, 3] for p in poses])
    update_report(
        "vo",
        {
            "lost_fraction": lost,
            "n_keyframes": len(vo.vo.keyframes),
            "path_length_m": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
            "net_displacement_m": float(np.linalg.norm(P[-1] - P[0])),
            "median_inliers": float(np.median([p.n_inliers for p in poses if p.quality == "ok"])),
        },
    )
    assert lost < 0.10


def test_forward_backward_consistency(video_path, cfg):
    """10 s segment read forwards then backwards: the final pose comes back near the start."""
    src = VideoFileSource.from_config(cfg)
    n = int(round(10 * src.fps))
    start = max(0, (src.frame_count - n) // 2)
    frames = []
    while (f := src.read()) is not None:
        if start <= f.frame_idx < start + n:
            frames.append(f)
    seq = frames + frames[-2::-1]
    poses, _ = run_vo_video(cfg, frames=seq, depth_est=video_depth_estimator(cfg))
    P = np.array([p.T_world_cam[:3, 3] for p in poses])
    dist = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    err = float(np.linalg.norm(P[-1] - P[0]))
    rot = rot_angle_deg(poses[-1].T_world_cam[:3, :3])
    update_report("vo", {"roundtrip_dist_m": dist, "roundtrip_err_m": err, "roundtrip_rel": err / max(dist, 1e-9), "roundtrip_rot_deg": rot})
    assert dist > 0.5
    assert err / dist < 0.10
