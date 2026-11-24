import cv2
import numpy as np
import pytest

from visionaid.mapping.costmap import FREE, OCC
from visionaid.mapping.world_model import export_real_map, run_mapping_video
from visionaid.utils.config import resolve_path
from visionaid.utils.report import update_report
from visionaid.viz.costmap_png import render_costmap

pytestmark = [pytest.mark.slow, pytest.mark.data]


@pytest.fixture(scope="module")
def mapping(video_perception, cfg):
    dets, _ = video_perception
    return run_mapping_video(cfg, {d.frame_idx: d for d in dets})


def test_costmap_nontrivial(mapping, cfg):
    wm, rec = mapping
    s = wm.grid.state()
    known = s != -1
    free = float((s == FREE).sum() / known.sum())
    occ = float((s == OCC).sum() / known.sum())
    traj = np.array([r["T_world_cam"][:3, 3] for r in rec if r["T_world_cam"] is not None])
    obj = rec[-1]["object"].position
    img = render_costmap(s, wm.grid.origin, wm.grid.res, traj, obj, cost=wm.grid.cost(s))
    out = resolve_path(cfg.paths.outputs) / "costmap_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    update_report(
        "mapping",
        {"free_frac_known": free, "occ_frac_known": occ, "known_area_m2": float(known.sum() * wm.grid.res**2), "costmap_png": str(out)},
    )
    export_real_map(wm, rec, resolve_path(cfg.paths.outputs) / "real_map.npz")
    assert out.exists()
    assert free > 0.20 and occ > 0.02


def test_object_position_stable(mapping):
    _, rec = mapping
    P = np.array([r["object"].position for r in rec if r["target_seen"] and r["object"].state == "seen"])
    assert len(P) > 10
    dev = np.linalg.norm(P - P.mean(0), axis=1)
    std = float(np.sqrt((dev**2).mean()))
    update_report("mapping", {"object_pos_std_m": std, "object_pos_mean": P.mean(0), "object_height_m": float(P[:, 2].mean())})
    assert std < 0.30


def test_camera_height_above_floor(mapping):
    """(phase 5) Camera height above the floor between 0.8 and 1.9 m, standard deviation < 0.15 m."""
    wm, _ = mapping
    h = np.array(wm.cam_heights)
    update_report("vo", {"camera_height_median_m": float(np.median(h)), "camera_height_std_m": float(h.std())})
    assert 0.8 <= np.median(h) <= 1.9
    assert h.std() < 0.15
