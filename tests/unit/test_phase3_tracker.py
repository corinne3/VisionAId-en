import numpy as np
import pytest

from visionaid.perception.detector import StubDetector
from visionaid.perception.reid import ColorModel, StubEmbedder, TargetReID, hue_shift, masked_crop
from visionaid.perception.tracker import TargetTracker, iou_matrix

pytestmark = pytest.mark.unit
IMG = np.zeros((240, 320, 3), np.uint8)
BOX_A = np.array([100, 80, 140, 180], float)
BOX_B = np.array([220, 60, 260, 160], float)


def run(tracker, seq):
    """seq: list of lists (box, reid_score) per frame → list of target_track_id."""
    out = []
    for k, items in enumerate(seq):
        det = StubDetector({k: [(b, 0.9, "bottle") for b, _ in items]}).detect(IMG, k / 30, k)
        d = tracker.update(det, np.array([s for _, s in items]))
        out.append((d.target_track_id, d.target_idx))
    return out


def test_iou():
    m = iou_matrix(np.array([BOX_A]), np.array([BOX_A, BOX_B]))
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == 0


def test_confirmation_after_n_frames(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    res = run(tr, [[(BOX_A, 0.8)]] * 5)
    ids = [r[0] for r in res]
    assert ids[:2] == [-1, -1]  # not confirmed yet
    assert ids[2] > 0 and len(set(ids[2:])) == 1  # confirmed at the 3rd frame
    assert res[3][1] == 0


def test_low_scores_never_confirm(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    res = run(tr, [[(BOX_A, 0.5), (BOX_B, 0.4)]] * 20)
    assert all(r[0] == -1 for r in res)


def test_isolated_spikes_dont_confirm(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    seq = [[(BOX_A, 0.8 if k % 2 == 0 else 0.4)] for k in range(20)]
    assert all(r[0] == -1 for r in run(tr, seq))


def test_loss_after_m_frames_and_reacquisition(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    seq = [[(BOX_A, 0.8)]] * 5 + [[]] * 9
    res = run(tr, seq)
    tid = res[4][0]
    assert tid > 0
    assert all(r[0] == tid for r in res[5:])  # 9 frames without detection: still confirmed (memory)
    assert all(r[1] == -1 for r in res[5:])  # but no detection index
    res2 = run(tr, [[]] * 2)
    assert res2[-1][0] == -1  # lost after 10 frames
    # reacquisition: new confirmation after N frames
    res3 = run(tr, [[(BOX_B, 0.85)]] * 4)
    assert [r[0] for r in res3][:2] == [-1, -1]
    assert res3[2][0] > 0


def test_target_moves_keeps_id(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    seq = [[(BOX_A + np.array([4 * k, 0, 4 * k, 0]), 0.8)] for k in range(15)]
    ids = [r[0] for r in run(tr, seq)]
    assert len(set(ids[2:])) == 1 and ids[-1] > 0


def test_picks_best_of_two(cfg):
    tr = TargetTracker(cfg, threshold=0.6)
    res = run(tr, [[(BOX_A, 0.65), (BOX_B, 0.9)]] * 5)
    t = tr.target_track
    assert t is not None and np.allclose(t.box, BOX_B)
    assert res[-1][1] == 1


def _bottle(gray=(150, 150, 150), orange=(0, 128, 255)):
    img = np.full((100, 40, 3), 104, np.uint8)
    img[:60] = gray
    img[60:] = orange
    return img


def test_color_score_and_hue_shift(cfg):
    cm = ColorModel.from_cfg(cfg.reid)
    b = _bottle()
    m = np.ones(b.shape[:2], bool)
    assert cm.proportion_score(b, m) > 0.9
    blue = hue_shift(b, 90)
    assert cm.proportion_score(blue, m) < 0.1


def test_reid_with_stub_embedder(cfg):
    reid = TargetReID(cfg, embedder=StubEmbedder(), backend="stub")
    b = _bottle()
    reid.build_gallery([(b, np.ones(b.shape[:2], bool))], n_aug=4)
    s_same, s_blue = reid.score_crops(
        [(b, np.ones(b.shape[:2], bool)), (hue_shift(b, 90), np.ones(b.shape[:2], bool))]
    )
    assert s_same > s_blue + 0.3


def test_masked_crop_bounds():
    img = np.zeros((50, 50, 3), np.uint8)
    c, m = masked_crop(img, np.array([-10, -10, 20, 20]), None)
    assert c.shape[:2] == m.shape and c.shape[0] > 0


def test_confirmed_track_does_not_jump_to_neighbour(cfg):
    """Missed target + neighbour with incompatible appearance: the track does not jump onto it."""
    tr = TargetTracker(cfg, threshold=0.6)
    near = BOX_A + np.array([25, 0, 25, 0])
    res = run(tr, [[(BOX_A, 0.8)]] * 4 + [[(near, 0.2)]] * 5)
    tid = res[3][0]
    assert tid > 0
    assert all(r[0] == tid and r[1] == -1 for r in res[4:])
    assert tr.target_track is not None and np.allclose(tr.target_track.box[:2], BOX_A[:2], atol=5)
