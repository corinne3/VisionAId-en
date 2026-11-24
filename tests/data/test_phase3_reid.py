import cv2
import numpy as np
import pytest

from tests_helpers import ref_paths_or_skip
from visionaid.apps.calibrate_reid import collect_negatives, positive_items, split_negatives
from visionaid.perception.pipeline import unpack_masks
from visionaid.perception.reid import hue_shift, masked_crop
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.slow, pytest.mark.data]


def test_reference_photos_detected_and_reidentified(reid_bundle):
    reid, rc, det = reid_bundle
    paths = ref_paths_or_skip()
    assert len(rc) == len(paths), "the bottle must be found on each photo"
    for p in paths:
        full = cv2.imread(str(p))
        small = cv2.resize(full, (1280, int(full.shape[0] * 1280 / full.shape[1])))
        d = det.detect(small)
        assert len(d.boxes) > 0
        s = reid.score_detections(small, d.boxes, d.masks)
        assert s.max() >= reid.threshold, f"{p.name}: best score {s.max():.3f} < threshold {reid.threshold:.3f}"


def test_augmentation_recall(reid_bundle):
    reid, rc, _ = reid_bundle
    items = positive_items(rc, 30, seed=123)  # seed different from the calibration
    s = reid.score_crops(items)
    recall = float((s >= reid.threshold).mean())
    update_report("reid", {"aug_recall": recall, "n_aug": len(items), "threshold": reid.threshold})
    assert recall >= 0.90


def test_negatives_false_positive_rate(reid_bundle, video_path):
    reid, _, _ = reid_bundle
    cfg = load_config()
    negs, excl = split_negatives(reid, collect_negatives(cfg, video_path, 5))
    s = reid.score_crops([(n["crop"], n["mask"]) for n in negs])
    fpr = float((s >= reid.threshold).mean())
    update_report("reid", {"neg_fpr": fpr, "n_neg": len(negs), "n_neg_excluded": len(excl), "neg_max": float(s.max())})
    assert fpr <= 0.05


def test_hue_shift_fails(reid_bundle):
    reid, rc, _ = reid_bundle
    for crop, mask, _info in rc:
        blue = hue_shift(crop, 90)  # orange (H≈15) → bleu (H≈105)
        s_orig, s_blue = reid.score_crops([(crop, mask), (blue, mask)])
        assert s_orig >= reid.threshold
        assert s_blue < reid.threshold, f"shifted hue score {s_blue:.3f} ≥ threshold"


def dominant_hue(crop, mask, s_min: int = 80):
    """Dominant hue (OpenCV H) of the saturated pixels, and fraction of saturated pixels."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[mask]
    sat = hsv[hsv[:, 1] >= s_min]
    if len(sat) == 0:
        return -1, 0.0
    hist = np.bincount(sat[:, 0] // 10, minlength=18)
    return int(hist.argmax() * 10 + 5), len(sat) / max(len(hsv), 1)


def test_video_target_confirmed(video_perception, cfg):
    dets, info = video_perception
    confirmed = [d for d in dets if d.target_idx >= 0]
    frac = len(confirmed) / len(dets)
    track_ids = {d.target_track_id for d in confirmed}
    update_report(
        "perception",
        {**info, "confirmed_fraction": frac, "n_target_tracks": len(track_ids), "first_confirmed_frame": confirmed[0].frame_idx if confirmed else None},
    )
    assert len(confirmed) >= 1


def test_video_confirmed_tracks_color_compatible(video_perception, cfg):
    """No confirmed track on an object whose dominant hue is incompatible."""
    from visionaid.io.sources import VideoFileSource

    dets, info = video_perception
    by_idx = {d.frame_idx: d for d in dets if d.target_idx >= 0}
    src = VideoFileSource.from_config(cfg)
    bad = []
    n = 0
    while (f := src.read()) is not None:
        d = by_idx.get(f.frame_idx)
        if d is None:
            continue
        masks = unpack_masks(d, f.image.shape[1])
        i = d.target_idx
        crop, m = masked_crop(f.image, d.boxes[i], None if masks is None else masks[i])
        n += 1
        hue, sat_frac = dominant_hue(crop, m)
        # incompatible dominant hue: clearly coloured object outside the bottle/warm metal hues
        if sat_frac > 0.25 and not (0 <= hue <= 35 or hue >= 170):
            bad.append((f.frame_idx, hue, sat_frac))
    update_report("perception", {"color_check_frames": n, "color_incompatible": len(bad)})
    assert len(bad) <= max(1, int(0.02 * n)), bad[:10]


def test_detector_throughput(reid_bundle, video_path):
    _, _, det = reid_bundle
    from visionaid.io.sources import VideoFileSource

    f = VideoFileSource.from_config(load_config()).read()
    fps = det.benchmark(f.image, 30)
    update_report("perception", {"detector_fps": fps, "detector_backend": det.backend})
    assert fps > 0


def test_threshold_file_exists():
    p = resolve_path(load_config().reid.threshold_file)
    assert p.exists(), "run `make calibrate`"
    _ = np.array(0)
