"""Calibrates the re-identification threshold (max F1): positives = augmentations of the references,
negatives = crops of other objects detected in the video."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from visionaid.perception.detector import YoloDetector
from visionaid.perception.reid import (
    TargetReID,
    TextOnlyReID,
    augment,
    masked_crop,
    reference_crops,
)
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.logging import get_logger

log = get_logger("calibrate")


def build_reid(cfg, detector=None) -> tuple[TargetReID, list, YoloDetector]:
    """Detector + ReID with a gallery built on the reference photos.

    Without a reference photo for the target: "text only" ReID (detector confidence).
    """
    det = detector or YoloDetector(cfg)
    if bool(getattr(cfg.reid, "text_only", False)):
        # demo mode: a single object of this category in the room; any detection of the
        # text prompt is the target (no reference photos and no colour rule)
        return TextOnlyReID(cfg), [], det
    refs = sorted(resolve_path(cfg.paths.ref_dir).glob(cfg.paths.ref_glob))
    if not refs:
        log.warning(
            "No reference photo (%s/%s): target recognised by text alone",
            cfg.paths.ref_dir,
            cfg.paths.ref_glob,
        )
        return TextOnlyReID(cfg), [], det
    rc = reference_crops(cfg, det, refs)
    if not rc:
        raise RuntimeError("No usable reference")
    reid = TargetReID(cfg)
    reid.build_gallery([(c, m) for c, m, _ in rc])
    return reid, rc, det


def positive_items(ref_crops, n_aug: int, seed: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Augmentations (with occlusion) of the reference crops; mask = non-background pixels."""
    from visionaid.perception.reid import BG_COLOR

    rng = np.random.default_rng(seed)
    out = []
    for c, _m, *_ in ref_crops:
        for _ in range(n_aug):
            a = augment(c, rng, occlude=True)
            mask = np.abs(a.astype(int) - np.array(BG_COLOR)).sum(2) > 12
            if mask.sum() < 16:
                mask[:] = True
            out.append((a, mask))
    return out


def collect_negatives(cfg, video: Path, step: int = 5, cache: bool = True) -> list[dict]:
    """(Masked) crops of objects detected with the broad prompt, every `step` frames."""
    from visionaid.io.session import open_media

    tag = "_".join(cfg.detector.negative_prompts)[:40].replace(" ", "-")
    cpath = resolve_path(cfg.paths.cache) / "reid" / f"negatives_{video.stem}_{step}_{tag}.pkl"
    legacy = resolve_path(cfg.paths.cache) / "reid" / f"negatives_{video.stem}_{step}.pkl"
    if cache and not cpath.exists() and legacy.exists() and str(cfg.target.name) == "gourde":
        cpath = legacy  # phase 3 cache (bottle prompts)
    if cache and cpath.exists():
        return pickle.loads(cpath.read_bytes())
    det = YoloDetector(cfg, prompts=list(cfg.detector.negative_prompts))
    src = open_media(cfg, video)  # video or phone session
    items = []
    while (f := src.read()) is not None:
        if f.frame_idx % step:
            continue
        d = det.detect(f.image, f.t, f.frame_idx)
        for i, b in enumerate(d.boxes):
            if (b[2] - b[0]) < 12 or (b[3] - b[1]) < 12:
                continue
            crop, m = masked_crop(f.image, b, None if d.masks is None else d.masks[i])
            items.append(
                {"frame_idx": f.frame_idx, "label": d.labels[i], "box": b, "crop": crop, "mask": m}
            )
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_bytes(pickle.dumps(items))
    return items


def split_negatives(
    reid: TargetReID, items: list[dict], exclude_sim: float = 0.60, exclude_sim_color: float = 0.45
) -> tuple[list[dict], list[dict]]:
    """Excludes from the negatives the crops too similar to the reference (the bottle is in the video).

    Rule: embedding similarity ≥ `exclude_sim`, or ≥ `exclude_sim_color` with a colour
    proportion score ≥ 0.5 (grey + orange).
    """
    if not items:
        return [], []
    emb = reid.embed_scores([it["crop"] for it in items])
    keep, excl = [], []
    for it, e in zip(items, emb, strict=True):
        prop = (
            reid.colors.proportion_score(it["crop"], it["mask"]) if reid.colors is not None else 0.0
        )
        it["emb"] = float(e)
        it["prop"] = float(prop)
        (excl if (e >= exclude_sim or (e >= exclude_sim_color and prop >= 0.5)) else keep).append(
            it
        )
    return keep, excl


def best_f1_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, dict]:
    """Threshold maximising F1 (on a tie, the highest one: fewer false positives)."""
    cands = np.unique(np.concatenate([pos, neg]))
    best = (-1.0, 0.5, {})
    for th in cands:
        tp = float((pos >= th).sum())
        fp = float((neg >= th).sum())
        fn = float((pos < th).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 >= best[0]:
            best = (f1, float(th), {"tp": tp, "fp": fp, "fn": fn})
    f1, th, c = best
    # puts the threshold halfway between the kept positive and the negative just below
    below = neg[neg < th]
    if below.size:
        th = 0.5 * (th + float(below.max()))
    return th, {
        "f1": f1,
        "recall": c["tp"] / max(len(pos), 1),
        "fpr": float((neg >= th).sum()) / max(len(neg), 1),
        **c,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--target", default=None, help="fixed prompt, e.g. « où sont mes clés »")
    ap.add_argument(
        "--negatives",
        default=None,
        help="video or session to draw the negatives from (default: video)",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"target.query": args.target} if args.target else None)
    reid, rc, _det = build_reid(cfg)
    if getattr(reid, "text_only", False):
        raise SystemExit(
            f"No reference photo ({cfg.paths.ref_glob}): nothing to calibrate for "
            f"\"{cfg.target.name}\" (text-only mode, threshold reid.textonly_conf)."
        )
    pos_items = positive_items(rc, int(cfg.reid.n_calib_aug))
    pos = reid.score_crops(pos_items)
    video = resolve_path(args.negatives or cfg.paths.video)
    negs_all = collect_negatives(cfg, video, args.step) if video.exists() else []
    negs, excl = split_negatives(reid, negs_all)
    neg = reid.score_crops([(it["crop"], it["mask"]) for it in negs]) if negs else np.zeros(0)
    th, stats = best_f1_threshold(pos, neg)
    out = {
        "threshold": round(float(th), 4),
        "backend": reid.backend,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_excluded": len(excl),
        "pos_scores": {
            "min": float(pos.min()),
            "p10": float(np.percentile(pos, 10)),
            "median": float(np.median(pos)),
        },
        "neg_scores": {
            "max": float(neg.max()) if len(neg) else None,
            "p99": float(np.percentile(neg, 99)) if len(neg) else None,
        },
        **{k: float(v) for k, v in stats.items()},
    }
    path = resolve_path(cfg.reid.threshold_file)
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    log.info("ReID threshold: %s", json.dumps(out))
    return out


if __name__ == "__main__":
    main()
