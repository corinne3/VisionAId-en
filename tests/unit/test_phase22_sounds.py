"""Guidance sounds: VisionAid_Sounds sounds, sound director (pan, tempo, obstacles), rendering."""

import math

import numpy as np
import pytest

from visionaid.audio.director import (
    GUIDE,
    LOST,
    MUTE_DB,
    SEARCH,
    PanTracker,
    SoundDirector,
    equal_power,
)
from visionaid.audio.renderer import AudioRenderer
from visionaid.bus.messages import GuidanceCue, SoundCommand, decode, encode
from visionaid.utils.config import load_config, resolve_path

pytestmark = pytest.mark.unit
DEG = math.pi / 180


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _cue(t, state, az=0.0, dist=3.0, obst=math.inf):
    return GuidanceCue(t, int(t * 10), state, az, 0.0, dist, 0.0, "", az, obst)


# ------------------------------------------------------------------------------ pan
def test_equal_power_full_left_is_left_ear_only():
    gl, gr = equal_power(-1.0)
    assert gl == pytest.approx(1.0) and gr == pytest.approx(0.0, abs=1e-12)
    gl, gr = equal_power(1.0)
    assert gl == pytest.approx(0.0, abs=1e-12) and gr == pytest.approx(1.0)
    gl, gr = equal_power(np.linspace(-1, 1, 21))
    assert np.allclose(gl**2 + gr**2, 1.0)  # puissance constante


def test_pan_linear_in_front_no_exaggeration():
    p = PanTracker()
    assert p(0.0) == pytest.approx(0.0)
    assert p(45 * DEG) == pytest.approx(-0.5)  # left positive → pan to the left (−)
    assert p(-45 * DEG) == pytest.approx(0.5)
    assert p(10 * DEG) == pytest.approx(-10 / 90)
    assert p(90 * DEG) == pytest.approx(-1.0)


def test_pan_behind_full_side_with_hysteresis():
    p = PanTracker(150)
    assert p(170 * DEG) == -1.0  # behind, rather on the left: fully left
    assert p(-175 * DEG) == -1.0  # barely crosses to the other side: we keep the left
    assert p(-155 * DEG) == -1.0
    assert p(-140 * DEG) == 1.0  # clearly on the right: we switch
    assert p(-100 * DEG) == 1.0
    assert p(-60 * DEG) == pytest.approx(60 / 90)  # in front again: proportional pan
    q = PanTracker(150)
    assert q(-170 * DEG) == 1.0  # entering behind on the right


# ------------------------------------------------------------------------------ directeur
def test_director_modes(cfg):
    d = SoundDirector(cfg.audio)
    assert d.command(_cue(0, "SEARCHING", 0.3)).mode == SEARCH
    assert d.command(_cue(0.1, "NAVIGATING", 0.3)).mode == GUIDE
    c = d.command(_cue(0.2, "OBSTACLE_ALERT", 0.3, obst=0.5))
    assert c.mode == GUIDE and c.obstacle_db > MUTE_DB  # the path continues under the obstacle
    lost = d.command(_cue(0.3, "TRACKING_LOST"))
    assert lost.mode == LOST and lost.obstacle_db == pytest.approx(cfg.audio.lost_obstacle_db)


def test_tempo_accelerates_near_target(cfg):
    a = cfg.audio
    d = SoundDirector(a)
    assert d.interval(10.0) == pytest.approx(a.guide_interval_far_s)
    assert d.interval(a.guide_accel_m) == pytest.approx(a.guide_interval_far_s)
    assert d.interval(a.guide_near_m) == pytest.approx(a.guide_interval_near_s)
    xs = [d.interval(x) for x in np.linspace(a.guide_accel_m, a.guide_near_m, 10)]
    assert np.all(np.diff(xs) < 0)


def test_obstacle_gain_rises_but_stays_capped(cfg):
    a = cfg.audio
    d = SoundDirector(a)
    assert d.obstacle_db(math.inf) == MUTE_DB
    assert d.obstacle_db(a.obstacle_far_m + 0.1) == MUTE_DB
    g = [d.obstacle_db(x) for x in np.linspace(a.obstacle_far_m - 0.01, a.obstacle_near_m, 10)]
    assert np.all(np.diff(g) > 0)
    assert d.obstacle_db(0.1) == pytest.approx(a.obstacle_max_db)
    assert a.obstacle_max_db < 0  # always below the path


def test_obstacle_layer_with_path_direction(cfg):
    d = SoundDirector(cfg.audio)
    c = d.command(_cue(0.0, "NAVIGATING", 30 * DEG, obst=0.8))
    assert c.pan == pytest.approx(-30 / 90) and c.obstacle_db > MUTE_DB


def test_arrival_event_once_with_cooldown(cfg):
    d = SoundDirector(cfg.audio)
    seq = ["APPROACHING", "ARRIVED", "ARRIVED", "APPROACHING", "ARRIVED"]
    ev = [d.command(_cue(0.5 * i, s, 0.0, 0.4)).event for i, s in enumerate(seq)]
    assert ev == ["", "arrived", "", "", ""]  # bounce < 5 s: no second sound
    assert d.command(_cue(20.0, "APPROACHING")).event == ""
    assert d.command(_cue(20.5, "ARRIVED")).event == "arrived"


def test_sound_command_on_bus():
    c = SoundCommand(1.0, 3, "NAVIGATING", GUIDE, 0.2, -0.3, 2.0, 0.9, 0.0, -12.0, "")
    assert decode(encode(c)) == c


# ------------------------------------------------------------------------------ rendu
def _onsets(mono, fs, thresh=0.05):
    env = np.abs(mono)
    on = np.nonzero(env > thresh * env.max())[0]
    starts = [on[0]] + [b for a, b in zip(on[:-1], on[1:], strict=False) if b - a > 0.15 * fs]
    return np.array(starts) / fs


def test_render_full_left_silent_on_right(cfg):
    r = AudioRenderer(cfg.audio)
    cues = [_cue(0.1 * k, "NAVIGATING", 120 * DEG, 6.0) for k in range(40)]
    st = r.render(cues, 0.0, 4.0)
    late = st[int(0.5 * r.fs) :]
    assert np.abs(late[:, 0]).max() > 0.05
    assert np.abs(late[:, 1]).max() < 1e-3  # fully left: nothing on the right
    gaps = np.diff(_onsets(st[:, 0], r.fs))
    assert np.median(gaps) == pytest.approx(cfg.audio.guide_interval_far_s, abs=0.03)


def test_render_center_balanced(cfg):
    r = AudioRenderer(cfg.audio)
    st = r.render([_cue(0.1 * k, "NAVIGATING", 0.0) for k in range(30)], 0.0, 3.0)
    rms = np.sqrt(np.mean(st.astype(float) ** 2, axis=0))
    assert abs(20 * np.log10(rms[0] / rms[1])) < 0.1


def test_render_obstacle_superposed_and_louder_when_closer(cfg):
    r = AudioRenderer(cfg.audio)

    def obstacle_rms(d):
        clean = r.render([_cue(0.1 * k, "NAVIGATING", 0.0) for k in range(40)], 0.0, 4.0)
        both = r.render([_cue(0.1 * k, "NAVIGATING", 0.0, obst=d) for k in range(40)], 0.0, 4.0)
        diff = (both - clean)[int(1.5 * r.fs) :]
        return float(np.sqrt(np.mean(diff.astype(float) ** 2)))

    far, near = obstacle_rms(1.3), obstacle_rms(0.4)
    assert obstacle_rms(math.inf) == 0
    assert 0 < far < near
    # obstacle alert: the path continues exactly as in navigation
    alert = r.render([_cue(0.1 * k, "OBSTACLE_ALERT", 0.0, obst=0.4) for k in range(40)], 0.0, 4.0)
    nav = r.render([_cue(0.1 * k, "NAVIGATING", 0.0, obst=0.4) for k in range(40)], 0.0, 4.0)
    assert np.array_equal(alert, nav)


def test_render_arrival_and_silence_after(cfg):
    r = AudioRenderer(cfg.audio)
    arr = r.render([_cue(0.0, "APPROACHING", 0.0, 0.6), _cue(0.5, "ARRIVED", 0.0, 0.4)], 0.0, 4.0)
    assert np.abs(arr[int(0.6 * r.fs) : int(1.0 * r.fs)]).max() > 0.01  # arrival sound
    assert np.abs(arr[int(2.0 * r.fs) :]).max() < 1e-3  # then silence


def test_render_silent_without_cues(cfg):
    r = AudioRenderer(cfg.audio)
    assert np.abs(r.render([], 0.0, 1.0)).max() == 0


def test_render_search_uses_search_sound(cfg):
    r = AudioRenderer(cfg.audio)
    assert len(r.search) != len(r.guide) or not np.allclose(r.search, r.guide)
    st = r.render([_cue(t / 10, "SEARCHING") for t in range(40)], 0.0, 4.0)
    # centred: each ear gets the search sound × 0.707, one every search_interval_s
    exp = np.zeros(len(st))
    k = 0.0
    while k < 4.0:
        AudioRenderer._add(exp, int(round(k * r.fs)), r.search)
        k += cfg.audio.search_interval_s
    assert np.allclose(st[:, 0], exp * math.sqrt(0.5), atol=1e-4)


# ------------------------------------------------------------------------------ fichiers
def test_processed_sounds_present_and_normalized(cfg):
    import json

    d = resolve_path(cfg.audio.sounds_dir)
    rep = json.loads((d / "levels.json").read_text())
    for s in ("guide", "search", "arrived", "obstacle"):
        f = getattr(cfg.audio.sounds, s).file
        assert (d / f).exists()
        assert rep["sounds"][f]["after"]["lufs_m"] <= -20.0 + 0.5
        assert rep["sounds"][f]["after"]["peak_dbfs"] <= -1.0 + 1e-6
        assert rep["sounds"][f]["src"].startswith("VisionAid_Sounds")


def test_obstacle_loop_is_seamless(cfg):
    import soundfile as sf

    x, fs = sf.read(str(resolve_path(cfg.audio.sounds_dir) / cfg.audio.sounds.obstacle.file))
    x = np.atleast_2d(x.T).T
    jump = np.abs(x[0] - x[-1]).max()
    typical = np.abs(np.diff(x, axis=0)).max()
    assert jump <= 1.5 * typical  # loop without a click at the junction


def test_prepare_trims_and_normalizes():
    from visionaid.audio.prepare import make_loop, momentary_loudness, prepare_one

    fs = 44100
    t = np.arange(int(0.5 * fs)) / fs
    tone = 0.01 * np.sin(2 * np.pi * 220 * t)
    x = np.concatenate([np.zeros(int(0.3 * fs)), tone, np.zeros(int(0.3 * fs))])
    y, info = prepare_one(x, fs, True, -20.0, -1.0)
    assert len(y) < 0.6 * fs  # silences trimmed
    assert momentary_loudness(y, fs) == pytest.approx(-20.0, abs=0.3)
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3  # fondus
    lp = make_loop(np.sin(2 * np.pi * 3.3 * np.arange(fs) / fs), fs)
    assert abs(lp[0] - lp[-1]) < 0.01
