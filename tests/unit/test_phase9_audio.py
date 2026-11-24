import math

import numpy as np
import pytest

from visionaid.audio.binaural import spatialize, woodworth_itd
from visionaid.audio.renderer import AudioRenderer
from visionaid.bus.messages import GuidanceCue
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit
FS = 44100


def noise(seconds=0.5, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, int(seconds * FS))


def xcorr_lag(st, max_lag=60):
    """Delay (samples) of the right channel relative to the left (>0: the left leads)."""
    left, r = st[2000:, 0], st[2000:, 1]
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [np.dot(left[max_lag:-max_lag], np.roll(r, -k)[max_lag:-max_lag]) for k in lags]
    return int(lags[int(np.argmax(cc))])


def db(x):
    return 10 * np.log10(np.mean(x**2) + 1e-20)


@pytest.mark.parametrize("side", [1, -1])
def test_itd_ild_lateral(side):
    az = side * math.radians(60)
    st = spatialize(noise(), az, FS)
    lag = xcorr_lag(st)
    itd_theory = woodworth_itd(az) * FS  # >0: left leads
    assert abs(lag - itd_theory) <= 0.1e-3 * FS + 1
    ild = db(st[:, 0]) - db(st[:, 1])
    assert side * ild > 3.0
    assert np.sign(lag) == side


def test_front_center_identical():
    st = spatialize(noise(), 0.0, FS)
    assert abs(xcorr_lag(st)) == 0
    assert abs(db(st[:, 0]) - db(st[:, 1])) < 0.1
    assert np.max(np.abs(st[:, 0] - st[:, 1])) < 1e-9


def hf_energy(x, fc=4000):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return X[f > fc].sum() / X.sum()


def test_front_back_cue():
    m = noise(1.0)
    front = spatialize(m, math.radians(60), FS)
    back = spatialize(m, math.radians(120), FS)  # same lateral angle, behind
    assert hf_energy(back.sum(1)) < 0.8 * hf_energy(front.sum(1))


def _cues(states_az, dt=0.1, dist=3.0):
    return [GuidanceCue(k * dt, k, s, az, 0.0, dist, 0.0) for k, (s, az) in enumerate(states_az)]


def test_no_clipping():
    r = AudioRenderer(load_config().audio)
    seq = (
        [("NAVIGATING", 1.0)] * 10
        + [("OBSTACLE_ALERT", -0.5)] * 10
        + [("APPROACHING", 0.2)] * 10
        + [("ARRIVED", 0.0)] * 10
    )
    st = r.render(_cues(seq), 0.0, 4.0)
    assert np.max(np.abs(st)) < 1.0
    assert np.max(np.abs(st)) > 0.05


def test_no_click_on_azimuth_jump():
    cfg = load_config()
    r = AudioRenderer(cfg.audio)
    s0 = r.render(_cues([("NAVIGATING", 0.0)] * 30, dist=0.5), 0.0, 3.0)
    s90 = r.render(_cues([("NAVIGATING", math.radians(90))] * 30, dist=0.5), 0.0, 3.0)
    jump = r.render(
        _cues([("NAVIGATING", 0.0)] * 15 + [("NAVIGATING", math.radians(90))] * 15, dist=0.5),
        0.0,
        3.0,
    )
    # a click = sample jump larger than those of the two steady states
    d_steady = max(np.abs(np.diff(s0, axis=0)).max(), np.abs(np.diff(s90, axis=0)).max())
    d_jump = np.abs(np.diff(jump, axis=0)).max()
    assert d_jump <= 1.5 * d_steady


def test_guide_interval_shorter_when_closer():
    r = AudioRenderer(load_config().audio)
    assert r.interval(0.5) < r.interval(1.5) < r.interval(10.0)


def test_wav_duration(tmp_path):
    import soundfile as sf

    r = AudioRenderer(load_config().audio)
    fps, n = 30.0, 97
    cues = [GuidanceCue(k / fps, k, "NAVIGATING", 0.3, 0.0, 2.0) for k in range(n)]
    duration = n / fps
    st = r.render(cues, 0.0, duration)
    p = tmp_path / "g.wav"
    r.write_wav(p, st)
    info = sf.info(str(p))
    assert abs(info.duration - duration) <= 1 / fps
    assert info.channels == 2
