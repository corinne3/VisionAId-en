"""Phase 29: voice request (STT) and voice announcements (TTS)."""

import math

import numpy as np
import pytest

from visionaid.bus.messages import (
    GuidanceCue,
    Landmarks,
    ObjectEstimate,
    Path,
    VoiceCommand,
    decode,
    encode,
)
from visionaid.guidance.narration import (
    Narrator,
    SpeechQueue,
    between_landmark,
    pass_side,
    support_of,
    where_phrase,
)
from visionaid.perception.target import known_targets_phrase, parse_command
from visionaid.utils.config import load_config, load_raw

pytestmark = pytest.mark.unit


def T_facing(yaw_deg: float = 0.0, xy=(0.0, 0.0), h: float = 1.4) -> np.ndarray:
    """Camera (x right, y down, z forward) looking horizontally in the direction `yaw`."""
    y = math.radians(yaw_deg)
    fwd = np.array([math.cos(y), math.sin(y), 0.0])
    left = np.array([-math.sin(y), math.cos(y), 0.0])
    T = np.eye(4)
    T[:3, 0] = -left
    T[:3, 1] = [0, 0, -1]
    T[:3, 2] = fwd
    T[:3, 3] = [xy[0], xy[1], h]
    return T


def scene():
    """Bottle on a table at the far end, a sofa between the user and the table."""
    lms = Landmarks(
        t=0.0,
        labels=["table", "sofa"],
        centers=np.array([[4.0, -1.0], [2.1, -0.5]]),
        boxes=np.array([[3.6, -1.5, 4.4, -0.5], [1.8, -1.2, 2.4, 0.2]]),
        top_z=np.array([0.75, 0.45]),
        n_obs=np.array([5, 5]),
    )
    obj = ObjectEstimate(0.0, 0, np.array([4.0, -1.0, 0.85]), np.eye(3) * 0.01, "seen", 0.9)
    pts = np.array([[0.0, 0.0], [1.5, 0.6], [2.6, 0.6], [3.4, -0.4]])
    path = Path(0.0, 0, pts, pts[1], pts[-1], True, 4.0)
    return lms, obj, path


def test_parse_command():
    assert parse_command("Je cherche ma gourde")[1].name == "gourde"
    assert parse_command(["euh", "où sont mes clés"])[1].name == "cles"
    assert parse_command("arrête")[0] == "stop"
    assert parse_command("je cherche mon téléphone") == ("unknown", None)
    assert parse_command("") == ("unknown", None)
    assert "gourde" in known_targets_phrase() and "clés" in known_targets_phrase()


def test_messages_roundtrip():
    c = VoiceCommand(1.0, 2, "search", "je cherche ma gourde", "x", "gourde", "la gourde", seq=3)
    assert decode(encode(c)) == c
    lms, _, _ = scene()
    d = decode(encode(lms))
    assert d.labels == lms.labels and np.allclose(d.boxes, lms.boxes)


def test_geometry_helpers():
    lms, obj, path = scene()
    assert support_of(obj.position, lms, 0.15) == 0
    assert between_landmark(np.zeros(2), obj.position[:2], lms, (0,)) == 1
    assert pass_side(path.points, lms.centers[1]) == "gauche"
    assert where_phrase(math.radians(40), 4.0, 3.0) == "au fond à gauche"
    assert where_phrase(math.radians(-90), 2.0, 3.0) == "sur ta droite"
    assert where_phrase(math.radians(175), 2.0, 3.0) == "derrière toi"


def test_found_and_approach_phrases():
    cfg = load_config()
    lms, obj, path = scene()
    n = Narrator(cfg, "la gourde")
    s = n.found_phrase(T_facing(0), obj.position, path, None, lms)
    assert s.startswith("La gourde est au fond de la pièce, en face de toi, à environ 4 mètres")
    assert "derrière le canapé" in s and "sur la table" in s
    assert "contourne le canapé par la gauche" in s
    a = n.approach_phrase(T_facing(0, (3.0, 0.0)), obj.position, path, lms)
    assert a.startswith("Tu arrives vers la table, la gourde sera")
    # keys on the floor, without furniture: height announced, plural
    k = Narrator(cfg, "les clés", plural=True)
    pos = np.array([0.0, 2.0, 0.05])
    s = k.found_phrase(T_facing(0), pos, None, None, None)
    assert s.startswith("Les clés sont sur ta gauche, à environ 2 mètres, par terre.")


def test_narrator_sequence():
    cfg = load_config()
    lms, obj, path = scene()
    q = SpeechQueue(1.0)
    n = Narrator(cfg, "la gourde", speech=q)
    unknown = ObjectEstimate(0.0, 0, np.full(3, np.nan), np.eye(3), "unknown")
    n.update(0.0, "SEARCHING", T_facing(0), unknown, None, None, lms, math.inf, math.inf)
    text, sid, duck = q.current(0.0)
    assert text.startswith("Je cherche la gourde") and sid == 1 and duck
    # object seen, no path yet: we wait (at most wait_path_s)
    n.update(1.0, "NAVIGATING", T_facing(0), obj, None, None, lms, 4.1, math.inf)
    assert not n.found_said
    n.update(1.2, "NAVIGATING", T_facing(0), obj, path, None, lms, 4.1, math.inf)
    assert n.found_said
    assert q.current(1.2)[1] == 2 and "sur la table" in q.current(1.2)[0]
    assert q.current(2.5)[0] == ""  # held for one second then nothing
    # on the way: no announcement; near the table: approach announcement, without suspending the path
    n.update(3.0, "NAVIGATING", T_facing(0, (1.0, 0.5)), obj, path, None, lms, 3.2, math.inf)
    assert not n.approach_said
    n.update(5.0, "NAVIGATING", T_facing(-20, (2.9, -0.4)), obj, path, None, lms, 1.2, 0.6)
    text, sid, duck = q.current(5.0)
    assert n.approach_said and text.startswith("Tu arrives vers la table") and not duck
    # object confirmed elsewhere: correction
    moved = ObjectEstimate(6.0, 0, np.array([1.0, 3.0, 0.8]), np.eye(3), "seen")
    q2 = SpeechQueue()
    n2 = Narrator(cfg, "la gourde", speech=q2, announce_start=False)
    n2.update(0.0, "NAVIGATING", T_facing(0), obj, path, None, lms, 4.1, math.inf)
    n2.update(1.0, "NAVIGATING", T_facing(0), moved, None, None, lms, 3.0, math.inf)
    q2.current(0.0), q2.current(2.0)
    assert any(t.startswith("Correction.") for _, t in n2.history)


def test_director_carries_speech():
    from visionaid.audio.director import SoundDirector

    cfg = load_config()
    d = SoundDirector(cfg.audio)
    cue = GuidanceCue(0.0, 0, "NAVIGATING", 0.1, speech="Bonjour", speech_id=4, speech_duck=True)
    c = d.command(cue)
    assert c.speech == "Bonjour" and c.speech_id == 4 and c.speech_duck and c.mode == "guide"
    assert d.command(GuidanceCue(0.0, 0, "IDLE")).mode == "silent"


def test_renderer_ducks_path_during_found_phrase(tmp_path):
    from visionaid.audio import tts
    from visionaid.audio.renderer import AudioRenderer

    if not tts.available():
        pytest.skip("`say` (macOS) absent")
    cfg = load_config()
    r = AudioRenderer(cfg.audio, cfg.tts, tmp_path)
    cues = [
        GuidanceCue(
            t, 0, "NAVIGATING", 0.0, distance=4.0,
            speech="La gourde est devant toi." if t < 1 else "", speech_id=1 if t < 1 else 0,
            speech_duck=True, guide_azimuth=0.0,
        )
        for t in np.arange(0, 8, 0.05)
    ]  # fmt: skip
    st = r.render(cues, 0.0, 8.0)
    ((start, end, text),) = r.speeches
    assert start == 0.0 and 0.8 < end < 4 and text.startswith("La gourde")
    fs = r.fs
    assert np.abs(st[int(end * fs) + int(0.4 * fs) :]).max() > 0.01  # the path resumes
    # during the sentence: only the voice (centred, left = right)
    seg = st[int(0.2 * fs) : int((end - 0.1) * fs)]
    assert np.allclose(seg[:, 0], seg[:, 1], atol=1e-3)


def test_landmark_map_merges_views():
    from visionaid.mapping.landmarks import LandmarkMap

    cfg = load_config()
    m = LandmarkMap(cfg.landmarks)
    rng = np.random.default_rng(0)
    for k in range(4):
        # 0.8 × 1.2 m table, top at 0.75 m, partially seen each time
        x = rng.uniform(2.0, 2.8 if k % 2 else 2.5, 400)
        y = rng.uniform(-0.6, 0.6, 400)
        z = np.where(rng.random(400) < 0.7, 0.75, rng.uniform(0.1, 0.75, 400))
        m.observe("table" if k != 2 else "desk", 0.8, np.c_[x, y, z], float(k))
    m.observe(
        "chair",
        0.6,
        np.c_[rng.uniform(0, 0.4, 200), rng.uniform(2, 2.4, 200), rng.uniform(0.1, 0.9, 200)],
        5.0,
    )  # noqa: E501
    msg = m.to_msg(5.0)
    assert msg.labels == ["table"]  # the chair was seen only once
    b = msg.boxes[0]
    assert b[0] < 2.1 and b[2] > 2.5 and abs(msg.top_z[0] - 0.75) < 0.05


def test_perception_splits_furniture_and_switches_target():
    from visionaid.bus.messages import Frame
    from visionaid.perception.detector import StubDetector
    from visionaid.perception.pipeline import PerceptionBlock
    from visionaid.perception.reid import TextOnlyReID
    from visionaid.perception.target import target_raw
    from visionaid.utils.config import _to_dataclass

    raw = load_raw()
    cfg = _to_dataclass("config", raw)
    items = [
        (np.array([10, 10, 40, 60]), 0.9, "bottle"),
        (np.array([50, 50, 200, 150]), 0.8, "table"),
        (np.array([0, 100, 30, 150]), 0.2, "chair"),  # not confident enough: ignored
    ]
    det = StubDetector(fn=lambda img, k: items)
    blk = PerceptionBlock(
        cfg, detector=det, active=False, reid_factory=lambda c, d: TextOnlyReID(c)
    )
    f = Frame(0.0, 0, np.zeros((200, 240, 3), np.uint8))
    d = blk(f)
    assert len(d.boxes) == 0 and blk.last_furniture.labels == ["table"]
    blk.set_target(_to_dataclass("config", target_raw(raw, "je cherche mes clés")), 3)
    assert blk.cfg.target.name == "cles" and blk.target_id == 3
    out = blk.process({"frame": f})
    assert out["detections"].labels == ["bottle"] and out["detections"].target_id == 3
    assert out["furniture"].labels == ["table"] and out["furniture"].masks.dtype == np.uint8
    blk.clear_target(4)
    assert len(blk(f).boxes) == 0


def test_target_raw_resets_previous_sheet():
    from visionaid.perception.target import target_raw

    base = load_raw()
    raw = load_raw(overrides={"target.query": "mes clés"})
    raw["_raw_base"] = base
    g = target_raw(raw, "ma gourde")
    assert g["target"]["name"] == "gourde" and g["detector"]["conf"] == base["detector"]["conf"]
    assert g["target"]["spoken"] == "la gourde"


def test_command_desk_numbers_requests():
    from visionaid.phone.block import CommandDesk

    raw = load_raw(overrides={"command.stt": "phone"})
    heard = []
    desk = CommandDesk(raw, reply=heard.append)
    desk.on_message({"type": "command", "text": "je cherche ma gourde"})
    desk.on_message({"type": "command", "text": "bla bla"})
    desk.on_message({"type": "command", "text": "stop"})
    a, b, c = desk.pop(), desk.pop(), desk.pop()
    assert (a.action, a.id, a.target, a.spoken) == ("search", 1, "gourde", "la gourde")
    assert (b.action, b.id, b.seq) == ("unknown", 1, 2)  # misunderstood: the search continues
    assert (c.action, c.id) == ("stop", 2)
    assert desk.pop() is None and desk.current is c
    assert [h["action"] for h in heard] == ["search", "unknown", "stop"]


def test_live_guidance_block_idle_then_search():
    from visionaid.apps.live_blocks import LiveGuidanceBlock
    from visionaid.bus.messages import Pose

    raw = load_raw()
    raw["_command_mode"] = "stt"
    raw["_tts"] = "phone"
    blk = LiveGuidanceBlock(raw)
    pose = Pose(0.0, 0, T_facing(0))
    out = blk.process({"pose_world": pose})
    assert out["sound"].mode == "silent" and out["cue"].state == "IDLE"
    cmd = VoiceCommand(0.0, 1, "search", "ma gourde", "ma gourde", "gourde", "la gourde", seq=1)
    out = blk.process({"pose_world": Pose(0.1, 1, T_facing(0)), "command": cmd})
    assert out["sound"].mode == "search"
    assert out["sound"].speech.startswith("Je cherche la gourde")
    first_id = out["sound"].speech_id
    bad = VoiceCommand(0.0, 1, "unknown", "téléphone", seq=2)
    blk.process({"pose_world": Pose(1.5, 2, T_facing(0)), "command": bad})
    out = blk.process({"pose_world": Pose(1.6, 3, T_facing(0)), "command": bad})
    assert out["sound"].mode == "search"  # the search continues
    assert (
        out["sound"].speech.startswith("Je n'ai pas compris") and out["sound"].speech_id > first_id
    )
    stop = VoiceCommand(0.0, 2, "stop", "stop", seq=3)
    out = blk.process({"pose_world": Pose(3.0, 4, T_facing(0)), "command": stop})
    assert out["sound"].mode == "silent" and "arrête" in out["sound"].speech
