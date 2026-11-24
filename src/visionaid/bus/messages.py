"""Messages exchanged between blocks (dataclasses) and their msgpack (de)serialisation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np


@dataclass
class Frame:
    """Timestamped BGR uint8 image with its intrinsics."""

    t: float
    frame_idx: int
    image: np.ndarray
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_phone: float | None = None  # capture time, phone clock (s)
    t_recv: float | None = None  # reception time, computer clock (s, time.time())


@dataclass
class Pose:
    """Camera pose T_world_cam (4×4 float64) and tracking quality."""

    t: float
    frame_idx: int
    T_world_cam: np.ndarray
    quality: str = "ok"  # ok | init | degraded (IMU-only orientation) | lost
    n_inliers: int = 0
    reproj_err: float = 0.0
    is_keyframe: bool = False
    up: np.ndarray | None = None  # "up" direction (gravity) in the world frame of the pose


@dataclass
class Depth:
    """Metric depth (m) at image resolution."""

    t: float
    frame_idx: int
    depth: np.ndarray
    confidence: np.ndarray | None = None


@dataclass
class Detections:
    """Detections of a frame, with tracks and the "target" decision."""

    t: float
    frame_idx: int
    boxes: np.ndarray  # (N,4) xyxy in pixels
    scores: np.ndarray  # (N,)
    labels: list[str] = field(default_factory=list)
    masks: np.ndarray | None = None  # (N,H,W) bool
    track_ids: np.ndarray | None = None  # (N,) int
    reid_scores: np.ndarray | None = None  # (N,)
    target_idx: int = -1  # index of the detection confirmed as the target, -1 otherwise
    target_track_id: int = -1
    candidate_idx: int = -1  # "probable" (unconfirmed) target, -1 otherwise
    target_id: int = 0  # number of the request (voice command) for which the target is searched


@dataclass
class ObjectEstimate:
    """Estimated world position of the target and its memory state."""

    t: float
    frame_idx: int
    position: np.ndarray  # (3,)
    cov: np.ndarray  # (3,3)
    state: str = "unknown"  # unknown | seen | remembered | stale
    confidence: float = 0.0
    target_id: int = 0  # number of the request (voice command) concerned


@dataclass
class Costmap:
    """Top-down 2D grid: state (−1 unknown, 0 free, 1 occupied) and cost."""

    t: float
    frame_idx: int
    state: np.ndarray  # (H,W) int8
    cost: np.ndarray  # (H,W) float32, inf = interdit
    origin: np.ndarray  # (2,) world (x,y) corner of cell (0,0)
    resolution: float = 0.05


@dataclass
class Path:
    """Planned path in world coordinates (x,y)."""

    t: float
    frame_idx: int
    points: np.ndarray  # (N,2)
    lookahead: np.ndarray  # (2,)
    goal: np.ndarray  # (2,)
    valid: bool = False
    length: float = 0.0
    target_id: int = 0  # number of the request (voice command) concerned


@dataclass
class GuidanceCue:
    """Guidance instruction relative to the head."""

    t: float
    frame_idx: int
    state: str
    azimuth: float = 0.0  # rad, positive to the left
    elevation: float = 0.0  # rad, positive upward
    distance: float = float("nan")
    alert_level: float = 0.0
    message: str = ""
    # direction of the path to follow (rad, left +), even during an obstacle alert; nan if none
    guide_azimuth: float = float("nan")
    # obstacle to signal in the frontal corridor (m); inf if none (or near the target)
    obstacle_distance: float = float("inf")
    # sentence to speak (speech synthesis): repeated on the instructions for ~1 s, same `speech_id`
    speech: str = ""
    speech_id: int = 0
    speech_duck: bool = False  # path layer suspended during the sentence


@dataclass
class SoundCommand:
    """What the speaker must play, derived from an instruction (`audio.director`).

    Computed once on the computer side, then rendered identically by the phone (Web Audio) and
    by the offline rendering (WAV). Two overlaid layers: the path (repeated note, stereo
    pan) and the obstacles (looping pad, volume according to distance).
    """

    t: float
    frame_idx: int
    state: str
    mode: str  # silent | search | guide | lost | arrived (path layer)
    azimuth: float = 0.0  # rad, positive to the left (path direction)
    pan: float = 0.0  # −1 = fully left, +1 = fully right (constant power)
    distance: float = float("nan")
    interval_s: float = 1.0  # spacing between two path notes
    gain_db: float = 0.0  # gain of the path note
    obstacle_db: float = -120.0  # gain of the obstacle pad (≤ −120: muted)
    event: str = ""  # "arrived": arrival sound to play once
    speech: str = ""  # sentence to speak (once per `speech_id`)
    speech_id: int = 0
    speech_duck: bool = False  # path suspended during the sentence (object found announcement)


@dataclass
class ImuSample:
    """Inertial sample, device frame (W3C: x right, y top of the screen, z towards the user)."""

    t: float  # phone clock (s)
    gyro: np.ndarray  # (3,) rad/s
    accel: np.ndarray  # (3,) m/s², gravity included
    accel_lin: np.ndarray | None = None  # (3,) m/s², without gravity (if provided)


@dataclass
class ImuBatch:
    """Batch of inertial samples as arrays (more compact on the bus)."""

    t: np.ndarray  # (N,) phone clock (s)
    gyro: np.ndarray  # (N,3) rad/s
    accel: np.ndarray  # (N,3) m/s²
    t_recv: float | None = None  # reception on the computer side

    @staticmethod
    def from_samples(samples: list[ImuSample], t_recv: float | None = None) -> ImuBatch:
        """Assembles samples into a batch."""
        if not samples:
            return ImuBatch(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), t_recv)
        return ImuBatch(
            np.array([s.t for s in samples], float),
            np.array([s.gyro for s in samples], float).reshape(-1, 3),
            np.array([s.accel for s in samples], float).reshape(-1, 3),
            t_recv,
        )

    def samples(self) -> list[ImuSample]:
        """Splits the batch into samples."""
        return [
            ImuSample(float(t), g, a) for t, g, a in zip(self.t, self.gyro, self.accel, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class ClockSync:
    """Clock offset: t_pc ≈ t_phone + offset (s), with uncertainty (minimal half-RTT)."""

    t: float
    offset: float
    uncertainty: float
    rtt: float = 0.0


@dataclass
class VoiceCommand:
    """User request (voice or fixed prompt): look for an object, or stop.

    `id` increases at each new search or stop: the blocks start again from zero when it changes.
    A misunderstood phrase keeps the current `id` (nothing changes) but has a new `seq`.
    """

    t: float  # time.time() of the request
    id: int
    action: str  # search | stop | unknown
    text: str = ""  # recognised phrase
    query: str = ""  # phrase passed to the target card
    target: str = ""  # name of the card (config/targets/<name>.yaml)
    spoken: str = ""  # name to pronounce ("la gourde")
    plural: bool = False
    seq: int = 0  # number of the received phrase (every phrase, even misunderstood)


@dataclass
class Landmarks:
    """Furniture spotted (aligned world frame, z up) to describe where the object is."""

    t: float
    labels: list[str]  # English detector prompt ("table", "sofa"…)
    centers: np.ndarray  # (N,2) x, y
    boxes: np.ndarray  # (N,4) xmin, ymin, xmax, ymax (footprint on the floor)
    top_z: np.ndarray  # (N,) height of the top (m)
    n_obs: np.ndarray  # (N,) number of observations


MESSAGE_TYPES: dict[str, type] = {
    c.__name__: c
    for c in (
        Frame,
        Pose,
        Depth,
        Detections,
        ObjectEstimate,
        Costmap,
        Path,
        GuidanceCue,
        SoundCommand,
        ImuSample,
        ImuBatch,
        ClockSync,
        VoiceCommand,
        Landmarks,
    )
}


def _default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    return msgpack_numpy.encode(obj)


def encode(msg: Any) -> bytes:
    """Serialises a message (dataclass) to msgpack."""
    payload = {f.name: getattr(msg, f.name) for f in dataclasses.fields(msg)}
    return msgpack.packb(
        {"_type": type(msg).__name__, "d": payload}, default=_default, use_bin_type=True
    )


def decode(data: bytes) -> Any:
    """Deserialises a msgpack message into a dataclass."""
    obj = msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)
    cls = MESSAGE_TYPES[obj["_type"]]
    return cls(**obj["d"])
