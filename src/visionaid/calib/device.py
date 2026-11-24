"""Calibrations specific to a phone (`phone.device`), loaded automatically if present.

- `config/calib/<device>_camera.yaml`: K and distortion (phase 17, checkerboard);
- `config/calib/<device>_imu.yaml`: R_cam_imu, gyroscope bias, time offset, units.
Otherwise: field of view from the config and prior values from `body`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from visionaid.io.intrinsics import Intrinsics, make_intrinsics
from visionaid.utils.config import resolve_path

CALIB_DIR = "config/calib"


def device_name(cfg) -> str:
    phone = getattr(cfg, "phone", None)
    return str(getattr(phone, "device", None) or "phone")


def calib_dir(cfg) -> Path:
    phone = getattr(cfg, "phone", None)
    return resolve_path(getattr(phone, "calib_dir", None) or CALIB_DIR)


def camera_calib_path(cfg) -> Path:
    return calib_dir(cfg) / f"{device_name(cfg)}_camera.yaml"


def imu_calib_path(cfg) -> Path:
    return calib_dir(cfg) / f"{device_name(cfg)}_imu.yaml"


def phone_intrinsics(cfg, width: int, height: int) -> Intrinsics:
    """Intrinsics for a phone image: device calibration if compatible.

    The calibration is scaled if the aspect ratio is the same, or rotated by 90°
    if the image is in the other orientation (portrait/landscape). Otherwise: field of view.
    """
    p = camera_calib_path(cfg)
    if getattr(cfg.intrinsics, "calibration_file", None) or not p.exists():
        return make_intrinsics(cfg.intrinsics, width, height)
    d = yaml.safe_load(p.read_text())
    K = np.asarray(d["K"], float)
    base = Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], int(d["width"]), int(d["height"]))
    ar = width / height
    if abs(base.width / base.height - ar) < 0.01 * ar:
        return base.scaled(width, height)
    rot = base.rotated90(clockwise=True)
    if abs(rot.width / rot.height - ar) < 0.01 * ar:
        return rot.scaled(width, height)
    return make_intrinsics(cfg.intrinsics, width, height)


@dataclass
class ImuCalib:
    """IMU ↔ camera parameters used by the fused pose."""

    R_cam_imu: np.ndarray
    gyro_bias: np.ndarray
    time_offset_s: float
    units_fix: float
    source: str  # "calibration" or "prior"
    R_cam_acc: np.ndarray | None = None  # None: same rotation as the gyroscope
    accel_sign: float = 1.0


def load_imu_calib(cfg) -> ImuCalib:
    """IMU calibration of the device, or prior values from `body`."""
    prior = np.asarray(cfg.body.R_cam_imu, float)
    p = imu_calib_path(cfg)
    if not p.exists():
        return ImuCalib(prior, np.zeros(3), 0.0, 1.0, "a priori")
    d = yaml.safe_load(p.read_text())
    units = float(d.get("units_fix", 1.0))
    if not math.isfinite(units) or units <= 0:
        units = 1.0
    return ImuCalib(
        np.asarray(d.get("R_cam_imu", prior), float),
        np.asarray(d.get("gyro_bias", [0, 0, 0]), float),
        float(d.get("time_offset_s", 0.0)),
        units,
        "calibration",
        np.asarray(d["R_cam_acc"], float) if "R_cam_acc" in d else None,
        float(d.get("accel_sign", 1.0)),
    )
