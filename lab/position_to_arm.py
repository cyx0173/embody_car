from __future__ import annotations

import math
from collections.abc import Sequence


# 单位: meter / radian
BASE_HEIGHT_M = 0.095
UPPER_ARM_M = 0.11257
FOREARM_M = 0.13490
TOOL_OFFSET_M = 0.0

TICKS_PER_REV = 4096
SERVO_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1788, "range_min": 715, "range_max": 3466},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1706, "range_min": 822, "range_max": 3226},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 1712, "range_min": 908, "range_max": 3123},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 1345, "range_min": 845, "range_max": 3176},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1900, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1313, "range_min": 1507, "range_max": 3026},
}
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {}


def position_to_arm(position_xyz: Sequence[float]) -> dict[str, float]:
    """
    Convert a 3D target position in the arm base frame to joint angles.

    Args:
        position_xyz: (x, y, z), unit is meter.

    Returns:
        Joint angles in radian:
        shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.

    Raises:
        ValueError: if the target is outside the simplified arm workspace.
    """
    if len(position_xyz) != 3:
        raise ValueError("position_xyz must be a 3D coordinate: (x, y, z)")

    x, y, z = [float(v) for v in position_xyz]
    shoulder_pan = math.atan2(y, x)

    radial = math.hypot(x, y) - TOOL_OFFSET_M
    height = z - BASE_HEIGHT_M
    if radial < 0:
        radial = 0.0

    distance = math.hypot(radial, height)
    max_reach = UPPER_ARM_M + FOREARM_M
    min_reach = abs(UPPER_ARM_M - FOREARM_M)
    if distance > max_reach or distance < min_reach:
        raise ValueError(
            f"target out of workspace: distance={distance:.4f}m, "
            f"valid=[{min_reach:.4f}, {max_reach:.4f}]m"
        )

    cos_elbow = (
        distance * distance - UPPER_ARM_M * UPPER_ARM_M - FOREARM_M * FOREARM_M
    ) / (2.0 * UPPER_ARM_M * FOREARM_M)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))

    elbow_flex = math.acos(cos_elbow)
    shoulder_lift = math.atan2(height, radial) - math.atan2(
        FOREARM_M * math.sin(elbow_flex),
        UPPER_ARM_M + FOREARM_M * math.cos(elbow_flex),
    )
    wrist_flex = -(shoulder_lift + elbow_flex)
    wrist_roll = 0.0

    joints = {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }

    for name, angle in joints.items():
        lo, hi = JOINT_LIMITS_RAD[name]
        if angle < lo or angle > hi:
            raise ValueError(f"{name} angle out of limit: {angle:.4f} rad, valid=[{lo}, {hi}]")

    return joints


def _calibration_range_rad(config: dict[str, int]) -> tuple[float, float]:
    lo = _raw_to_rad(config["range_min"], config)
    hi = _raw_to_rad(config["range_max"], config)
    return min(lo, hi), max(lo, hi)


def _raw_to_rad(raw: int, config: dict[str, int]) -> float:
    ticks = int(raw) - _resolve_zero_raw(config)
    if config["drive_mode"]:
        ticks = -ticks
    return ticks / TICKS_PER_REV * 2.0 * math.pi


def _resolve_zero_raw(config: dict[str, int]) -> int:
    lower = int(config["range_min"])
    upper = int(config["range_max"])
    offset = int(config["homing_offset"])
    candidates = (offset, offset % TICKS_PER_REV, 2048 + offset, 2048 - offset)
    for candidate in candidates:
        if lower <= int(candidate) <= upper:
            return int(candidate)
    return int((lower + upper) / 2)


JOINT_LIMITS_RAD = {
    name: _calibration_range_rad(config)
    for name, config in SERVO_CALIBRATION.items()
    if name != "gripper"
}
