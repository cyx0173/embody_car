from __future__ import annotations

import math
from collections.abc import Sequence

from Angle_config import MY_CONFIG

# 单位: meter / radian
BASE_HEIGHT_M = 0.1
UPPER_ARM_M = 0.11257
FOREARM_M = 0.13490
# 腕关节到末端参考点的长度。没实测前保持 0，函数会退化成肩肘两连杆求解。
TOOL_OFFSET_M = 0.13
ANGLE_SEARCH_STEP_RAD = math.radians(1.0)
MIN_RADIAL_M = 0.10

JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {}


def position_to_arm(position_xyz: Sequence[float], tool_offset_m: float = TOOL_OFFSET_M) -> dict[str, float]:
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

    radial = math.hypot(x, y)
    if radial < MIN_RADIAL_M:
        raise ValueError(
            f"target too close to base vertical axis: radial={radial:.4f}m, "
            f"minimum={MIN_RADIAL_M:.4f}m"
        )

    height = z - BASE_HEIGHT_M

    distance = math.hypot(radial, height)
    max_reach = UPPER_ARM_M + FOREARM_M + tool_offset_m
    min_reach = _min_reach(UPPER_ARM_M, FOREARM_M, tool_offset_m)
    if distance > max_reach or distance < min_reach:
        raise ValueError(
            f"target out of workspace: distance={distance:.4f}m, "
            f"valid=[{min_reach:.4f}, {max_reach:.4f}]m"
        )

    if tool_offset_m <= 1e-9:
        shoulder_lift, elbow_flex, wrist_flex = _solve_two_link(radial, height)
    else:
        shoulder_lift, elbow_flex, wrist_flex = _solve_three_link_with_priority(
            radial,
            height,
            tool_offset_m,
        )

    joints = _check_joint_limits(
        {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
            "wrist_roll": 0.0,
        }
    )
    return joints


def _check_joint_limits(joints: dict[str, float]) -> dict[str, float]:
    for name, angle in joints.items():
        lo, hi = JOINT_LIMITS_RAD[name]
        if angle < lo or angle > hi:
            raise ValueError(f"{name} angle out of limit: {angle:.4f} rad, valid=[{lo}, {hi}]")
    return joints


def _solve_two_link(radial: float, height: float) -> tuple[float, float, float]:
    distance = math.hypot(radial, height)
    cos_elbow = (
        distance * distance - UPPER_ARM_M * UPPER_ARM_M - FOREARM_M * FOREARM_M
    ) / (2.0 * UPPER_ARM_M * FOREARM_M)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))

    elbow_flex = math.acos(cos_elbow)
    shoulder_lift = math.atan2(height, radial) + math.atan2(
        FOREARM_M * math.sin(elbow_flex),
        UPPER_ARM_M + FOREARM_M * math.cos(elbow_flex),
    )
    return shoulder_lift, elbow_flex, 0.0


def _solve_three_link_with_priority(
    radial: float,
    height: float,
    tool_offset_m: float,
) -> tuple[float, float, float]:
    preferred_shoulder = math.atan2(height, radial)
    shoulder_lo, shoulder_hi = JOINT_LIMITS_RAD["shoulder_lift"]
    best_solution: tuple[float, float, float] | None = None
    best_score: tuple[float, float, float] | None = None

    steps = int(math.ceil((shoulder_hi - shoulder_lo) / ANGLE_SEARCH_STEP_RAD))
    for i in range(steps + 1):
        shoulder_lift = shoulder_lo + i * ANGLE_SEARCH_STEP_RAD
        if shoulder_lift > shoulder_hi:
            shoulder_lift = shoulder_hi

        elbow_origin_r = UPPER_ARM_M * math.cos(shoulder_lift)
        elbow_origin_z = UPPER_ARM_M * math.sin(shoulder_lift)
        remain_r = radial - elbow_origin_r
        remain_z = height - elbow_origin_z

        for elbow_flex, wrist_flex in _solve_remaining_two_link(
            remain_r,
            remain_z,
            shoulder_lift,
            tool_offset_m,
        ):
            if not _angle_in_limit("elbow_flex", elbow_flex):
                continue
            if not _angle_in_limit("wrist_flex", wrist_flex):
                continue

            # Priority: shoulder first, then elbow, then wrist.
            score = (
                abs(shoulder_lift - preferred_shoulder),
                abs(elbow_flex),
                abs(wrist_flex),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_solution = (shoulder_lift, elbow_flex, wrist_flex)

    if best_solution is None:
        raise ValueError("target has no valid three-link solution within joint limits")
    return best_solution


def _solve_remaining_two_link(
    remain_r: float,
    remain_z: float,
    shoulder_lift: float,
    tool_offset_m: float,
) -> list[tuple[float, float]]:
    distance = math.hypot(remain_r, remain_z)
    if distance > FOREARM_M + tool_offset_m:
        return []
    if distance < abs(FOREARM_M - tool_offset_m):
        return []

    cos_wrist = (
        distance * distance - FOREARM_M * FOREARM_M - tool_offset_m * tool_offset_m
    ) / (2.0 * FOREARM_M * tool_offset_m)
    cos_wrist = max(-1.0, min(1.0, cos_wrist))
    wrist_abs = math.acos(cos_wrist)

    solutions = []
    target_angle = math.atan2(remain_z, remain_r)
    for tool_relative_angle in (wrist_abs, -wrist_abs):
        forearm_abs_angle = target_angle - math.atan2(
            tool_offset_m * math.sin(tool_relative_angle),
            FOREARM_M + tool_offset_m * math.cos(tool_relative_angle),
        )
        elbow_flex = shoulder_lift - forearm_abs_angle
        wrist_flex = -tool_relative_angle
        solutions.append((elbow_flex, wrist_flex))
    return solutions


def _angle_in_limit(name: str, angle: float) -> bool:
    lo, hi = JOINT_LIMITS_RAD[name]
    return lo <= angle <= hi


def _min_reach(*lengths: float) -> float:
    longest = max(lengths)
    others = sum(lengths) - longest
    return max(0.0, longest - others)


def _config_range_rad(config: dict[str, int]) -> tuple[float, float]:
    if config["label"] == 0:
        return 0.0, math.pi
    if config["label"] == 1:
        return -math.pi / 2.0, math.pi / 2.0
    raise ValueError(f"unsupported joint label: {config['label']}")


JOINT_LIMITS_RAD = {
    name: _config_range_rad(config)
    for name, config in MY_CONFIG.items()
    if name != "gripper"
}
