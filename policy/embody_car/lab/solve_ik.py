from __future__ import annotations
import numpy as np
import math
from collections.abc import Sequence
from Angle_config import MY_CONFIG, angles_to_ticks 

# 单位: meter / radian
BASE_HEIGHT_M = 0.075
UPPER_ARM_M = 0.11257
FOREARM_M = 0.13490
# 腕关节到抓取参考点的等效长度。这里用实机手动对准样本校准到夹爪前方参考点。
TOOL_OFFSET_M = 0.16
ANGLE_SEARCH_STEP_RAD = math.radians(1.0)

# 这组偏好来自手动扳到目标 [0.2554, 0.015968, 0.22571] 时读取的 tick:
# 1:2048, 2:2483, 3:1247, 4:3173, 5:317。
PREFERRED_SHOULDER_LIFT_RAD = math.radians(54.5)
PREFERRED_ELBOW_FLEX_RAD = math.radians(16.7)
PREFERRED_WRIST_FLEX_RAD = math.radians(98.8)
PREFERRED_WRIST_ROLL_RAD = math.radians(-148.1)

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
    position_xyz = np.asarray(position_xyz, dtype=float).reshape(-1)
    if len(position_xyz) != 3:
        raise ValueError("position_xyz must be a 3D coordinate: (x, y, z)")

    x, y, z = [float(v) for v in position_xyz]
    shoulder_pan = -math.atan2(y, x)

    radial = math.hypot(x, y)
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
            "wrist_roll": PREFERRED_WRIST_ROLL_RAD,
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
    shoulder_lift = math.atan2(height, radial) - math.atan2(
        FOREARM_M * math.sin(elbow_flex),
        UPPER_ARM_M + FOREARM_M * math.cos(elbow_flex),
    )
    return shoulder_lift, elbow_flex, 0.0


def _solve_three_link_with_priority(
    radial: float,
    height: float,
    tool_offset_m: float,
) -> tuple[float, float, float]:
    shoulder_lo, shoulder_hi = JOINT_LIMITS_RAD["shoulder_lift"]
    best_solution: tuple[float, float, float] | None = None
    best_score: tuple[float, float, float] | None = None

    steps = int(math.ceil((shoulder_hi - shoulder_lo) / ANGLE_SEARCH_STEP_RAD))
    for i in range(steps + 1):
        shoulder_lift = shoulder_lo + i * ANGLE_SEARCH_STEP_RAD
        if shoulder_lift > shoulder_hi:
            shoulder_lift = shoulder_hi

        upper_abs_angle = shoulder_lift
        elbow_origin_r = UPPER_ARM_M * math.cos(upper_abs_angle)
        elbow_origin_z = UPPER_ARM_M * math.sin(upper_abs_angle)
        remain_r = radial - elbow_origin_r
        remain_z = height - elbow_origin_z

        for elbow_flex, wrist_flex in _solve_remaining_two_link(
            remain_r,
            remain_z,
            upper_abs_angle,
            tool_offset_m,
        ):
            if not _angle_in_limit("elbow_flex", elbow_flex):
                continue
            if not _angle_in_limit("wrist_flex", wrist_flex):
                continue

            # Priority: shoulder first, then elbow, then wrist.
            score = (
                abs(shoulder_lift - PREFERRED_SHOULDER_LIFT_RAD),
                abs(elbow_flex - PREFERRED_ELBOW_FLEX_RAD),
                abs(wrist_flex - PREFERRED_WRIST_FLEX_RAD),
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
    upper_abs_angle: float,
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
        elbow_flex = forearm_abs_angle - upper_abs_angle
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


def _config_tick_to_rad(config: dict[str, int], tick: int) -> float:
    if config["label"] == 0:
        ref_low, ref_mid, ref_high = 0.0, 90.0, 180.0
    elif config["label"] == 1:
        ref_low, ref_mid, ref_high = -90.0, 0.0, 90.0
    else:
        raise ValueError(f"unsupported joint label: {config['label']}")

    p_low, p_mid, p_high = config["a_low"], config["a_mid"], config["a_high"]
    if (p_low <= tick <= p_mid) or (p_mid <= tick <= p_low):
        pct = (tick - p_low) / (p_mid - p_low)
        deg = ref_low + (ref_mid - ref_low) * pct
    else:
        pct = (tick - p_mid) / (p_high - p_mid)
        deg = ref_mid + (ref_high - ref_mid) * pct
    return math.radians(deg)


def _config_range_rad(config: dict[str, int]) -> tuple[float, float]:
    range_a = _config_tick_to_rad(config, config["range_min"])
    range_b = _config_tick_to_rad(config, config["range_max"])
    return min(range_a, range_b), max(range_a, range_b)


JOINT_LIMITS_RAD = {
    name: _config_range_rad(config)
    for name, config in MY_CONFIG.items()
    if name != "gripper"
}

def forward_kinematics_check(joints: dict[str, float], target_xyz: tuple[float, float, float]) -> None:

    pan = joints["shoulder_pan"]
    lift = joints["shoulder_lift"]
    elbow = joints["elbow_flex"]
    wrist = joints["wrist_flex"]

    # 1. 计算各个连杆在二维平面 (R, Z) 的绝对朝向角
    abs_angle_upper = lift
    abs_angle_forearm = abs_angle_upper + elbow
    abs_angle_tool = abs_angle_forearm - wrist

    # 2. 从基座逐步叠加计算末端在平面上的 R 和 Z
    # 原点 Z 需要加上基座高度
    r_end = (
        UPPER_ARM_M * math.cos(abs_angle_upper) +
        FOREARM_M * math.cos(abs_angle_forearm) +
        TOOL_OFFSET_M * math.cos(abs_angle_tool)
    )
    z_end = (
        BASE_HEIGHT_M +
        UPPER_ARM_M * math.sin(abs_angle_upper) +
        FOREARM_M * math.sin(abs_angle_forearm) +
        TOOL_OFFSET_M * math.sin(abs_angle_tool)
    )

    # 3. 将平面的 R 转换回三维空间的 X 和 Y
    x_end = r_end * math.cos(pan)
    y_end = -r_end * math.sin(pan)

    # 4. 计算误差
    tx, ty, tz = target_xyz
    err_x = abs(x_end - tx)
    err_y = abs(y_end - ty)
    err_z = abs(z_end - tz)
    total_err = math.hypot(err_x, math.hypot(err_y, err_z))

    print(f"  --> [FK 校验] 实际末端到达: X={x_end:.4f}, Y={y_end:.4f}, Z={z_end:.4f}")
    print(f"  --> [误差] 距目标点偏差: {total_err * 1000:.2f} mm")
    if total_err > 0.001:  # 误差大于 1mm 时警告
        print("  --> [警告] 误差过大，请检查计算逻辑或硬件限位！")

def solve_ik(target: np.ndarray):
    result_joints = position_to_arm(target)
    servo_values = angles_to_ticks(result_joints)
    return servo_values

if __name__ == "__main__":
    # 定义几个典型的测试用例
    test_cases = [
        # (X, Y, Z) 单位：米
        ("正前方中等距离", (0.15, 0.0, 0.15)),
        ("斜前方较高位置", (0.15, 0.10, 0.25)),
        ("极近距离(可能触发特殊姿态)", (0.05, 0.05, 0.10)),
        ("超出工作空间(预期报错)", (0.50, 0.0, 0.20)),
    ]

    print("================ 机械臂 IK 测试开始 ================\n")

    for name, target in test_cases:
        print(f"▶ 测试用例: {name}")
        print(f"  目标坐标: X={target[0]:.4f}, Y={target[1]:.4f}, Z={target[2]:.4f}")
        
        try:
            # 调用你的核心函数
            result_joints = position_to_arm(target)
            
            # 将弧度转换为更容易人类阅读的角度值 (度)
            degrees = {k: math.degrees(v) for k, v in result_joints.items()}
            
            print(f"  计算结果 (度):")
            print(f"    - Shoulder Pan : {degrees['shoulder_pan']:>6.2f}°")
            print(f"    - Shoulder Lift: {degrees['shoulder_lift']:>6.2f}°")
            print(f"    - Elbow Flex   : {degrees['elbow_flex']:>6.2f}°")
            print(f"    - Wrist Flex   : {degrees['wrist_flex']:>6.2f}°")
            
            # 进行 FK 逆向校验
            forward_kinematics_check(result_joints, target)

        except ValueError as e:
            print(f"  [预期内的报错捕获] {e}")
        except Exception as e:
            print(f"  [意外崩溃] {e}")
            
        print("-" * 50)
