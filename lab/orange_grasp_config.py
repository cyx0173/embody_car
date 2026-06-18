"""Shared defaults for the orange wrist-grasp workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent

DEFAULT_TARGET = "orange"

DEFAULT_FOLLOWER_PORT = "/dev/cu.usbmodem5AE60562991"
DEFAULT_LEADER_PORT = "/dev/cu.usbmodem5AE60825831"
SERVO_IDS = (1, 2, 3, 4, 5, 6)

LOOP_INTERVAL_S = 0.012
MIN_DELTA_TICKS = 3
READ_RETRIES = 3
READ_RETRY_DELAY_S = 0.004
SERVO_TICKS_PER_TURN = 4096
LEADER_SPIKE_MAX_DELTA_TICKS = 1000
LEADER_SPIKE_CONFIRM_FRAMES = 3
FOLLOWER_MAX_TARGET_STEP_TICKS = 520
GRIPPER_MAX_TARGET_STEP_TICKS = 1200

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_MAP = {
    1: {"leader_min": 2132, "leader_max": 4851, "follower_min": 715, "follower_max": 3466},
    2: {"leader_min": 2596, "leader_max": 1006, "follower_min": 822, "follower_max": 3226, "wrap": True},
    3: {"leader_min": 224, "leader_max": 1954, "follower_min": 908, "follower_max": 3123},
    4: {"leader_min": 2024, "leader_max": 4352, "follower_min": 880, "follower_max": 3176, "follower_wrap": True},
    5: {"leader_min": 1990, "leader_max": 5789, "follower_min": 71, "follower_max": 3922},
    6: {"leader_min": 4473, "leader_max": 3295, "follower_min": 800, "follower_max": 2302},
}

GRIPPER_LEADER_OPEN = 4473
GRIPPER_LEADER_CLOSE = 3295
GRIPPER_FOLLOWER_OPEN = 2302
GRIPPER_FOLLOWER_CLOSE = 800

RESET_BEFORE_POS = {1: 2048, 2: 863, 3: 2962, 4: 2675, 5: 1029, 6: 2302}
RESET_AFTER_POS = {1: 2048, 2: 863, 3: 2962, 4: 2675, 5: 1029, 6: 1200}
RESET_ORDER_GROUPS = ((3, 4), (2,), (1, 5, 6))

ORANGE_APPROACH_SPEED = 170
ORANGE_WHEEL_SEARCH_SPEED = 150
ORANGE_WRIST_ALIGN_MAX_SPEED = 80
ORANGE_WRIST_ALIGN_MIN_SPEED = 12
ORANGE_WRIST_MIN_BOX_AREA_RATIO = 0.07
ORANGE_WRIST_READY_MIN_BOX_OVERLAP_RATIO = 0.30

ORANGE_POLICY_DURATION_S = 8.0
ORANGE_POLICY_FPS = 30.0
ORANGE_POLICY_SPEED = 1800
ORANGE_POLICY_ACC = 45
ORANGE_PLACE_POLICY_DURATION_S = 10.0
ORANGE_FINAL_CLOSE_POS = GRIPPER_FOLLOWER_CLOSE
ORANGE_PRE_CLOSE_SERVO4_OFFSET = 0
ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD = 1400
ORANGE_PRE_CLOSE_HOLD_S = 0.25

DEFAULT_DATASET_NAME = "orange_wrist_grasp_formal_clean_v1"
DEFAULT_DATASET_REPO_ID = f"embody_car/{DEFAULT_DATASET_NAME}"
DEFAULT_RECORD_DATASET_NAME = DEFAULT_DATASET_NAME
DEFAULT_RECORD_REPO_ID = DEFAULT_DATASET_REPO_ID
DEFAULT_RECORD_TASK = "grasp the orange with the gripper"
DEFAULT_PLACE_DATASET_NAME = "orange_to_bowl_wrist_place_v1"
DEFAULT_PLACE_REPO_ID = f"embody_car/{DEFAULT_PLACE_DATASET_NAME}"
DEFAULT_PLACE_TASK = "place the orange into the bowl"
DEFAULT_DUAL_PLACE_DATASET_NAME = "orange_to_bowl_dual_camera_place_v1"
DEFAULT_DUAL_PLACE_REPO_ID = f"embody_car/{DEFAULT_DUAL_PLACE_DATASET_NAME}"
DEFAULT_JOB_NAME = "orange_act_clean_live"

PLACE_RESET_OPEN_POS = dict(RESET_BEFORE_POS)
PLACE_RESET_HOLD_POS = dict(RESET_AFTER_POS)

DEFAULT_POLICY_PATH = (
    WORKSPACE_DIR
    / "outputs"
    / "train"
    / "orange_act_plus_dual_wrist_4gpu_v1"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)

DEFAULT_PLACE_POLICY_PATH = (
    WORKSPACE_DIR
    / "outputs"
    / "train"
    / "orange_to_bowl_wrist_merged_act_cuda_v1"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)

DEFAULT_DUAL_PLACE_POLICY_PATH = (
    WORKSPACE_DIR
    / "outputs"
    / "train"
    / "orange_to_bowl_marked_dual_act_plus_4gpu_v1"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)

DEFAULT_DUAL_GRASP_POLICY_PATH = (
    WORKSPACE_DIR
    / "outputs"
    / "train"
    / "orange_grasp_red_external_dual_act_4gpu_v1"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)


def clamp(pos: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(pos)))


def circular_distance(a: int, b: int) -> int:
    delta = abs((int(a) - int(b)) % SERVO_TICKS_PER_TURN)
    return min(delta, SERVO_TICKS_PER_TURN - delta)


def is_in_wrapped_range(pos: int, start: int, end: int, *, margin: int = 0) -> bool:
    span = (int(end) - int(start)) % SERVO_TICKS_PER_TURN
    delta = (int(pos) - int(start)) % SERVO_TICKS_PER_TURN
    return -int(margin) <= delta <= span + int(margin)


def clamp_wrapped_range(pos: int, start: int, end: int) -> int:
    pos = int(pos) % SERVO_TICKS_PER_TURN
    if is_in_wrapped_range(pos, start, end):
        return pos
    if circular_distance(pos, start) <= circular_distance(pos, end):
        return int(start) % SERVO_TICKS_PER_TURN
    return int(end) % SERVO_TICKS_PER_TURN


def clamp_follower_target(servo_id: int, pos: int) -> int:
    cfg = JOINT_MAP[servo_id]
    if cfg.get("follower_wrap"):
        return clamp_wrapped_range(pos, int(cfg["follower_min"]), int(cfg["follower_max"]))
    return clamp(pos, int(cfg["follower_min"]), int(cfg["follower_max"]))


def follower_position_in_range(servo_id: int, pos: int, *, margin: int = 0) -> bool:
    cfg = JOINT_MAP.get(servo_id)
    if cfg is None:
        return False
    if cfg.get("follower_wrap"):
        return is_in_wrapped_range(
            pos,
            int(cfg["follower_min"]),
            int(cfg["follower_max"]),
            margin=margin,
        )
    lower = int(cfg["follower_min"]) - int(margin)
    upper = int(cfg["follower_max"]) + int(margin)
    return lower <= int(pos) <= upper


def parse_ids(text: str) -> tuple[int, ...]:
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return tuple(ids)


def normalize_servo_reading(pos: int) -> int:
    if pos >= 32768:
        return pos - 32768
    return pos


def parse_positions(text: str) -> dict[int, int]:
    positions: dict[int, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        servo_id, value = part.split(":", maxsplit=1)
        positions[int(servo_id)] = int(value)
    return positions


def positions_to_array(
    positions: dict[int, int],
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    values: list[float] = []
    for idx, servo_id in enumerate(SERVO_IDS):
        pos = positions.get(servo_id)
        if pos is None:
            if fallback is None:
                pos = 0
            else:
                pos = int(fallback[idx])
        values.append(float(pos))
    return np.asarray(values, dtype=np.float32)


def sanitize_follower_positions(
    positions: dict[int, int],
    *,
    last_state: np.ndarray | None,
    max_jump: int,
    range_margin: int,
) -> dict[int, int]:
    clean: dict[int, int] = {}
    for servo_id, pos in positions.items():
        cfg = JOINT_MAP.get(servo_id)
        if cfg is None:
            continue

        if not follower_position_in_range(servo_id, int(pos), margin=range_margin):
            continue

        if last_state is not None and max_jump > 0:
            previous = int(last_state[servo_id - 1])
            if abs(int(pos) - previous) > max_jump:
                continue

        clean[servo_id] = int(pos)
    return clean
