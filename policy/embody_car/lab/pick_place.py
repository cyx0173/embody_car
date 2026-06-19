from __future__ import annotations

import argparse
from pathlib import Path

from policy_config import (
    DEFAULT_FOLLOWER_PORT,
    GRIPPER_FOLLOWER_OPEN,
    PLACE_POLICY_DURATION_S,
    POLICY_FPS,
    POLICY_SPEED,
)
from run_grasp_clean import (
    DEFAULT_POLICY_PATH as DEFAULT_GRASP_POLICY_PATH,
    DualCameraGraspRunner,
    build_arg_parser as build_runner_arg_parser,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DEFAULT_PLACE_POLICY_PATH = PROJECT_DIR / "place_model"
DEFAULT_PLACE_FORWARD_STEP = 75
DEFAULT_PLACE_BACK_SERVO = 4
DEFAULT_PLACE_BACK_STEP = -45


def normalize_bowl(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"l", "left", "leftmost"}:
        return "left"
    if text in {"r", "right", "rightmost", ""}:
        return "right"
    return "right"


def bowl_marker_choice(target_bowl: str) -> str:
    return "leftmost" if target_bowl == "left" else "rightmost"


class PickPlaceSkill:
    """Voice-callable wrapper for clean dual-camera grasp -> marked bowl place."""

    def __init__(
        self,
        *,
        grasp_policy_path: Path = DEFAULT_GRASP_POLICY_PATH,
        place_policy_path: Path = DEFAULT_PLACE_POLICY_PATH,
        follower_port: str = DEFAULT_FOLLOWER_PORT,
        wrist_camera: int = 2,
        external_camera: int = 0,
        marker_model: Path = BASE_DIR / "yolo11s.pt",
        marker_device: str = "auto",
        fps: float = POLICY_FPS,
        speed: int = POLICY_SPEED,
        grasp_duration_s: float = 25.0,
        place_duration_s: float = PLACE_POLICY_DURATION_S,
        show: bool = True,
        execute: bool = True,
    ) -> None:
        self.grasp_policy_path = grasp_policy_path
        self.place_policy_path = place_policy_path
        self.follower_port = follower_port
        self.wrist_camera = wrist_camera
        self.external_camera = external_camera
        self.marker_model = marker_model
        self.marker_device = marker_device
        self.fps = fps
        self.speed = speed
        self.grasp_duration_s = grasp_duration_s
        self.place_duration_s = place_duration_s
        self.show = show
        self.execute = execute

    def _base_runner_args(self) -> argparse.Namespace:
        runner_args = build_runner_arg_parser().parse_args([])
        runner_args.follower_port = self.follower_port
        runner_args.wrist_camera = self.wrist_camera
        runner_args.external_camera = self.external_camera
        runner_args.fps = self.fps
        runner_args.speed = self.speed
        runner_args.marker_model = str(self.marker_model)
        runner_args.marker_device = self.marker_device
        runner_args.show = self.show
        runner_args.execute = self.execute
        runner_args.quiet_controls = True
        return runner_args

    def _run_grasp(self, target: str) -> str:
        args = self._base_runner_args()
        args.policy_path = self.grasp_policy_path
        args.duration_s = self.grasp_duration_s
        args.marker_target = target
        args.marker_choose = "largest"
        args.nudge_enabled = True
        args.nudge_step = 45
        args.nudge_max = 320
        args.nudge_left_delta = 45
        args.nudge_forward_delta = 45
        args.extra_nudge_enabled = True
        args.extra_nudge_step = 35
        args.nudge_up_delta = 35
        args.nudge_wrist_delta = 35
        args.nudge_roll_delta = 35
        args.next_stage_key = "n"
        args.open_gripper_key = "o"
        args.manual_gripper_step = 220
        args.manual_open_step = 180
        args.manual_open_hold_s = 0.7
        args.manual_open_stutter_steps = 3
        args.manual_open_stutter_pause_s = 0.04
        args.lock_gripper_until_open_key = True
        return DualCameraGraspRunner(args).run()

    def _run_place(self, target_bowl: str) -> str:
        args = self._base_runner_args()
        args.policy_path = self.place_policy_path
        args.duration_s = self.place_duration_s
        args.marker_target = "bowl"
        args.marker_choose = bowl_marker_choice(target_bowl)
        args.close_gripper_key = ""
        args.nudge_enabled = True
        args.nudge_step = 45
        args.nudge_max = 320
        args.nudge_left_delta = 45
        args.nudge_forward_delta = DEFAULT_PLACE_FORWARD_STEP
        args.nudge_back_servo = DEFAULT_PLACE_BACK_SERVO
        args.nudge_back_delta = DEFAULT_PLACE_BACK_STEP
        args.next_stage_key = "n"
        args.open_gripper_key = "t"
        args.manual_open_pos = GRIPPER_FOLLOWER_OPEN
        args.manual_open_hold_s = 1.2
        args.manual_open_step = 260
        args.manual_open_stutter_steps = 5
        args.manual_open_stutter_pause_s = 0.08
        args.lock_gripper_until_open_key = True
        return DualCameraGraspRunner(args).run()

    def run(self, *, target: str, target_bowl: str | None = None) -> str:
        target = (target or "apple").strip().lower()
        bowl = normalize_bowl(target_bowl)

        print(
            "Voice pick-place controls: "
            "stage1 w/s/a/d=nudge, e/c=up/down, i/k=wrist, j/l=roll, "
            "t=close little, o=open little, n=place; "
            "stage2 w=servo2 forward lean, s=servo4 adjust, a/d=left/right, t=release, n=finish."
        )
        reason = self._run_grasp(target)
        if reason == "quit":
            return "已停止抓取放置流程。"

        reason = self._run_place(bowl)
        if reason == "quit":
            return "已停止放置流程。"
        return f"已完成{target}到{bowl}边碗的抓取放置，可以继续下达指令。"
