#!/usr/bin/env python3
"""Test clean grasp runner followed by marked left/right bowl placement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    GRIPPER_FOLLOWER_OPEN,
    ORANGE_PLACE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
)
from run_grasp_clean import (
    DEFAULT_POLICY_PATH as DEFAULT_GRASP_POLICY_PATH,
    DualCameraGraspRunner,
    build_arg_parser as build_grasp_arg_parser,
)

DEFAULT_PLACE_POLICY_PATH = PROJECT_DIR / "place_model"
DEFAULT_PLACE_GRIPPER_STUTTER_STEPS = 5
DEFAULT_PLACE_GRIPPER_STUTTER_PAUSE_S = 0.08
DEFAULT_PLACE_OPEN_STEP = 260
DEFAULT_GRASP_OPEN_STEP = 180
DEFAULT_GRASP_CLOSE_STEP = 220
DEFAULT_PLACE_FORWARD_STEP = 75
DEFAULT_PLACE_BACK_SERVO = 4
DEFAULT_PLACE_BACK_STEP = -45


def normalize_bowl(value: str) -> str:
    text = value.strip().lower()
    if text in {"l", "left", "leftmost"}:
        return "left"
    if text in {"r", "right", "rightmost"}:
        return "right"
    raise ValueError(f"target bowl must be left/right/l/r, got {value!r}")


def bowl_marker_choice(target_bowl: str) -> str:
    return "leftmost" if target_bowl == "left" else "rightmost"


def run_policy_stage(
    args: argparse.Namespace,
    *,
    label: str,
    policy_path: Path,
    duration_s: float,
    marker_target: str,
    marker_choose: str,
    record: bool,
    allow_manual_grasp: bool,
    enable_place_controls: bool,
) -> str:
    runner_args = build_grasp_arg_parser().parse_args([])
    runner_args.policy_path = policy_path
    runner_args.follower_port = args.follower_port
    runner_args.wrist_camera = args.wrist_camera
    runner_args.external_camera = args.external_camera
    runner_args.fps = args.fps
    runner_args.speed = args.speed
    runner_args.duration_s = duration_s
    runner_args.marker_target = marker_target
    runner_args.marker_choose = marker_choose
    runner_args.marker_model = str(args.marker_model)
    runner_args.marker_device = args.marker_device
    runner_args.show = args.show
    runner_args.execute = args.execute
    runner_args.record = record
    runner_args.record_images = args.record_images and record
    runner_args.quiet_controls = True
    if not allow_manual_grasp:
        runner_args.close_gripper_key = ""
    if allow_manual_grasp and args.grasp_controls:
        runner_args.nudge_enabled = True
        runner_args.nudge_step = args.grasp_nudge_step
        runner_args.nudge_max = args.grasp_nudge_max
        runner_args.nudge_left_delta = args.grasp_nudge_step
        runner_args.nudge_forward_delta = args.grasp_nudge_step
        runner_args.extra_nudge_enabled = args.grasp_extra_controls
        runner_args.extra_nudge_step = args.grasp_extra_nudge_step
        runner_args.nudge_up_delta = args.grasp_extra_nudge_step
        runner_args.nudge_wrist_delta = args.grasp_extra_nudge_step
        runner_args.nudge_roll_delta = args.grasp_extra_nudge_step
        runner_args.next_stage_key = args.next_stage_key
        runner_args.open_gripper_key = args.grasp_open_key
        runner_args.manual_gripper_step = args.grasp_close_step
        runner_args.manual_open_step = args.grasp_open_step
        runner_args.manual_open_hold_s = args.grasp_open_hold_s
        runner_args.manual_open_stutter_steps = args.grasp_open_stutter_steps
        runner_args.manual_open_stutter_pause_s = args.grasp_open_stutter_pause_s
        runner_args.lock_gripper_until_open_key = True
    if enable_place_controls:
        runner_args.nudge_enabled = True
        runner_args.nudge_step = args.place_nudge_step
        runner_args.nudge_max = args.place_nudge_max
        runner_args.nudge_left_delta = args.place_nudge_step
        runner_args.nudge_forward_delta = args.place_forward_step
        runner_args.nudge_back_servo = args.place_back_servo
        runner_args.nudge_back_delta = args.place_back_step
        runner_args.next_stage_key = args.next_stage_key
        runner_args.open_gripper_key = args.place_open_key
        runner_args.manual_open_hold_s = args.place_open_hold_s
        runner_args.manual_open_pos = args.place_open_pos
        runner_args.manual_open_stutter_steps = args.place_gripper_stutter_steps
        runner_args.manual_open_stutter_pause_s = args.place_gripper_stutter_pause_s
        runner_args.manual_open_step = args.place_open_step
        runner_args.lock_gripper_until_open_key = True

    print(f"\n=== {label} ===")
    print(
        f"policy={runner_args.policy_path} marker={runner_args.marker_target} "
        f"choose={runner_args.marker_choose}"
    )
    return DualCameraGraspRunner(runner_args).run()


def confirm_grasp_before_place(args: argparse.Namespace, attempt: int) -> str:
    if args.no_prompt or args.transition_mode == "auto":
        return "continue"

    prompt = (
        "\nCheck the gripper. "
        "Press Enter to place, type r then Enter to retry grasp, or q then Enter to stop: "
    )
    answer = input(prompt).strip().lower()
    if answer in {"r", "retry"}:
        if attempt >= args.max_grasp_attempts:
            print(f"Max grasp attempts reached ({args.max_grasp_attempts}).")
            return "quit"
        return "retry"
    if answer in {"q", "quit", "stop"}:
        return "quit"
    return "continue"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test clean grasp + marked left/right bowl place without changing the main flow."
    )
    parser.add_argument("target_bowl", choices=("l", "r", "left", "right"))
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--grasp-duration-s", type=float, default=25.0)
    parser.add_argument("--grasp-policy-path", type=Path, default=DEFAULT_GRASP_POLICY_PATH)
    parser.add_argument("--place-duration-s", type=float, default=ORANGE_PLACE_POLICY_DURATION_S)
    parser.add_argument("--place-policy-path", type=Path, default=DEFAULT_PLACE_POLICY_PATH)
    parser.add_argument("--marker-model", type=Path, default=BASE_DIR / "yolo11s.pt")
    parser.add_argument("--marker-device", default="auto")
    parser.add_argument("--grasp-marker-target", default="apple")
    parser.add_argument("--place-marker-target", default="bowl")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--record-grasp", action="store_true")
    parser.add_argument("--record-place", action="store_true")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--grasp-controls", dest="grasp_controls", action="store_true", default=True)
    parser.add_argument("--no-grasp-controls", dest="grasp_controls", action="store_false")
    parser.add_argument("--grasp-nudge-step", type=int, default=45)
    parser.add_argument("--grasp-nudge-max", type=int, default=320)
    parser.add_argument("--grasp-extra-controls", dest="grasp_extra_controls", action="store_true", default=True)
    parser.add_argument("--no-grasp-extra-controls", dest="grasp_extra_controls", action="store_false")
    parser.add_argument("--grasp-extra-nudge-step", type=int, default=35)
    parser.add_argument("--grasp-close-step", type=int, default=DEFAULT_GRASP_CLOSE_STEP)
    parser.add_argument("--grasp-open-key", default="o")
    parser.add_argument("--grasp-open-step", type=int, default=DEFAULT_GRASP_OPEN_STEP)
    parser.add_argument("--grasp-open-hold-s", type=float, default=0.7)
    parser.add_argument("--grasp-open-stutter-steps", type=int, default=3)
    parser.add_argument("--grasp-open-stutter-pause-s", type=float, default=0.04)
    parser.add_argument("--next-stage-key", default="n")
    parser.add_argument("--place-controls", dest="place_controls", action="store_true", default=True)
    parser.add_argument("--no-place-controls", dest="place_controls", action="store_false")
    parser.add_argument("--place-nudge-step", type=int, default=45)
    parser.add_argument("--place-nudge-max", type=int, default=320)
    parser.add_argument("--place-forward-step", type=int, default=DEFAULT_PLACE_FORWARD_STEP)
    parser.add_argument("--place-back-servo", type=int, default=DEFAULT_PLACE_BACK_SERVO)
    parser.add_argument("--place-back-step", type=int, default=DEFAULT_PLACE_BACK_STEP)
    parser.add_argument("--place-open-key", default="t")
    parser.add_argument("--place-open-pos", type=int, default=GRIPPER_FOLLOWER_OPEN)
    parser.add_argument("--place-open-hold-s", type=float, default=1.2)
    parser.add_argument("--place-open-step", type=int, default=DEFAULT_PLACE_OPEN_STEP)
    parser.add_argument("--place-gripper-stutter-steps", type=int, default=DEFAULT_PLACE_GRIPPER_STUTTER_STEPS)
    parser.add_argument("--place-gripper-stutter-pause-s", type=float, default=DEFAULT_PLACE_GRIPPER_STUTTER_PAUSE_S)
    parser.add_argument("--transition-mode", choices=("prompt", "auto"), default="prompt")
    parser.add_argument("--max-grasp-attempts", type=int, default=2)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--show", dest="show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--execute", dest="execute", action="store_true", default=True)
    parser.add_argument("--dry-run", dest="execute", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.target_bowl = normalize_bowl(args.target_bowl)
    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")

    print(
        "Controls in preview window: "
        "grasp w/s/a/d=nudge, e/c=up/down, i/k=wrist, j/l=roll, "
        "t=close little, o=open little, n=place; "
        "place w=servo2 forward lean, s=servo4 adjust, a/d=left/right, t=release, n=finish."
    )

    if not args.skip_grasp:
        for attempt in range(1, max(1, args.max_grasp_attempts) + 1):
            reason = run_policy_stage(
                args,
                label=f"Stage 1/2: grasp with clean ACT runner (attempt {attempt})",
                policy_path=args.grasp_policy_path,
                duration_s=args.grasp_duration_s,
                marker_target=args.grasp_marker_target,
                marker_choose="largest",
                record=args.record_grasp,
                allow_manual_grasp=True,
                enable_place_controls=False,
            )
            if reason == "next":
                break
            if reason == "quit":
                return
            decision = confirm_grasp_before_place(args, attempt)
            if decision == "continue":
                break
            if decision == "quit":
                return
        else:
            return
    elif not args.no_prompt:
        input(f"\nObject should already be held. Press Enter to place into the {args.target_bowl} bowl.")

    run_policy_stage(
        args,
        label=f"Stage 2/2: place into {args.target_bowl} bowl",
        policy_path=args.place_policy_path,
        duration_s=args.place_duration_s,
        marker_target=args.place_marker_target,
        marker_choose=bowl_marker_choice(args.target_bowl),
        record=args.record_place,
        allow_manual_grasp=False,
        enable_place_controls=args.place_controls,
    )


if __name__ == "__main__":
    main()
