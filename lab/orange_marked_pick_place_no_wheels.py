#!/usr/bin/env python3
"""Grasp an orange, then place it into a marked bowl without moving the wheels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from orange_grasp_config import (
    DEFAULT_DUAL_GRASP_POLICY_PATH,
    DEFAULT_DUAL_PLACE_POLICY_PATH,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_POLICY_PATH,
    ORANGE_FINAL_CLOSE_POS,
    ORANGE_PLACE_POLICY_DURATION_S,
    ORANGE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD,
    ORANGE_PRE_CLOSE_HOLD_S,
    ORANGE_PRE_CLOSE_SERVO4_OFFSET,
)


def normalize_target_bowl(target_bowl: str) -> str:
    value = target_bowl.strip().lower()
    if value in {"l", "left", "leftmost"}:
        return "left"
    if value in {"r", "right", "rightmost"}:
        return "right"
    if value == "prompt":
        return "prompt"
    raise ValueError(f"target_bowl must be l/r/left/right/prompt, got {target_bowl!r}")


def choose_target_bowl_at_start(args: argparse.Namespace) -> str:
    target = normalize_target_bowl(args.target_bowl_pos or args.target_bowl)
    if target != "prompt":
        return target
    if args.no_prompt:
        raise ValueError("--target-bowl prompt cannot be used together with --no-prompt.")
    while True:
        raw = input("\nSelect target bowl for the full pick-and-place run [l/r]: ").strip().lower()
        if raw in {"l", "left", "leftmost"}:
            return "left"
        if raw in {"r", "right", "rightmost"}:
            return "right"
        print("Please enter l or r.")


def run_command(command: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_grasp(args: argparse.Namespace) -> None:
    if args.single_camera_grasp:
        command = [
            sys.executable,
            str(BASE_DIR / "run_orange_act_policy.py"),
            "--policy-path",
            str(args.single_grasp_policy_path),
            "--follower-port",
            args.follower_port,
            "--wrist-camera",
            str(args.wrist_camera),
            "--duration-s",
            str(args.grasp_duration_s),
            "--fps",
            str(args.fps),
            "--speed",
            str(args.speed),
            "--pre-close-servo4-offset",
            str(args.pre_close_servo4_offset),
            "--pre-close-gripper-threshold",
            str(args.pre_close_gripper_threshold),
            "--pre-close-hold-s",
            str(args.pre_close_hold_s),
            "--final-close",
            "--final-close-pos",
            str(args.final_close_pos),
        ]
    else:
        marker_model = args.grasp_marker_model or args.marker_model
        marker_device = args.grasp_marker_device or args.marker_device
        command = [
            sys.executable,
            str(BASE_DIR / "run_orange_dual_camera_grasp_policy.py"),
            "--policy-path",
            str(args.grasp_policy_path),
            "--follower-port",
            args.follower_port,
            "--wrist-camera",
            str(args.wrist_camera),
            "--external-camera",
            str(args.external_camera),
            "--duration-s",
            str(args.grasp_duration_s),
            "--fps",
            str(args.fps),
            "--speed",
            str(args.speed),
            "--final-close",
            "--final-close-pos",
            str(args.final_close_pos),
            "--marker-model",
            str(marker_model),
            "--marker-target",
            args.grasp_marker_target,
            "--marker-device",
            marker_device,
            "--marker-conf",
            str(args.grasp_marker_conf),
            "--marker-iou",
            str(args.grasp_marker_iou),
            "--marker-mode",
            args.grasp_marker_mode,
            "--marker-choose",
            args.grasp_marker_choose,
        ]
    if args.grasp_reset_before:
        command.append("--reset-before")
    if args.return_to_holding_reset_after_grasp:
        command.append("--return-after-close")
    if args.show:
        command.append("--show")
    if args.execute:
        command.append("--execute")
    run_command(command, label="Stage 1/2: grasp orange")


def run_marked_place(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(BASE_DIR / "run_orange_to_bowl_dual_camera_policy.py"),
        "--policy-path",
        str(args.place_policy_path),
        "--follower-port",
        args.follower_port,
        "--wrist-camera",
        str(args.wrist_camera),
        "--external-camera",
        str(args.external_camera),
        "--duration-s",
        str(args.place_duration_s),
        "--fps",
        str(args.fps),
        "--speed",
        str(args.speed),
        "--mark-external-bowl",
        "--marker-target",
        args.marker_target,
        "--target-bowl",
        args.target_bowl,
    ]
    if args.marker_model is not None:
        command.extend(["--marker-model", str(args.marker_model)])
    if args.marker_device:
        command.extend(["--marker-device", args.marker_device])
    if args.return_open_after_place:
        command.append("--return-open-after")
    if args.show:
        command.append("--show")
    if args.execute:
        command.append("--execute")
    run_command(command, label="Stage 2/2: place orange into marked bowl")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No-wheel orange pick-and-place. The orange and two bowls should "
            "already be visible; the external camera marks the target bowl."
        )
    )
    parser.add_argument(
        "target_bowl_pos",
        nargs="?",
        choices=("l", "r", "left", "right"),
        help="Target bowl selected at the start of the full run.",
    )
    parser.add_argument("--grasp-policy-path", type=Path, default=DEFAULT_DUAL_GRASP_POLICY_PATH)
    parser.add_argument("--single-grasp-policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--place-policy-path", type=Path, default=DEFAULT_DUAL_PLACE_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--grasp-duration-s", type=float, default=ORANGE_POLICY_DURATION_S)
    parser.add_argument("--place-duration-s", type=float, default=ORANGE_PLACE_POLICY_DURATION_S)
    parser.add_argument("--final-close-pos", type=int, default=ORANGE_FINAL_CLOSE_POS)
    parser.add_argument("--pre-close-servo4-offset", type=int, default=ORANGE_PRE_CLOSE_SERVO4_OFFSET)
    parser.add_argument("--pre-close-gripper-threshold", type=int, default=ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD)
    parser.add_argument("--pre-close-hold-s", type=float, default=ORANGE_PRE_CLOSE_HOLD_S)
    parser.add_argument(
        "--target-bowl",
        choices=("prompt", "l", "r", "left", "right"),
        default="prompt",
        help="Which bowl should receive the orange. Use l/r to avoid any mid-run prompt.",
    )
    parser.add_argument("--marker-target", default="bowl", help="YOLO class used to find bowls in the external view.")
    parser.add_argument("--marker-model", type=Path, default=BASE_DIR / "yolo11s.pt")
    parser.add_argument("--marker-device", default="cpu")
    parser.add_argument(
        "--single-camera-grasp",
        action="store_true",
        help="Use the old wrist-only grasp policy instead of the red-marked dual-camera grasp policy.",
    )
    parser.add_argument("--grasp-marker-target", default="orange")
    parser.add_argument("--grasp-marker-model", type=Path, default=None)
    parser.add_argument("--grasp-marker-device", default=None)
    parser.add_argument("--grasp-marker-conf", type=float, default=0.25)
    parser.add_argument("--grasp-marker-iou", type=float, default=0.7)
    parser.add_argument("--grasp-marker-mode", choices=("bbox", "mask", "both"), default="bbox")
    parser.add_argument(
        "--grasp-marker-choose",
        choices=("largest", "highest_conf", "nearest_center", "leftmost", "rightmost"),
        default="largest",
    )
    parser.add_argument(
        "--no-grasp-reset-before",
        dest="grasp_reset_before",
        action="store_false",
        help="Do not reset the arm before the grasp policy.",
    )
    parser.add_argument(
        "--no-return-to-holding-reset-after-grasp",
        dest="return_to_holding_reset_after_grasp",
        action="store_false",
        help="Leave the arm at the end of the grasp instead of returning to reset while holding the orange.",
    )
    parser.add_argument("--return-open-after-place", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true", help="Only run the marked-bowl place policy.")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move the follower arm.")
    parser.set_defaults(
        grasp_reset_before=True,
        return_to_holding_reset_after_grasp=True,
    )
    return parser


def orange_marked_pick_place(
    *,
    target_bowl: str,
    show: bool = False,
    execute: bool = False,
    **overrides,
) -> None:
    """Run no-wheel orange grasp and place into a preselected marked bowl.

    This is the programmatic interface intended for main.py / voice control.
    The caller must pass target_bowl="left" or "right" so the flow can run
    without an interactive prompt.
    """
    args = build_arg_parser().parse_args([])
    args.target_bowl = normalize_target_bowl(target_bowl)
    args.show = show
    args.execute = execute
    args.no_prompt = True
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown orange marked pick-place option: {key}")
        setattr(args, key, value)

    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")
    if not args.execute:
        print("DRY RUN: add execute=True to move the follower arm.")
    if not args.skip_grasp:
        run_grasp(args)
    run_marked_place(args)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.target_bowl = choose_target_bowl_at_start(args)
    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")
    if not args.execute:
        print("DRY RUN: add --execute to move the follower arm.")
    if not args.skip_grasp:
        run_grasp(args)
    run_marked_place(args)


if __name__ == "__main__":
    main()
