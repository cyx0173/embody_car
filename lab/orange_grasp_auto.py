#!/usr/bin/env python3
"""Navigate with the chassis until orange is wrist-ready, then run ACT grasp."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from grasp_pipeline import (
    BASE_SEARCH_TIMEOUT_S,
    FRAME_MODE_LETTERBOX,
    NAV_FRAME_HEIGHT,
    NAV_FRAME_WIDTH,
    TargetDetection,
    WRIST_READY_CENTER_X_DEADZONE_PX,
    WRIST_READY_CENTER_Y_DEADZONE_PX,
    WRIST_MIN_CONF,
    WRIST_TARGET_OFFSET_X_PX,
    WRIST_TARGET_OFFSET_Y_PX,
    GraspPipeline,
)
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_POLICY_PATH,
    DEFAULT_TARGET,
    ORANGE_APPROACH_SPEED,
    ORANGE_FINAL_CLOSE_POS,
    ORANGE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD,
    ORANGE_PRE_CLOSE_HOLD_S,
    ORANGE_PRE_CLOSE_SERVO4_OFFSET,
    ORANGE_WHEEL_SEARCH_SPEED,
    ORANGE_WRIST_ALIGN_MAX_SPEED,
    ORANGE_WRIST_ALIGN_MIN_SPEED,
    ORANGE_WRIST_MIN_BOX_AREA_RATIO,
    ORANGE_WRIST_READY_MIN_BOX_OVERLAP_RATIO,
)


BASE_DIR = Path(__file__).resolve().parent


def build_default_args(**overrides) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown orange grasp option: {key}")
        setattr(args, key, value)
    return args


def run_policy(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(BASE_DIR / "run_orange_act_policy.py"),
        "--policy-path",
        str(args.policy_path),
        "--follower-port",
        args.follower_port,
        "--wrist-camera",
        str(args.hand_camera_id),
        "--duration-s",
        str(args.policy_duration_s),
        "--fps",
        str(args.policy_fps),
        "--speed",
        str(args.policy_speed),
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
    if args.policy_reset_before:
        command.append("--reset-before")
    if args.return_after_close:
        command.append("--return-after-close")
    if args.show:
        command.append("--show")
    if args.execute:
        command.append("--execute")

    print("Starting ACT grasp policy:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_wrist_ready_navigation(args: argparse.Namespace) -> TargetDetection:
    print(
        "Wrist-ready target offset: "
        f"x={args.wrist_target_offset_x:+.0f}px, "
        f"y={args.wrist_target_offset_y:+.0f}px"
    )
    print(
        "Wrist-ready deadzone: "
        f"x={args.wrist_ready_deadzone_x:.0f}px, "
        f"y={args.wrist_ready_deadzone_y:.0f}px, "
        f"overlap>={args.wrist_ready_min_overlap:.2f}, "
        f"area>={args.wrist_min_box_area:.3f}"
    )
    if not args.execute:
        print("DRY RUN: wheels and follower arm will not move. Add --execute to move hardware.")
    print(
        "Navigation frame: "
        f"mode={args.nav_frame_mode}, size={args.nav_frame_width}x{args.nav_frame_height}"
    )
    print(
        "Wheel speeds: "
        f"approach={args.approach_speed}, "
        f"search={args.wheel_search_speed}, "
        f"wrist_align={args.wrist_align_min_speed}-{args.wrist_align_max_speed}"
    )

    pipeline = GraspPipeline(
        model_path=args.model,
        timeout_s=args.timeout,
        show=args.show,
        dry_run=not args.execute,
        approach_speed=args.approach_speed,
        hand_camera_id=args.hand_camera_id,
        base_camera_id=args.base_camera_id,
        frame_mode=args.nav_frame_mode,
        frame_width=args.nav_frame_width,
        frame_height=args.nav_frame_height,
        wrist_target_offset_x_px=args.wrist_target_offset_x,
        wrist_target_offset_y_px=args.wrist_target_offset_y,
        wrist_ready_center_x_deadzone_px=args.wrist_ready_deadzone_x,
        wrist_ready_center_y_deadzone_px=args.wrist_ready_deadzone_y,
        wrist_ready_min_box_overlap_ratio=args.wrist_ready_min_overlap,
        wrist_min_conf=args.wrist_min_conf,
        wrist_min_box_area_ratio=args.wrist_min_box_area,
        wheel_search_speed=args.wheel_search_speed,
        wrist_align_max_speed=args.wrist_align_max_speed,
        wrist_align_min_speed=args.wrist_align_min_speed,
    )
    try:
        detection = pipeline.approach_until_wrist_ready(args.target)
        print(
            "Wrist target ready: "
            f"uv={detection.uv}, conf={detection.conf:.2f}, "
            f"area={detection.box_area_ratio:.3f}"
        )
        return detection
    finally:
        pipeline.close()


def orange_grasp(
    *,
    target: str = DEFAULT_TARGET,
    show: bool = False,
    execute: bool = False,
    **overrides,
) -> TargetDetection:
    """Run the full orange grasp flow for use by main.py or other modules.

    The function intentionally keeps execute=False by default, matching the CLI
    safety behavior. Callers that really want hardware motion must pass
    execute=True.
    """
    args = build_default_args(target=target, show=show, execute=execute, **overrides)
    detection = run_wrist_ready_navigation(args)
    run_policy(args)
    return detection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full orange grasp: wheel visual servo to wrist-ready pose, then ACT grasp."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--timeout", type=float, default=BASE_SEARCH_TIMEOUT_S)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move wheels and follower arm.")
    parser.add_argument("--hand-camera-id", type=int, default=1)
    parser.add_argument("--base-camera-id", type=int, default=0)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--approach-speed", type=int, default=ORANGE_APPROACH_SPEED)
    parser.add_argument("--wheel-search-speed", type=int, default=ORANGE_WHEEL_SEARCH_SPEED)
    parser.add_argument("--wrist-align-max-speed", type=int, default=ORANGE_WRIST_ALIGN_MAX_SPEED)
    parser.add_argument("--wrist-align-min-speed", type=int, default=ORANGE_WRIST_ALIGN_MIN_SPEED)
    parser.add_argument("--nav-frame-mode", choices=("crop", "letterbox"), default=FRAME_MODE_LETTERBOX)
    parser.add_argument("--nav-frame-width", type=int, default=NAV_FRAME_WIDTH)
    parser.add_argument("--nav-frame-height", type=int, default=NAV_FRAME_HEIGHT)

    parser.add_argument("--wrist-target-offset-x", type=float, default=WRIST_TARGET_OFFSET_X_PX)
    parser.add_argument("--wrist-target-offset-y", type=float, default=WRIST_TARGET_OFFSET_Y_PX)
    parser.add_argument("--wrist-ready-deadzone-x", type=float, default=70.0)
    parser.add_argument("--wrist-ready-deadzone-y", type=float, default=70.0)
    parser.add_argument("--wrist-ready-min-overlap", type=float, default=ORANGE_WRIST_READY_MIN_BOX_OVERLAP_RATIO)
    parser.add_argument("--wrist-min-conf", type=float, default=WRIST_MIN_CONF)
    parser.add_argument("--wrist-min-box-area", type=float, default=ORANGE_WRIST_MIN_BOX_AREA_RATIO)

    parser.add_argument("--policy-duration-s", type=float, default=ORANGE_POLICY_DURATION_S)
    parser.add_argument("--policy-fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--policy-speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--no-policy-reset-before", dest="policy_reset_before", action="store_false")
    parser.add_argument("--pre-close-servo4-offset", type=int, default=ORANGE_PRE_CLOSE_SERVO4_OFFSET)
    parser.add_argument("--pre-close-gripper-threshold", type=int, default=ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD)
    parser.add_argument("--pre-close-hold-s", type=float, default=ORANGE_PRE_CLOSE_HOLD_S)
    parser.add_argument("--final-close-pos", type=int, default=ORANGE_FINAL_CLOSE_POS)
    parser.add_argument("--return-after-close", action="store_true")
    parser.set_defaults(policy_reset_before=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_wrist_ready_navigation(args)
    run_policy(args)


if __name__ == "__main__":
    main()
