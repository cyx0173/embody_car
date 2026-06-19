#!/usr/bin/env python3
"""Full orange pick-and-place: grasp orange, find bowl, then place into bowl."""

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
    WRIST_MIN_CONF,
    WRIST_TARGET_OFFSET_X_PX,
    WRIST_TARGET_OFFSET_Y_PX,
)
from orange_grasp_auto import build_default_args, run_policy as run_grasp_policy, run_wrist_ready_navigation
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_POLICY_PATH,
    DEFAULT_PLACE_POLICY_PATH,
    DEFAULT_TARGET,
    ORANGE_APPROACH_SPEED,
    ORANGE_FINAL_CLOSE_POS,
    ORANGE_PLACE_POLICY_DURATION_S,
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


def run_place_policy(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(BASE_DIR / "run_orange_to_bowl_policy.py"),
        "--policy-path",
        str(args.place_policy_path),
        "--follower-port",
        args.follower_port,
        "--wrist-camera",
        str(args.hand_camera_id),
        "--duration-s",
        str(args.place_policy_duration_s),
        "--fps",
        str(args.policy_fps),
        "--speed",
        str(args.policy_speed),
    ]
    if args.place_reset_hold_before:
        command.append("--reset-hold-before")
    if args.return_open_after_place:
        command.append("--return-open-after")
    if args.show:
        command.append("--show")
    if args.execute:
        command.append("--execute")

    print("Starting ACT place policy:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def make_nav_args(
    args: argparse.Namespace,
    *,
    target: str,
    wrist_target: str | None,
    policy_path: Path,
    reset_arm: bool,
    wrist_target_offset_x: float,
    wrist_target_offset_y: float,
    wrist_min_box_area: float,
    wrist_ready_min_overlap: float,
    wrist_ready_position_mode: str,
    wrist_preferred_box_area: float,
    wrist_preferred_timeout_s: float,
    wrist_min_box_visible_ratio: float,
    wrist_visible_margin_px: float,
    wrist_max_box_area: float,
    wrist_visual_servo_mode: str,
    policy_reset_before: bool,
) -> argparse.Namespace:
    return build_default_args(
        target=target,
        wrist_target=wrist_target,
        model=args.model,
        policy_path=policy_path,
        timeout=args.timeout,
        show=args.show,
        execute=args.execute,
        hand_camera_id=args.hand_camera_id,
        base_camera_id=args.base_camera_id,
        follower_port=args.follower_port,
        approach_speed=args.approach_speed,
        wheel_search_speed=args.wheel_search_speed,
        wrist_align_max_speed=args.wrist_align_max_speed,
        wrist_align_min_speed=args.wrist_align_min_speed,
        nav_frame_mode=args.nav_frame_mode,
        nav_frame_width=args.nav_frame_width,
        nav_frame_height=args.nav_frame_height,
        wrist_target_offset_x=wrist_target_offset_x,
        wrist_target_offset_y=wrist_target_offset_y,
        wrist_ready_deadzone_x=args.wrist_ready_deadzone_x,
        wrist_ready_deadzone_y=args.wrist_ready_deadzone_y,
        wrist_ready_min_overlap=wrist_ready_min_overlap,
        wrist_ready_position_mode=wrist_ready_position_mode,
        wrist_preferred_box_area=wrist_preferred_box_area,
        wrist_preferred_timeout_s=wrist_preferred_timeout_s,
        wrist_min_box_visible_ratio=wrist_min_box_visible_ratio,
        wrist_visible_margin_px=wrist_visible_margin_px,
        wrist_max_box_area=wrist_max_box_area,
        wrist_min_conf=args.wrist_min_conf,
        wrist_min_box_area=wrist_min_box_area,
        wrist_visual_servo_mode=wrist_visual_servo_mode,
        policy_duration_s=args.grasp_policy_duration_s,
        policy_fps=args.policy_fps,
        policy_speed=args.policy_speed,
        policy_reset_before=policy_reset_before,
        pre_close_servo4_offset=args.pre_close_servo4_offset,
        pre_close_gripper_threshold=args.pre_close_gripper_threshold,
        pre_close_hold_s=args.pre_close_hold_s,
        final_close_pos=args.final_close_pos,
        return_after_close=False,
        nav_reset_arm=reset_arm,
    )


def orange_pick_and_place(args: argparse.Namespace) -> None:
    print("\n=== Stage 1/3: find orange and grasp ===")
    orange_args = make_nav_args(
        args,
        target=args.orange_target,
        wrist_target=args.orange_wrist_target,
        policy_path=args.grasp_policy_path,
        reset_arm=True,
        wrist_target_offset_x=args.orange_wrist_target_offset_x,
        wrist_target_offset_y=args.orange_wrist_target_offset_y,
        wrist_min_box_area=args.orange_wrist_min_box_area,
        wrist_ready_min_overlap=args.orange_wrist_ready_min_overlap,
        wrist_ready_position_mode="center_or_overlap",
        wrist_preferred_box_area=0.0,
        wrist_preferred_timeout_s=0.0,
        wrist_min_box_visible_ratio=0.0,
        wrist_visible_margin_px=0.0,
        wrist_max_box_area=0.0,
        wrist_visual_servo_mode="sequential",
        policy_reset_before=True,
    )
    run_wrist_ready_navigation(orange_args)
    run_grasp_policy(orange_args)

    print("\n=== Stage 2/3: find bowl and move it into wrist-ready view ===")
    bowl_args = make_nav_args(
        args,
        target=args.bowl_target,
        wrist_target=args.bowl_wrist_target,
        policy_path=args.grasp_policy_path,
        reset_arm=False,
        wrist_target_offset_x=args.bowl_wrist_target_offset_x,
        wrist_target_offset_y=args.bowl_wrist_target_offset_y,
        wrist_min_box_area=args.bowl_wrist_min_box_area,
        wrist_ready_min_overlap=args.bowl_wrist_ready_min_overlap,
        wrist_ready_position_mode=args.bowl_wrist_ready_position_mode,
        wrist_preferred_box_area=args.bowl_wrist_preferred_box_area,
        wrist_preferred_timeout_s=args.bowl_wrist_preferred_timeout_s,
        wrist_min_box_visible_ratio=args.bowl_wrist_min_box_visible_ratio,
        wrist_visible_margin_px=args.bowl_wrist_visible_margin_px,
        wrist_max_box_area=args.bowl_wrist_max_box_area,
        wrist_visual_servo_mode=args.bowl_wrist_visual_servo_mode,
        policy_reset_before=False,
    )
    run_wrist_ready_navigation(bowl_args)

    print("\n=== Stage 3/3: place orange into bowl ===")
    run_place_policy(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full orange pick-and-place: grasp orange, align to bowl, then place orange into bowl."
    )
    parser.add_argument("--orange-target", default=DEFAULT_TARGET)
    parser.add_argument("--orange-wrist-target", default=None)
    parser.add_argument("--bowl-target", default="bowl")
    parser.add_argument(
        "--bowl-wrist-target",
        default="cup",
        help="YOLO class used by wrist camera for bowl-ready detection. Your wrist camera often sees the bowl as cup.",
    )
    parser.add_argument("--model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--grasp-policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--place-policy-path", type=Path, default=DEFAULT_PLACE_POLICY_PATH)
    parser.add_argument("--timeout", type=float, default=BASE_SEARCH_TIMEOUT_S)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move wheels and follower arm.")
    parser.add_argument("--hand-camera-id", type=int, default=2)
    parser.add_argument("--base-camera-id", type=int, default=1)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)

    parser.add_argument("--approach-speed", type=int, default=ORANGE_APPROACH_SPEED)
    parser.add_argument("--wheel-search-speed", type=int, default=ORANGE_WHEEL_SEARCH_SPEED)
    parser.add_argument("--wrist-align-max-speed", type=int, default=ORANGE_WRIST_ALIGN_MAX_SPEED)
    parser.add_argument("--wrist-align-min-speed", type=int, default=ORANGE_WRIST_ALIGN_MIN_SPEED)
    parser.add_argument("--nav-frame-mode", choices=("crop", "letterbox"), default=FRAME_MODE_LETTERBOX)
    parser.add_argument("--nav-frame-width", type=int, default=NAV_FRAME_WIDTH)
    parser.add_argument("--nav-frame-height", type=int, default=NAV_FRAME_HEIGHT)
    parser.add_argument("--wrist-ready-deadzone-x", type=float, default=70.0)
    parser.add_argument("--wrist-ready-deadzone-y", type=float, default=70.0)
    parser.add_argument("--wrist-min-conf", type=float, default=WRIST_MIN_CONF)

    parser.add_argument("--orange-wrist-target-offset-x", type=float, default=WRIST_TARGET_OFFSET_X_PX)
    parser.add_argument("--orange-wrist-target-offset-y", type=float, default=WRIST_TARGET_OFFSET_Y_PX)
    parser.add_argument("--orange-wrist-min-box-area", type=float, default=ORANGE_WRIST_MIN_BOX_AREA_RATIO)
    parser.add_argument("--orange-wrist-ready-min-overlap", type=float, default=ORANGE_WRIST_READY_MIN_BOX_OVERLAP_RATIO)

    parser.add_argument("--bowl-wrist-target-offset-x", type=float, default=0.0)
    parser.add_argument("--bowl-wrist-target-offset-y", type=float, default=0.0)
    parser.add_argument("--bowl-wrist-min-box-area", type=float, default=0.08)
    parser.add_argument("--bowl-wrist-ready-min-overlap", type=float, default=0.25)
    parser.add_argument(
        "--bowl-wrist-ready-position-mode",
        choices=("center_or_overlap", "center", "overlap"),
        default="center",
        help="For bowl placement, require the detected bowl/cup center to reach the target by default.",
    )
    parser.add_argument(
        "--bowl-wrist-visual-servo-mode",
        choices=("sequential", "proportional"),
        default="proportional",
        help="Wheel visual servo mode used after the bowl/cup appears in the wrist camera.",
    )
    parser.add_argument(
        "--bowl-wrist-min-box-visible-ratio",
        type=float,
        default=0.97,
        help="Require nearly the whole wrist bowl/cup bbox to stay inside the safe view box.",
    )
    parser.add_argument(
        "--bowl-wrist-visible-margin-px",
        type=float,
        default=24.0,
        help="Safe margin from wrist image edges when checking whether the bowl/cup is fully visible.",
    )
    parser.add_argument(
        "--bowl-wrist-max-box-area",
        type=float,
        default=0.32,
        help="Back up when the bowl/cup bbox is too large in the wrist image. 0 disables it.",
    )
    parser.add_argument(
        "--bowl-wrist-preferred-box-area",
        type=float,
        default=0.14,
        help="Soft bowl approach target. Keep moving closer until the wrist bbox area reaches this value or timeout.",
    )
    parser.add_argument(
        "--bowl-wrist-preferred-timeout-s",
        type=float,
        default=4.0,
        help="Maximum time to chase the preferred bowl area after normal wrist-ready is reached.",
    )

    parser.add_argument("--grasp-policy-duration-s", type=float, default=ORANGE_POLICY_DURATION_S)
    parser.add_argument("--place-policy-duration-s", type=float, default=ORANGE_PLACE_POLICY_DURATION_S)
    parser.add_argument("--policy-fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--policy-speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--pre-close-servo4-offset", type=int, default=ORANGE_PRE_CLOSE_SERVO4_OFFSET)
    parser.add_argument("--pre-close-gripper-threshold", type=int, default=ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD)
    parser.add_argument("--pre-close-hold-s", type=float, default=ORANGE_PRE_CLOSE_HOLD_S)
    parser.add_argument("--final-close-pos", type=int, default=ORANGE_FINAL_CLOSE_POS)
    parser.add_argument(
        "--place-reset-hold-before",
        action="store_true",
        help="Reset to the place holding pose before place policy. Usually leave this off in the full pipeline.",
    )
    parser.add_argument("--return-open-after-place", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    orange_pick_and_place(args)


if __name__ == "__main__":
    main()
