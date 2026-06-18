#!/usr/bin/env python3
"""Record orange grasp episodes with wrist + external camera observations."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from leader_follower_teleop import LeaderFollowerTeleop
from mark_one_bowl import YOLOSegMarker
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    DEFAULT_RECORD_TASK,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    LEADER_SPIKE_CONFIRM_FRAMES,
    LEADER_SPIKE_MAX_DELTA_TICKS,
    LOOP_INTERVAL_S,
    RESET_AFTER_POS,
    RESET_BEFORE_POS,
    parse_ids,
    parse_positions,
)
from record_orange_dataset import (
    create_or_resume_dataset,
    make_rebase_anchor,
    reset_follower,
    send_rebased_follower_positions,
    should_save_episode,
    wait_for_enter,
)
from record_orange_to_bowl_dual_camera_dataset import (
    add_dual_record_frame,
    make_dual_camera_features,
    open_camera,
    record_dual_reset_motion,
)


DEFAULT_DUAL_GRASP_DATASET_NAME = "orange_dual_camera_grasp_red_external_v1"
DEFAULT_DUAL_GRASP_REPO_ID = f"embody_car/{DEFAULT_DUAL_GRASP_DATASET_NAME}"
DEFAULT_DUAL_GRASP_TASK = DEFAULT_RECORD_TASK
RED_BGR = (0, 0, 255)


def record_dual_grasp_episode(
    *,
    dataset,
    teleop: LeaderFollowerTeleop,
    wrist_camera: cv2.VideoCapture,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
    reset_before_pos: dict[int, int],
    reset_after_pos: dict[int, int],
) -> None:
    if args.reset_before_episode:
        reset_follower(
            teleop,
            reset_pos=reset_before_pos,
            speed=args.reset_speed,
            acc=args.reset_acc,
            stage_delay_s=args.reset_stage_delay_s,
        )

    wait_for_enter(
        "\nPlace the orange for grasping, check wrist/external cameras, then press Enter.",
        enabled=not args.no_prompt,
    )

    if args.pre_record_delay_s > 0:
        print(f"Waiting {args.pre_record_delay_s:.1f}s before recording. Move your hand out of view.")
        time.sleep(args.pre_record_delay_s)

    frame_count = int(round(args.episode_time_s * args.fps))
    last_state: np.ndarray | None = None
    last_action: np.ndarray | None = None
    rebase_anchor: dict[int, int] = {}
    reset_recorded = False

    if args.rebase_leader_to_reset:
        leader_start = teleop._read_leader_positions()
        rebase_anchor = make_rebase_anchor(teleop, leader_start)
        teleop.last_sent.update({sid: reset_before_pos[sid] for sid in teleop.servo_ids if sid in reset_before_pos})
        print("Leader pose rebased to follower open reset for this grasp episode.")

    print(f"Recording dual-camera orange grasp episode: {frame_count} frames at {args.fps} fps.")
    frames_recorded = 0
    try:
        start = time.monotonic()
        for frame_index in range(frame_count):
            loop_start = time.monotonic()

            leader_positions = teleop._read_leader_positions()
            if args.rebase_leader_to_reset:
                send_rebased_follower_positions(
                    teleop,
                    leader_positions,
                    anchor=rebase_anchor,
                    reset_pos=reset_before_pos,
                )
            else:
                teleop._send_follower_positions(leader_positions)

            last_state, last_action, keep_recording = add_dual_record_frame(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                last_state=last_state,
                last_action=last_action,
                label="grasp",
                frame_index=frame_index + 1,
                frame_count=frame_count,
            )
            frames_recorded += 1

            if not keep_recording:
                print("Episode ended by user; choose whether to save it below.")
                break

            next_time = start + (frame_index + 1) / args.fps
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif args.verbose:
                print(f"Frame {frame_index}: loop overran by {-sleep_s:.3f}s")

            if args.verbose:
                print(
                    f"frame={frame_index + 1} "
                    f"loop={time.monotonic() - loop_start:.3f}s "
                    f"state={last_state.astype(int).tolist()} "
                    f"action={last_action.astype(int).tolist()} "
                    f"marker={getattr(args, 'last_marker_meta', None)}"
                )

        if args.reset_after_episode:
            added, last_state, last_action = record_dual_reset_motion(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                reset_pos=reset_after_pos,
                last_state=last_state,
                last_action=last_action,
            )
            frames_recorded += added
            reset_recorded = True
    finally:
        if frames_recorded > 0:
            if should_save_episode(
                enabled=args.confirm_save and not args.no_prompt,
                frames_recorded=frames_recorded,
            ):
                dataset.save_episode()
                print(f"Saved episode {dataset.num_episodes - 1}.")
            else:
                dataset.clear_episode_buffer()
                print("Discarded this episode.")
        else:
            print("No frames recorded; episode was not saved.")

        if args.reset_after_episode and not reset_recorded:
            reset_follower(
                teleop,
                reset_pos=reset_after_pos,
                speed=args.reset_speed,
                acc=args.reset_acc,
                stage_delay_s=args.reset_stage_delay_s,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record orange grasp episodes with wrist camera and red-marked external camera."
    )
    parser.add_argument("--task", default=DEFAULT_DUAL_GRASP_TASK)
    parser.add_argument("--repo-id", default=DEFAULT_DUAL_GRASP_REPO_ID)
    parser.add_argument("--root", default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--servo-ids", default="1,2,3,4,5,6")
    parser.add_argument("--speed", type=int, default=1800)
    parser.add_argument("--acc", type=int, default=45)
    parser.add_argument("--min-delta", type=int, default=6)
    parser.add_argument("--leader-max-jump", type=int, default=LEADER_SPIKE_MAX_DELTA_TICKS)
    parser.add_argument("--spike-confirm-frames", type=int, default=LEADER_SPIKE_CONFIRM_FRAMES)
    parser.add_argument("--target-max-step", type=int, default=FOLLOWER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--gripper-target-max-step", type=int, default=GRIPPER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--gripper-mode", choices=("linear",), default="linear")
    parser.add_argument("--gripper-threshold", type=int, default=-1)
    parser.add_argument("--state-max-jump", type=int, default=900)
    parser.add_argument("--state-range-margin", type=int, default=80)

    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-s", type=float, default=20.0)
    parser.add_argument("--vcodec", default="libsvtav1")

    parser.add_argument(
        "--reset-before-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in RESET_BEFORE_POS.items()),
        help="Follower reset ticks before recording. Defaults to open gripper.",
    )
    parser.add_argument(
        "--reset-after-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in RESET_AFTER_POS.items()),
        help="Follower reset ticks recorded after grasp. Defaults to holding the orange.",
    )
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument("--pre-record-delay-s", type=float, default=2.0)
    parser.add_argument("--no-reset-before-episode", dest="reset_before_episode", action="store_false")
    parser.add_argument("--no-reset-after-episode", dest="reset_after_episode", action="store_false")
    parser.add_argument("--no-rebase-leader-to-reset", dest="rebase_leader_to_reset", action="store_false")

    parser.add_argument("--no-wrist-rotate-180", action="store_true")
    parser.add_argument("--external-rotate-180", action="store_true")
    parser.add_argument("--marker-model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--marker-target", default="orange")
    parser.add_argument("--marker-device", default="cpu")
    parser.add_argument("--marker-conf", type=float, default=0.25)
    parser.add_argument("--marker-iou", type=float, default=0.7)
    parser.add_argument("--marker-alpha", type=float, default=0.20)
    parser.add_argument("--marker-thickness", type=int, default=6)
    parser.add_argument("--marker-mode", choices=("bbox", "mask", "both"), default="bbox")
    parser.add_argument(
        "--marker-choose",
        choices=("largest", "highest_conf", "nearest_center", "leftmost", "rightmost"),
        default="largest",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument(
        "--no-confirm-save",
        dest="confirm_save",
        action="store_false",
        help="Save every recorded episode automatically without asking.",
    )
    parser.add_argument("--debug-targets", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(
        reset_before_episode=True,
        reset_after_episode=True,
        rebase_leader_to_reset=True,
        confirm_save=True,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root) if args.root else WORKSPACE_DIR / "datasets" / DEFAULT_DUAL_GRASP_DATASET_NAME
    reset_before_pos = parse_positions(args.reset_before_pos)
    reset_after_pos = parse_positions(args.reset_after_pos)
    servo_ids = parse_ids(args.servo_ids)
    args.reset_order_groups = tuple(
        tuple(sid for sid in group if sid in servo_ids)
        for group in ((3, 4), (2,), (1, 5, 6))
    )

    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")

    args.external_marker = YOLOSegMarker(
        model=args.marker_model,
        device=args.marker_device,
        conf=args.marker_conf,
        iou=args.marker_iou,
        alpha=args.marker_alpha,
        thickness=args.marker_thickness,
        mode=args.marker_mode,
        choose=args.marker_choose,
        color_bgr=RED_BGR,
    )
    args.episode_target_bowl_choice = args.marker_choose
    args.episode_target_bowl_label = args.marker_target
    args.episode_marker_bbox = None
    args.last_marker_meta = None

    if args.overwrite and root.exists():
        shutil.rmtree(root)

    features = make_dual_camera_features(height=args.height, width=args.width)
    dataset = create_or_resume_dataset(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=features,
        resume=args.resume,
        vcodec=args.vcodec,
    )

    wrist_camera = open_camera(
        args.wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="wrist",
    )
    external_camera = open_camera(
        args.external_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="external",
    )

    teleop = LeaderFollowerTeleop(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        speed=args.speed,
        acc=args.acc,
        min_delta=args.min_delta,
        dry_run=False,
        direct_raw=False,
        servo_ids=servo_ids,
        debug_targets=args.debug_targets,
        debug_unchanged=False,
        gripper_mode=args.gripper_mode,
        gripper_threshold=args.gripper_threshold,
        leader_max_jump=args.leader_max_jump,
        spike_confirm_frames=args.spike_confirm_frames,
        target_max_step=args.target_max_step,
        gripper_target_max_step=args.gripper_target_max_step,
        loop_interval_s=LOOP_INTERVAL_S,
    )

    print(f"Recording dual-camera orange grasp dataset: repo_id={args.repo_id}, root={root}")
    print(f"Task: {args.task}")
    print(f"Reset before: {reset_before_pos}")
    print(f"Reset after: {reset_after_pos}")
    print(f"Wrist camera: {args.wrist_camera}, external camera: {args.external_camera}")
    print(
        "External red marker: "
        f"target={args.marker_target}, choose={args.marker_choose}, model={args.marker_model}"
    )
    print(f"Image size={args.width}x{args.height}, fps={args.fps}")
    print("Press q in the preview window to end an episode early.")

    try:
        for episode_idx in range(args.num_episodes):
            print(f"\n=== Dual-Camera Grasp Episode {episode_idx + 1}/{args.num_episodes} ===")
            record_dual_grasp_episode(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                reset_before_pos=reset_before_pos,
                reset_after_pos=reset_after_pos,
            )
    finally:
        teleop.close()
        wrist_camera.release()
        external_camera.release()
        if args.show:
            cv2.destroyAllWindows()
        dataset.finalize()
        dataset.stop_image_writer()
        print(f"Dataset finalized at {root}")


if __name__ == "__main__":
    main()
