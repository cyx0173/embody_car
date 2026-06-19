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
    DEFAULT_DUAL_PLACE_DATASET_NAME,
    DEFAULT_DUAL_PLACE_REPO_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    DEFAULT_PLACE_TASK,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    JOINT_NAMES,
    LEADER_SPIKE_CONFIRM_FRAMES,
    LEADER_SPIKE_MAX_DELTA_TICKS,
    LOOP_INTERVAL_S,
    PLACE_RESET_HOLD_POS,
    PLACE_RESET_OPEN_POS,
    clamp_follower_target,
    parse_ids,
    parse_positions,
    positions_to_array,
    sanitize_follower_positions,
)
from record_orange_dataset import (
    create_or_resume_dataset,
    reset_follower,
    should_save_episode,
    wait_for_enter,
)
from record_orange_to_bowl_dataset import (
    make_place_rebase_anchor,
    send_place_rebased_follower_positions,
)


DEFAULT_MARKED_BOWL_DATASET_NAME = "orange_to_bowl_marked_dual_camera_place_v1"
DEFAULT_MARKED_BOWL_REPO_ID = f"embody_car/{DEFAULT_MARKED_BOWL_DATASET_NAME}"
DEFAULT_MARKED_BOWL_TASK = "place the orange into the marked bowl"


def make_dual_camera_features(*, height: int, width: int) -> dict:
    image_feature = {
        "dtype": "video",
        "shape": (3, height, width),
        "names": ["channels", "height", "width"],
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "observation.images.wrist": dict(image_feature),
        "observation.images.external": dict(image_feature),
    }


def open_camera(
    camera_index: int,
    *,
    width: int,
    height: int,
    fps: int,
    name: str,
) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open {name} camera index {camera_index}.")
    return camera


def read_camera_frame(
    camera: cv2.VideoCapture,
    *,
    name: str,
    width: int,
    height: int,
    rotate_180: bool,
) -> np.ndarray:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read {name} camera frame.")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_dual_camera_frames(
    *,
    wrist_camera: cv2.VideoCapture,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    wrist_image = read_camera_frame(
        wrist_camera,
        name="wrist",
        width=args.width,
        height=args.height,
        rotate_180=not args.no_wrist_rotate_180,
    )
    external_image = read_camera_frame(
        external_camera,
        name="external",
        width=args.width,
        height=args.height,
        rotate_180=args.external_rotate_180,
    )
    marker = getattr(args, "external_marker", None)
    target_bowl_choice = getattr(args, "episode_target_bowl_choice", None)
    if marker is not None and target_bowl_choice is not None:
        external_bgr = cv2.cvtColor(external_image, cv2.COLOR_RGB2BGR)
        bbox = getattr(args, "episode_marker_bbox", None)
        if bbox is not None:
            marked_bgr = marker.draw_bbox(external_bgr, bbox)
        else:
            marker.choose = target_bowl_choice
            marked_bgr, meta = marker.infer(
                external_bgr,
                args.marker_target,
                return_meta=True,
                mode=args.marker_mode,
            )
            args.last_marker_meta = meta
        external_image = cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)
    return wrist_image, external_image


def capture_episode_marker_bbox(
    *,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
) -> tuple[int, int, int, int] | None:
    marker = getattr(args, "external_marker", None)
    target_bowl_choice = getattr(args, "episode_target_bowl_choice", None)
    if marker is None or target_bowl_choice is None:
        return None

    external_image = read_camera_frame(
        external_camera,
        name="external",
        width=args.width,
        height=args.height,
        rotate_180=args.external_rotate_180,
    )
    external_bgr = cv2.cvtColor(external_image, cv2.COLOR_RGB2BGR)
    marker.choose = target_bowl_choice
    marked_bgr, meta = marker.infer(
        external_bgr,
        args.marker_target,
        return_meta=True,
        mode=args.marker_mode,
    )
    args.last_marker_meta = meta
    bbox = meta.get("bbox_xyxy")
    if not meta.get("found") or bbox is None:
        print(f"WARNING: target bowl marker not found: {meta}")
        return None
    print(
        "Marked target bowl: "
        f"{args.episode_target_bowl_label}, "
        f"class={meta.get('class_name')}, "
        f"conf={meta.get('confidence'):.2f}, "
        f"bbox={bbox}"
    )
    if args.show:
        cv2.imshow("selected_target_bowl_marker", marked_bgr)
        cv2.waitKey(300)
    return tuple(int(v) for v in bbox)


def show_dual_preview(
    *,
    wrist_image: np.ndarray,
    external_image: np.ndarray,
    label: str,
    frame_index: int,
    frame_count: int,
) -> bool:
    wrist_preview = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
    external_preview = cv2.cvtColor(external_image, cv2.COLOR_RGB2BGR)
    cv2.putText(
        wrist_preview,
        f"wrist {label} {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        external_preview,
        f"external {label} {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    preview = cv2.hconcat([wrist_preview, external_preview])
    cv2.imshow("orange_to_bowl_dual_camera_record", preview)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def add_dual_record_frame(
    *,
    dataset,
    teleop: LeaderFollowerTeleop,
    wrist_camera: cv2.VideoCapture,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
    last_state: np.ndarray | None,
    last_action: np.ndarray | None,
    label: str,
    frame_index: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    wrist_image, external_image = read_dual_camera_frames(
        wrist_camera=wrist_camera,
        external_camera=external_camera,
        args=args,
    )
    follower_positions = sanitize_follower_positions(
        teleop._read_follower_positions(),
        last_state=last_state,
        max_jump=args.state_max_jump,
        range_margin=args.state_range_margin,
    )
    action = positions_to_array(teleop.last_sent, fallback=last_action)
    state = positions_to_array(
        follower_positions,
        fallback=last_state if last_state is not None else action,
    )
    dataset.add_frame(
        {
            "observation.state": state,
            "action": action,
            "observation.images.wrist": wrist_image,
            "observation.images.external": external_image,
            "task": args.task,
        }
    )

    keep_recording = True
    if args.show:
        keep_recording = show_dual_preview(
            wrist_image=wrist_image,
            external_image=external_image,
            label=label,
            frame_index=frame_index,
            frame_count=frame_count,
        )
    return state, action, keep_recording


def record_dual_reset_motion(
    *,
    dataset,
    teleop: LeaderFollowerTeleop,
    wrist_camera: cv2.VideoCapture,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
    reset_pos: dict[int, int],
    last_state: np.ndarray | None,
    last_action: np.ndarray | None,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if teleop.follower is None:
        return 0, last_state, last_action

    print(f"Recording reset motion to {reset_pos}")
    total_frames = max(1, int(round(args.reset_stage_delay_s * args.fps))) * len(args.reset_order_groups)
    recorded = 0
    frame_number = 0
    next_time = time.monotonic()

    for group in args.reset_order_groups:
        for servo_id in group:
            if servo_id not in teleop.servo_ids:
                continue
            target = reset_pos.get(servo_id)
            if target is None:
                continue
            teleop.follower.move_to(servo_id, target, speed=args.reset_speed, acc=args.reset_acc)
            teleop.last_sent[servo_id] = target

        stage_frames = max(1, int(round(args.reset_stage_delay_s * args.fps)))
        for _ in range(stage_frames):
            frame_number += 1
            last_state, last_action, keep_recording = add_dual_record_frame(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                last_state=last_state,
                last_action=last_action,
                label="reset-open",
                frame_index=frame_number,
                frame_count=total_frames,
            )
            recorded += 1
            if not keep_recording:
                return recorded, last_state, last_action

            next_time += 1.0 / args.fps
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    return recorded, last_state, last_action


def record_dual_place_episode(
    *,
    dataset,
    teleop: LeaderFollowerTeleop,
    wrist_camera: cv2.VideoCapture,
    external_camera: cv2.VideoCapture,
    args: argparse.Namespace,
    reset_open_pos: dict[int, int],
    reset_hold_pos: dict[int, int],
) -> None:
    if args.reset_before_episode:
        reset_follower(
            teleop,
            reset_pos=reset_open_pos,
            speed=args.reset_speed,
            acc=args.reset_acc,
            stage_delay_s=args.reset_stage_delay_s,
        )

    if getattr(args, "mark_external_bowl", False):
        print(f"Target bowl marker: {args.episode_target_bowl_label}")

    wait_for_enter(
        "\nPlace the orange in the gripper, place the bowl, check both cameras, then press Enter.",
        enabled=not args.no_prompt,
    )
    if getattr(args, "mark_external_bowl", False):
        args.episode_marker_bbox = capture_episode_marker_bbox(
            external_camera=external_camera,
            args=args,
        )

    if args.close_gripper_before_record:
        reset_follower(
            teleop,
            reset_pos=reset_hold_pos,
            speed=args.reset_speed,
            acc=args.reset_acc,
            stage_delay_s=args.reset_stage_delay_s,
        )
        time.sleep(args.hold_settle_s)

    if args.pre_record_delay_s > 0:
        print(f"Waiting {args.pre_record_delay_s:.1f}s before recording. Move your hand out of view.")
        time.sleep(args.pre_record_delay_s)

    frame_count = int(round(args.episode_time_s * args.fps))
    last_state: np.ndarray | None = None
    last_action: np.ndarray | None = None
    rebase_anchor: dict[int, int] = {}

    if args.rebase_leader_to_reset:
        leader_start = teleop._read_leader_positions()
        rebase_anchor = make_place_rebase_anchor(teleop, leader_start)
        teleop.last_sent.update({sid: reset_hold_pos[sid] for sid in teleop.servo_ids if sid in reset_hold_pos})
        print("Leader pose rebased to follower holding reset for this episode, including gripper.")

    print(f"Recording dual-camera place episode: {frame_count} frames at {args.fps} fps.")
    frames_recorded = 0
    reset_recorded = False
    try:
        start = time.monotonic()
        for frame_index in range(frame_count):
            loop_start = time.monotonic()

            leader_positions = teleop._read_leader_positions()
            if args.rebase_leader_to_reset:
                send_place_rebased_follower_positions(
                    teleop,
                    leader_positions,
                    anchor=rebase_anchor,
                    reset_pos=reset_hold_pos,
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
                label="place",
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
                    f"action={last_action.astype(int).tolist()}"
                )

        if args.reset_after_episode:
            added, last_state, last_action = record_dual_reset_motion(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                reset_pos=reset_open_pos,
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
                reset_pos=reset_open_pos,
                speed=args.reset_speed,
                acc=args.reset_acc,
                stage_delay_s=args.reset_stage_delay_s,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record dual-camera episodes for placing a held orange into a bowl."
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--repo-id", default=None)
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
        "--reset-open-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in PLACE_RESET_OPEN_POS.items()),
        help="Follower reset ticks with gripper open, used before loading and after each episode.",
    )
    parser.add_argument(
        "--reset-hold-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in PLACE_RESET_HOLD_POS.items()),
        help="Follower reset ticks with gripper holding the orange, used as the recording start pose.",
    )
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument("--hold-settle-s", type=float, default=0.4)
    parser.add_argument("--pre-record-delay-s", type=float, default=2.0)
    parser.add_argument("--no-reset-before-episode", dest="reset_before_episode", action="store_false")
    parser.add_argument("--no-reset-after-episode", dest="reset_after_episode", action="store_false")
    parser.add_argument("--no-close-gripper-before-record", dest="close_gripper_before_record", action="store_false")
    parser.add_argument("--no-rebase-leader-to-reset", dest="rebase_leader_to_reset", action="store_false")
    parser.add_argument("--no-wrist-rotate-180", action="store_true")
    parser.add_argument("--external-rotate-180", action="store_true")
    parser.add_argument(
        "--mark-external-bowl",
        action="store_true",
        help="Draw a blue marker on the selected target bowl in the external camera observation.",
    )
    parser.add_argument("--marker-model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--marker-target", default="bowl")
    parser.add_argument("--marker-device", default="cpu")
    parser.add_argument("--marker-conf", type=float, default=0.25)
    parser.add_argument("--marker-iou", type=float, default=0.7)
    parser.add_argument("--marker-alpha", type=float, default=0.20)
    parser.add_argument("--marker-thickness", type=int, default=6)
    parser.add_argument("--marker-mode", choices=("bbox", "mask", "both"), default="bbox")
    parser.add_argument(
        "--target-bowl",
        choices=("prompt", "left", "right", "alternate-left-first", "alternate-right-first"),
        default="prompt",
        help="Which bowl to mark in the external camera for each episode.",
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
        close_gripper_before_record=True,
        rebase_leader_to_reset=True,
        confirm_save=True,
    )
    return parser


def resolve_target_bowl_choice(args: argparse.Namespace, episode_idx: int) -> tuple[str | None, str | None]:
    if not args.mark_external_bowl:
        return None, None

    choice = args.target_bowl
    if choice == "prompt":
        if args.no_prompt:
            raise ValueError("--target-bowl prompt cannot be used together with --no-prompt.")
        while True:
            raw = input("\nSelect target bowl for this episode [l/r]: ").strip().lower()
            if raw in {"l", "left", "leftmost"}:
                return "leftmost", "left"
            if raw in {"r", "right", "rightmost"}:
                return "rightmost", "right"
            print("Please enter l or r.")
    if choice == "left":
        return "leftmost", "left"
    if choice == "right":
        return "rightmost", "right"
    if choice == "alternate-left-first":
        return ("leftmost", "left") if episode_idx % 2 == 0 else ("rightmost", "right")
    if choice == "alternate-right-first":
        return ("rightmost", "right") if episode_idx % 2 == 0 else ("leftmost", "left")
    raise ValueError(f"Unsupported target bowl choice: {choice}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.task is None:
        args.task = DEFAULT_MARKED_BOWL_TASK if args.mark_external_bowl else DEFAULT_PLACE_TASK
    if args.repo_id is None:
        args.repo_id = DEFAULT_MARKED_BOWL_REPO_ID if args.mark_external_bowl else DEFAULT_DUAL_PLACE_REPO_ID

    default_root_name = DEFAULT_MARKED_BOWL_DATASET_NAME if args.mark_external_bowl else DEFAULT_DUAL_PLACE_DATASET_NAME
    root = Path(args.root) if args.root else WORKSPACE_DIR / "datasets" / default_root_name
    reset_open_pos = parse_positions(args.reset_open_pos)
    reset_hold_pos = parse_positions(args.reset_hold_pos)
    servo_ids = parse_ids(args.servo_ids)
    args.reset_order_groups = tuple(
        tuple(sid for sid in group if sid in servo_ids)
        for group in ((3, 4), (2,), (1, 5, 6))
    )

    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")

    args.external_marker = None
    args.episode_target_bowl_choice = None
    args.episode_target_bowl_label = None
    args.episode_marker_bbox = None
    args.last_marker_meta = None
    if args.mark_external_bowl:
        args.external_marker = YOLOSegMarker(
            model=args.marker_model,
            device=args.marker_device,
            conf=args.marker_conf,
            iou=args.marker_iou,
            alpha=args.marker_alpha,
            thickness=args.marker_thickness,
            mode=args.marker_mode,
            choose="leftmost",
        )

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

    print(f"Recording dual-camera place dataset: repo_id={args.repo_id}, root={root}")
    print(f"Task: {args.task}")
    print(f"Reset open: {reset_open_pos}")
    print(f"Reset hold: {reset_hold_pos}")
    print(f"Wrist camera: {args.wrist_camera}, external camera: {args.external_camera}")
    if args.mark_external_bowl:
        print(
            "External marker: "
            f"target={args.marker_target}, model={args.marker_model}, "
            f"choice={args.target_bowl}"
        )
    print(f"Image size={args.width}x{args.height}, fps={args.fps}")
    print("Press q in the preview window to end an episode early.")

    try:
        for episode_idx in range(args.num_episodes):
            print(f"\n=== Dual-Camera Place Episode {episode_idx + 1}/{args.num_episodes} ===")
            choice, label = resolve_target_bowl_choice(args, episode_idx)
            args.episode_target_bowl_choice = choice
            args.episode_target_bowl_label = label
            args.episode_marker_bbox = None
            record_dual_place_episode(
                dataset=dataset,
                teleop=teleop,
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
                reset_open_pos=reset_open_pos,
                reset_hold_pos=reset_hold_pos,
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
