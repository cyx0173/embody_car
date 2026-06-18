from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from leader_follower_teleop import LeaderFollowerTeleop
from orange_grasp_config import (
    DEFAULT_RECORD_DATASET_NAME,
    DEFAULT_RECORD_REPO_ID,
    DEFAULT_RECORD_TASK,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    JOINT_MAP,
    JOINT_NAMES,
    LEADER_SPIKE_CONFIRM_FRAMES,
    LEADER_SPIKE_MAX_DELTA_TICKS,
    LOOP_INTERVAL_S,
    RESET_AFTER_POS,
    RESET_BEFORE_POS,
    RESET_ORDER_GROUPS,
    clamp_follower_target,
    parse_positions,
    parse_ids,
    positions_to_array,
    sanitize_follower_positions,
)


def make_features(*, height: int, width: int) -> dict:
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
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
    }


def read_camera_frame(
    camera: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    rotate_180: bool,
) -> np.ndarray:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read wrist camera frame.")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def reset_follower(
    teleop: LeaderFollowerTeleop,
    *,
    reset_pos: dict[int, int],
    speed: int,
    acc: int,
    stage_delay_s: float,
) -> None:
    if teleop.follower is None:
        return

    print(f"Resetting follower to {reset_pos}")
    for group in RESET_ORDER_GROUPS:
        for servo_id in group:
            if servo_id not in teleop.servo_ids:
                continue
            target = reset_pos.get(servo_id)
            if target is None:
                continue
            teleop.follower.move_to(servo_id, target, speed=speed, acc=acc)
            teleop.last_sent[servo_id] = target
        time.sleep(stage_delay_s)


def make_rebase_anchor(
    teleop: LeaderFollowerTeleop,
    leader_positions: dict[int, int],
) -> dict[int, int]:
    anchor: dict[int, int] = {}
    for servo_id, leader_pos in leader_positions.items():
        if servo_id == 6:
            continue
        anchor[servo_id] = teleop._clamp_follower_target(servo_id, leader_pos)
    return anchor


def send_rebased_follower_positions(
    teleop: LeaderFollowerTeleop,
    leader_positions: dict[int, int],
    *,
    anchor: dict[int, int],
    reset_pos: dict[int, int],
) -> None:
    follower_targets: dict[int, int] = {}
    for servo_id, leader_pos in leader_positions.items():
        if servo_id == 6:
            follower_targets[servo_id] = teleop._clamp_follower_target(servo_id, leader_pos)
            continue

        baseline = anchor.get(servo_id)
        if baseline is None:
            baseline = teleop._clamp_follower_target(servo_id, leader_pos)
            anchor[servo_id] = baseline

        absolute_target = teleop._clamp_follower_target(servo_id, leader_pos)
        rebased_target = reset_pos.get(servo_id, absolute_target) + (absolute_target - baseline)
        follower_targets[servo_id] = clamp_follower_target(servo_id, rebased_target)

    for servo_id, target in follower_targets.items():
        target = teleop._limit_target_step(servo_id, target)
        last = teleop.last_sent.get(servo_id)
        if last is not None and abs(target - last) < teleop.min_delta:
            continue
        teleop.last_sent[servo_id] = target
        if teleop.follower is not None:
            teleop.follower.move_to(servo_id, target, speed=teleop.speed, acc=teleop.acc)


def add_record_frame(
    *,
    dataset: LeRobotDataset,
    teleop: LeaderFollowerTeleop,
    camera: cv2.VideoCapture,
    args: argparse.Namespace,
    last_state: np.ndarray | None,
    last_action: np.ndarray | None,
    label: str,
    frame_index: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = read_camera_frame(
        camera,
        width=args.width,
        height=args.height,
        rotate_180=not args.no_rotate_180,
    )
    follower_positions = sanitize_follower_positions(
        teleop._read_follower_positions(),
        last_state=last_state,
        max_jump=args.state_max_jump,
        range_margin=args.state_range_margin,
    )
    action = positions_to_array(teleop.last_sent, fallback=last_action)
    state = positions_to_array(follower_positions, fallback=last_state if last_state is not None else action)
    dataset.add_frame(
        {
            "observation.state": state,
            "action": action,
            "observation.images.wrist": image,
            "task": args.task,
        }
    )

    if args.show:
        preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(
            preview,
            f"{label} frame {frame_index}/{frame_count}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("custom_lerobot_wrist_record", preview)
        cv2.waitKey(1)

    return state, action


def record_reset_motion(
    *,
    dataset: LeRobotDataset,
    teleop: LeaderFollowerTeleop,
    camera: cv2.VideoCapture,
    args: argparse.Namespace,
    reset_pos: dict[int, int],
    last_state: np.ndarray | None,
    last_action: np.ndarray | None,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if teleop.follower is None:
        return 0, last_state, last_action

    print(f"Recording reset motion to {reset_pos}")
    total_frames = max(1, int(round(args.reset_stage_delay_s * args.fps))) * len(RESET_ORDER_GROUPS)
    recorded = 0
    frame_number = 0
    next_time = time.monotonic()

    for group in RESET_ORDER_GROUPS:
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
            last_state, last_action = add_record_frame(
                dataset=dataset,
                teleop=teleop,
                camera=camera,
                args=args,
                last_state=last_state,
                last_action=last_action,
                label="reset-hold",
                frame_index=frame_number,
                frame_count=total_frames,
            )
            recorded += 1
            next_time += 1.0 / args.fps
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    return recorded, last_state, last_action


def create_or_resume_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
    features: dict,
    resume: bool,
    vcodec: str,
) -> LeRobotDataset:
    if root.exists() and not resume:
        raise FileExistsError(
            f"Dataset root already exists: {root}\n"
            "Use --resume to append episodes, or --overwrite to start over."
        )
    if root.exists() and resume:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Dataset root exists but is missing {info_path}.\n"
                "This usually means a previous recording was interrupted before the dataset was finalized, "
                "or --root points to the wrong directory.\n"
                "Use --overwrite to start a fresh dataset, or pass the correct --root to an existing dataset."
            )
        return LeRobotDataset(
            repo_id,
            root=root,
            batch_encoding_size=1,
            vcodec=vcodec,
        )
    return LeRobotDataset.create(
        repo_id,
        fps,
        root=root,
        robot_type="custom_so101_follower",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=vcodec,
    )


def wait_for_enter(prompt: str, *, enabled: bool) -> None:
    if not enabled:
        print(prompt)
        return
    try:
        input(prompt)
    except EOFError as exc:
        raise RuntimeError(
            "Recorder needs an interactive terminal between episodes. "
            "Run it from your terminal or pass --no-prompt intentionally."
        ) from exc


def should_save_episode(*, enabled: bool, frames_recorded: int) -> bool:
    if frames_recorded <= 0:
        return False
    if not enabled:
        return True

    while True:
        try:
            answer = input(
                f"Save this episode with {frames_recorded} frames? [Y/n] "
            ).strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                "Recorder needs an interactive terminal to confirm episode saving. "
                "Run it from your terminal or pass --no-confirm-save intentionally."
            ) from exc

        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter y or n.")


def record_episode(
    *,
    dataset: LeRobotDataset,
    teleop: LeaderFollowerTeleop,
    camera: cv2.VideoCapture,
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
        "\nPlace the orange in the wrist view, hold the leader arm comfortably, then press Enter.",
        enabled=not args.no_prompt,
    )

    frame_count = int(round(args.episode_time_s * args.fps))
    last_state: np.ndarray | None = None
    last_action: np.ndarray | None = None
    rebase_anchor: dict[int, int] = {}

    if args.rebase_leader_to_reset:
        leader_start = teleop._read_leader_positions()
        rebase_anchor = make_rebase_anchor(teleop, leader_start)
        teleop.last_sent.update({sid: reset_before_pos[sid] for sid in teleop.servo_ids if sid in reset_before_pos})
        print("Leader pose rebased to follower reset for this episode.")

    print(f"Recording {frame_count} frames at {args.fps} fps.")
    frames_recorded = 0
    reset_recorded = False
    try:
        start = time.monotonic()
        for frame_index in range(frame_count):
            loop_start = time.monotonic()

            image = read_camera_frame(
                camera,
                width=args.width,
                height=args.height,
                rotate_180=not args.no_rotate_180,
            )

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
            follower_positions = sanitize_follower_positions(
                teleop._read_follower_positions(),
                last_state=last_state,
                max_jump=args.state_max_jump,
                range_margin=args.state_range_margin,
            )
            action = positions_to_array(teleop.last_sent, fallback=last_action)
            state = positions_to_array(follower_positions, fallback=last_state if last_state is not None else action)
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "observation.images.wrist": image,
                    "task": args.task,
                }
            )
            frames_recorded += 1

            last_state = state
            last_action = action

            if args.show:
                preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    preview,
                    f"recording frame {frame_index + 1}/{frame_count}",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("custom_lerobot_wrist_record", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
                    f"state={state.astype(int).tolist()} "
                    f"action={action.astype(int).tolist()}"
                )

        if args.reset_after_episode:
            added, last_state, last_action = record_reset_motion(
                dataset=dataset,
                teleop=teleop,
                camera=camera,
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
        description="Record orange wrist-ready grasp episodes as a LeRobotDataset using local teleop code."
    )
    parser.add_argument("--target", default="orange")
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
    parser.add_argument("--gripper-mode", choices=("binary", "linear"), default="linear")
    parser.add_argument("--gripper-threshold", type=int, default=-1)
    parser.add_argument(
        "--state-max-jump",
        type=int,
        default=900,
        help="Reject follower state readings that jump more than this many ticks from the previous saved state.",
    )
    parser.add_argument(
        "--state-range-margin",
        type=int,
        default=80,
        help="Allow this many ticks outside the calibrated follower range before rejecting a state reading.",
    )

    parser.add_argument("--wrist-camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-s", type=float, default=20.0)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument(
        "--reset-pos",
        default=None,
        help="Legacy alias: use the same reset ticks before and after the episode.",
    )
    parser.add_argument(
        "--reset-before-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in RESET_BEFORE_POS.items()),
        help="Follower reset ticks before recording. Defaults to open gripper.",
    )
    parser.add_argument(
        "--reset-after-pos",
        default=",".join(f"{sid}:{pos}" for sid, pos in RESET_AFTER_POS.items()),
        help="Follower reset ticks recorded after grasp. Defaults to closed gripper.",
    )
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument("--no-reset-before-episode", dest="reset_before_episode", action="store_false")
    parser.add_argument("--no-reset-after-episode", dest="reset_after_episode", action="store_false")
    parser.add_argument("--no-rebase-leader-to-reset", dest="rebase_leader_to_reset", action="store_false")
    parser.add_argument("--no-rotate-180", action="store_true")
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
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_name = DEFAULT_RECORD_DATASET_NAME if args.target == "orange" else f"{args.target}_wrist_grasp_wrist_ready_v1"
    args.repo_id = args.repo_id or (DEFAULT_RECORD_REPO_ID if args.target == "orange" else f"embody_car/{dataset_name}")
    root = Path(args.root) if args.root else WORKSPACE_DIR / "datasets" / dataset_name
    args.task = args.task or (DEFAULT_RECORD_TASK if args.target == "orange" else f"grasp the {args.target} with the gripper")
    if args.reset_pos is not None:
        reset_before_pos = parse_positions(args.reset_pos)
        reset_after_pos = parse_positions(args.reset_pos)
    else:
        reset_before_pos = parse_positions(args.reset_before_pos)
        reset_after_pos = parse_positions(args.reset_after_pos)

    if args.overwrite and root.exists():
        shutil.rmtree(root)

    features = make_features(height=args.height, width=args.width)
    dataset = create_or_resume_dataset(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=features,
        resume=args.resume,
        vcodec=args.vcodec,
    )

    camera = cv2.VideoCapture(args.wrist_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    camera.set(cv2.CAP_PROP_FPS, float(args.fps))
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open wrist camera index {args.wrist_camera}.")

    teleop = LeaderFollowerTeleop(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        speed=args.speed,
        acc=args.acc,
        min_delta=args.min_delta,
        dry_run=False,
        direct_raw=False,
        servo_ids=parse_ids(args.servo_ids),
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

    print(f"Recording dataset: repo_id={args.repo_id}, root={root}")
    print(f"Task: {args.task}")
    print(f"Wrist camera: {args.wrist_camera}, size={args.width}x{args.height}, fps={args.fps}")
    print("Press q in the preview window to end an episode early.")

    try:
        for episode_idx in range(args.num_episodes):
            print(f"\n=== Episode {episode_idx + 1}/{args.num_episodes} ===")
            record_episode(
                dataset=dataset,
                teleop=teleop,
                camera=camera,
                args=args,
                reset_before_pos=reset_before_pos,
                reset_after_pos=reset_after_pos,
            )
    finally:
        teleop.close()
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        dataset.finalize()
        dataset.stop_image_writer()
        print(f"Dataset finalized at {root}")


if __name__ == "__main__":
    main()
