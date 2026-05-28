#!/usr/bin/env python3
"""Run the trained orange wrist-grasp ACT policy on the follower arm."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from arm_control import ServoController
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_POLICY_PATH,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    JOINT_MAP,
    MIN_DELTA_TICKS,
    ORANGE_FINAL_CLOSE_POS,
    ORANGE_POLICY_ACC,
    ORANGE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD,
    ORANGE_PRE_CLOSE_HOLD_S,
    ORANGE_PRE_CLOSE_SERVO4_OFFSET,
    RESET_BEFORE_POS,
    RESET_ORDER_GROUPS,
    clamp,
    normalize_servo_reading,
    positions_to_array,
    sanitize_follower_positions,
)


def parse_offsets(text: str) -> dict[int, int]:
    offsets: dict[int, int] = {}
    if not text:
        return offsets
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        servo_id, value = part.split(":", maxsplit=1)
        servo_id_int = int(servo_id)
        if servo_id_int not in JOINT_MAP:
            raise ValueError(f"Unknown servo id in --action-offsets: {servo_id_int}")
        offsets[servo_id_int] = int(value)
    return offsets


def read_follower_positions(arm: ServoController, servo_ids: tuple[int, ...]) -> dict[int, int]:
    positions: dict[int, int] = {}
    for servo_id in servo_ids:
        pos = arm.get_position(servo_id)
        if pos < 0:
            continue
        positions[servo_id] = normalize_servo_reading(int(pos))
    return positions


def reset_follower(
    arm: ServoController,
    *,
    reset_pos: dict[int, int],
    speed: int,
    acc: int,
    stage_delay_s: float,
) -> dict[int, int]:
    last_sent: dict[int, int] = {}
    print(f"Resetting follower to {reset_pos}")
    for group in RESET_ORDER_GROUPS:
        for servo_id in group:
            target = reset_pos.get(servo_id)
            if target is None:
                continue
            arm.move_to(servo_id, target, speed=speed, acc=acc)
            last_sent[servo_id] = target
        time.sleep(stage_delay_s)
    return last_sent


def read_wrist_frame(
    camera: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    rotate_180: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read wrist camera frame.")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0
    return frame, image


def clamp_action(action: np.ndarray) -> np.ndarray:
    clamped = action.copy()
    for idx, servo_id in enumerate(range(1, 7)):
        cfg = JOINT_MAP[servo_id]
        clamped[idx] = clamp(
            int(round(clamped[idx])),
            int(cfg["follower_min"]),
            int(cfg["follower_max"]),
        )
    return clamped.astype(np.int32)


def apply_action_offsets(action: np.ndarray, offsets: dict[int, int]) -> np.ndarray:
    if not offsets:
        return action
    adjusted = action.copy()
    for servo_id, offset in offsets.items():
        adjusted[servo_id - 1] = int(adjusted[servo_id - 1]) + int(offset)
    return clamp_action(adjusted)


def limit_action_step(
    action: np.ndarray,
    *,
    last_sent: dict[int, int],
    target_max_step: int,
    gripper_target_max_step: int,
) -> np.ndarray:
    limited = action.copy()
    for idx, servo_id in enumerate(range(1, 7)):
        last = last_sent.get(servo_id)
        if last is None:
            continue
        max_step = gripper_target_max_step if servo_id == 6 else target_max_step
        if max_step <= 0:
            continue
        delta = int(limited[idx]) - int(last)
        if abs(delta) > max_step:
            limited[idx] = int(last) + (max_step if delta > 0 else -max_step)
    return limited.astype(np.int32)


def send_action(
    arm: ServoController,
    action: np.ndarray,
    *,
    last_sent: dict[int, int],
    speed: int,
    acc: int,
    min_delta: int,
    execute: bool,
) -> None:
    for idx, servo_id in enumerate(range(1, 7)):
        target = int(action[idx])
        last = last_sent.get(servo_id)
        if last is not None and abs(target - last) < min_delta:
            continue
        last_sent[servo_id] = target
        if execute:
            arm.move_to(servo_id, target, speed=speed, acc=acc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained ACT orange grasp policy with wrist camera observations."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--duration-s", type=float, default=ORANGE_POLICY_DURATION_S)
    parser.add_argument("--speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--acc", type=int, default=ORANGE_POLICY_ACC)
    parser.add_argument("--min-delta", type=int, default=MIN_DELTA_TICKS)
    parser.add_argument("--target-max-step", type=int, default=FOLLOWER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--gripper-target-max-step", type=int, default=GRIPPER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--state-max-jump", type=int, default=900)
    parser.add_argument("--state-range-margin", type=int, default=80)
    parser.add_argument("--reset-before", action="store_true")
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument(
        "--action-offsets",
        default="",
        help="Comma-separated servo tick offsets applied after policy output, e.g. '2:-40,3:60'.",
    )
    parser.add_argument(
        "--close-after-s",
        type=float,
        default=-1.0,
        help="After this many seconds, force gripper target toward --close-pos. Use -1 to disable.",
    )
    parser.add_argument("--close-pos", type=int, default=900)
    parser.add_argument(
        "--pre-close-servo4-offset",
        type=int,
        default=ORANGE_PRE_CLOSE_SERVO4_OFFSET,
        help="Apply this one-shot servo 4 tick offset immediately before the gripper starts closing.",
    )
    parser.add_argument(
        "--pre-close-gripper-threshold",
        type=int,
        default=ORANGE_PRE_CLOSE_GRIPPER_THRESHOLD,
        help="Trigger pre-close compensation when policy gripper target is at or below this tick value.",
    )
    parser.add_argument("--pre-close-hold-s", type=float, default=ORANGE_PRE_CLOSE_HOLD_S)
    parser.add_argument("--final-close", action="store_true", help="Close the gripper at the end and hold briefly.")
    parser.add_argument("--final-close-pos", type=int, default=ORANGE_FINAL_CLOSE_POS)
    parser.add_argument("--final-close-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--final-reach-offsets",
        default="",
        help="Relative servo tick offsets applied once before final close, e.g. '3:-120,4:80'.",
    )
    parser.add_argument("--final-reach-hold-s", type=float, default=0.5)
    parser.add_argument(
        "--return-after-close",
        action="store_true",
        help="After final close, move joints 1-5 to reset while keeping the gripper closed.",
    )
    parser.add_argument("--no-rotate-180", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send policy actions to the follower arm.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    action_offsets = parse_offsets(args.action_offsets)
    final_reach_offsets = parse_offsets(args.final_reach_offsets)
    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    print(f"Loading policy from {policy_path}")
    policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
    preprocessor = DataProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
    )
    policy.eval()
    policy.reset()

    camera = cv2.VideoCapture(args.wrist_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    camera.set(cv2.CAP_PROP_FPS, float(args.fps))
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open wrist camera index {args.wrist_camera}.")

    arm = ServoController(port=args.follower_port)
    last_state: np.ndarray | None = None
    last_sent: dict[int, int] = {}
    if args.reset_before:
        last_sent.update(
            reset_follower(
                arm,
                reset_pos=RESET_BEFORE_POS,
                speed=args.reset_speed,
                acc=args.reset_acc,
                stage_delay_s=args.reset_stage_delay_s,
            )
        )
        last_state = positions_to_array(last_sent)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: duration={args.duration_s}s fps={args.fps}")
    if action_offsets:
        print(f"Action offsets: {action_offsets}")
    if final_reach_offsets:
        print(f"Final reach offsets: {final_reach_offsets}")
    if args.close_after_s >= 0:
        print(f"Force gripper close after {args.close_after_s:.2f}s -> {args.close_pos}")
    if args.pre_close_servo4_offset:
        print(
            "Pre-close compensation: "
            f"servo4 offset={args.pre_close_servo4_offset}, "
            f"gripper_threshold={args.pre_close_gripper_threshold}"
        )
    print("Press Ctrl+C to stop.")

    frame_count = max(1, int(round(args.duration_s * args.fps)))
    start_time = time.monotonic()
    next_time = time.monotonic()
    pre_close_applied = False
    overruns = 0
    try:
        for frame_index in range(frame_count):
            preview, image = read_wrist_frame(
                camera,
                width=args.width,
                height=args.height,
                rotate_180=not args.no_rotate_180,
            )
            follower_positions = sanitize_follower_positions(
                read_follower_positions(arm, (1, 2, 3, 4, 5, 6)),
                last_state=last_state,
                max_jump=args.state_max_jump,
                range_margin=args.state_range_margin,
            )
            if last_state is None and len(follower_positions) < 6:
                raise RuntimeError(f"Could not read all follower positions: {follower_positions}")

            state = positions_to_array(
                follower_positions,
                fallback=last_state if last_state is not None else None,
            )
            observation = {
                "observation.state": torch.from_numpy(state).float(),
                "observation.images.wrist": image,
            }
            batch = preprocessor.process_observation(observation)
            with torch.no_grad():
                normalized_action = policy.select_action(batch)
                action_tensor = postprocessor.process_action(normalized_action)
            action = action_tensor.squeeze(0).detach().cpu().numpy()
            action = clamp_action(action)
            action = apply_action_offsets(action, action_offsets)
            elapsed_s = time.monotonic() - start_time
            force_close_now = args.close_after_s >= 0 and elapsed_s >= args.close_after_s
            pre_close_requested = (
                bool(args.pre_close_servo4_offset)
                and not pre_close_applied
                and (force_close_now or int(action[5]) <= args.pre_close_gripper_threshold)
            )
            if pre_close_requested:
                pre_close_action = action.copy()
                pre_close_action[3] = clamp(
                    int(pre_close_action[3]) + args.pre_close_servo4_offset,
                    JOINT_MAP[4]["follower_min"],
                    JOINT_MAP[4]["follower_max"],
                )
                pre_close_action[5] = int(last_sent.get(6, state[5]))
                pre_close_action = clamp_action(pre_close_action)
                pre_close_action = limit_action_step(
                    pre_close_action,
                    last_sent=last_sent,
                    target_max_step=args.target_max_step,
                    gripper_target_max_step=args.gripper_target_max_step,
                )
                print(f"Pre-close servo4 compensation -> {pre_close_action.tolist()}")
                send_action(
                    arm,
                    pre_close_action,
                    last_sent=last_sent,
                    speed=args.speed,
                    acc=args.acc,
                    min_delta=0,
                    execute=args.execute,
                )
                if args.execute and args.pre_close_hold_s > 0:
                    time.sleep(args.pre_close_hold_s)
                pre_close_applied = True
                action[3] = pre_close_action[3]

            if force_close_now:
                action[5] = clamp(args.close_pos, JOINT_MAP[6]["follower_min"], JOINT_MAP[6]["follower_max"])
            action = limit_action_step(
                action,
                last_sent=last_sent,
                target_max_step=args.target_max_step,
                gripper_target_max_step=args.gripper_target_max_step,
            )

            send_action(
                arm,
                action,
                last_sent=last_sent,
                speed=args.speed,
                acc=args.acc,
                min_delta=args.min_delta,
                execute=args.execute,
            )
            last_state = state

            if frame_index % max(1, int(args.fps)) == 0:
                print(f"frame={frame_index + 1}/{frame_count} state={state.astype(int).tolist()} action={action.tolist()}")

            if args.show:
                cv2.putText(
                    preview,
                    f"{mode} frame {frame_index + 1}/{frame_count}",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if args.execute else (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("orange_act_policy", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            next_time += 1.0 / args.fps
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                overruns += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if final_reach_offsets:
            base = np.asarray([last_sent.get(sid, RESET_BEFORE_POS.get(sid, 2048)) for sid in range(1, 7)])
            reach_action = apply_action_offsets(base, final_reach_offsets)
            print(f"Final reach -> {reach_action.tolist()}")
            send_action(
                arm,
                reach_action,
                last_sent=last_sent,
                speed=args.speed,
                acc=args.acc,
                min_delta=0,
                execute=args.execute,
            )
            if args.execute:
                time.sleep(args.final_reach_hold_s)
        if args.final_close:
            target = clamp(args.final_close_pos, JOINT_MAP[6]["follower_min"], JOINT_MAP[6]["follower_max"])
            print(f"Final close gripper -> {target}")
            last_sent[6] = target
            if args.execute:
                arm.move_to(6, target, speed=args.speed, acc=args.acc)
                time.sleep(args.final_close_hold_s)
        if args.return_after_close:
            reset_closed = {
                1: RESET_BEFORE_POS[1],
                2: RESET_BEFORE_POS[2],
                3: RESET_BEFORE_POS[3],
                4: RESET_BEFORE_POS[4],
                5: RESET_BEFORE_POS[5],
                6: clamp(args.final_close_pos, JOINT_MAP[6]["follower_min"], JOINT_MAP[6]["follower_max"]),
            }
            print(f"Return after close -> {reset_closed}")
            if args.execute:
                reset_follower(
                    arm,
                    reset_pos=reset_closed,
                    speed=args.reset_speed,
                    acc=args.reset_acc,
                    stage_delay_s=args.reset_stage_delay_s,
                )
        if overruns:
            print(f"Loop overran target fps on {overruns}/{frame_count} frames.")
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        if hasattr(arm, "_ser"):
            arm._ser.close()


if __name__ == "__main__":
    main()
