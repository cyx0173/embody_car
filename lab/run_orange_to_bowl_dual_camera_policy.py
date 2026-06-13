#!/usr/bin/env python3
"""Run a dual-camera ACT policy that places a held orange into a bowl."""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_DUAL_PLACE_POLICY_PATH,
    DEFAULT_FOLLOWER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    MIN_DELTA_TICKS,
    ORANGE_PLACE_POLICY_DURATION_S,
    ORANGE_POLICY_ACC,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    PLACE_RESET_OPEN_POS,
    positions_to_array,
    sanitize_follower_positions,
)
from run_orange_act_policy import (
    clamp_action,
    limit_action_step,
    read_follower_positions,
    read_wrist_frame,
    reset_follower,
    send_action,
)
from run_orange_to_bowl_policy import (
    REQUIRED_POLICY_FILES,
    build_hold_reset_pos,
    choose_device,
    prepare_held_orange,
    reset_or_preview,
)


def validate_policy_path(policy_path: Path) -> None:
    missing = [name for name in REQUIRED_POLICY_FILES if not (policy_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Policy path is missing required files: {missing}\n"
            f"path={policy_path}"
        )

    train_config_path = policy_path / "train_config.json"
    with train_config_path.open("r", encoding="utf-8") as f:
        train_config = json.load(f)
    repo_id = train_config.get("dataset", {}).get("repo_id")
    policy_type = train_config.get("policy", {}).get("type")
    input_features = train_config.get("policy", {}).get("input_features", {})
    output_features = train_config.get("policy", {}).get("output_features", {})

    if policy_type != "act":
        print(f"WARNING: train_config policy type is {policy_type!r}, expected 'act'.")
    if "observation.images.wrist" not in input_features:
        print("WARNING: policy input does not list observation.images.wrist.")
    if "observation.images.external" not in input_features:
        print("WARNING: policy input does not list observation.images.external.")
    if output_features.get("action", {}).get("shape") != [6]:
        print(f"WARNING: policy action shape is {output_features.get('action', {}).get('shape')}, expected [6].")
    print(f"Policy dataset repo_id: {repo_id}")


def open_camera(camera_index: int, *, width: int, height: int, fps: float, name: str) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open {name} camera index {camera_index}.")
    return camera


def read_external_frame(
    camera: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    rotate_180: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read external camera frame.")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0
    return frame, image


def show_dual_preview(
    *,
    wrist_preview: np.ndarray,
    external_preview: np.ndarray,
    mode: str,
    frame_index: int,
    frame_count: int,
    execute: bool,
) -> bool:
    cv2.putText(
        wrist_preview,
        f"{mode} wrist {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if execute else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        external_preview,
        f"{mode} external {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if execute else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("orange_to_bowl_dual_camera_policy", cv2.hconcat([wrist_preview, external_preview]))
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the orange-to-bowl ACT policy with wrist + external camera observations."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_DUAL_PLACE_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--duration-s", type=float, default=ORANGE_PLACE_POLICY_DURATION_S)
    parser.add_argument("--speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--acc", type=int, default=ORANGE_POLICY_ACC)
    parser.add_argument("--min-delta", type=int, default=MIN_DELTA_TICKS)
    parser.add_argument("--target-max-step", type=int, default=FOLLOWER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--gripper-target-max-step", type=int, default=GRIPPER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--state-max-jump", type=int, default=900)
    parser.add_argument("--state-range-margin", type=int, default=80)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Policy inference device. auto prefers MPS on Mac, then CUDA, then CPU.",
    )
    parser.add_argument("--reset-hold-before", action="store_true")
    parser.add_argument("--prepare-held-orange", action="store_true")
    parser.add_argument("--hold-gripper-pos", type=int, default=None)
    parser.add_argument("--hold-settle-s", type=float, default=0.4)
    parser.add_argument("--pre-run-delay-s", type=float, default=2.0)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--return-open-after", action="store_true")
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument("--no-wrist-rotate-180", action="store_true")
    parser.add_argument("--external-rotate-180", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send policy actions to the follower arm.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")

    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    validate_policy_path(policy_path)

    device = choose_device(args.device)
    print(f"Loading dual-camera orange-to-bowl policy from {policy_path}")
    print(f"Inference device: {device}")
    policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True, device=device)
    preprocessor = DataProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": device}},
    )
    policy.to(device)
    policy.eval()
    policy.reset()

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

    arm = ServoController(port=args.follower_port)
    last_state: np.ndarray | None = None
    last_sent: dict[int, int] = {}
    if args.prepare_held_orange:
        last_sent.update(prepare_held_orange(arm, args=args))
        last_state = positions_to_array(last_sent)
    elif args.reset_hold_before:
        last_sent.update(
            reset_or_preview(
                arm,
                reset_pos=build_hold_reset_pos(args.hold_gripper_pos),
                args=args,
                label="holding reset",
            )
        )
        last_state = positions_to_array(last_sent)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: duration={args.duration_s}s fps={args.fps}")
    print(
        "Task assumption: start with the orange already held in the gripper, "
        "and the bowl visible in wrist/external views."
    )
    print(f"Camera indexes: wrist={args.wrist_camera}, external={args.external_camera}")
    print("Press Ctrl+C to stop.")

    frame_count = max(1, int(round(args.duration_s * args.fps)))
    next_time = time.monotonic()
    overruns = 0
    try:
        for frame_index in range(frame_count):
            wrist_preview, wrist_image = read_wrist_frame(
                wrist_camera,
                width=args.width,
                height=args.height,
                rotate_180=not args.no_wrist_rotate_180,
            )
            external_preview, external_image = read_external_frame(
                external_camera,
                width=args.width,
                height=args.height,
                rotate_180=args.external_rotate_180,
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
                "observation.images.wrist": wrist_image,
                "observation.images.external": external_image,
            }
            batch = preprocessor.process_observation(observation)
            with torch.no_grad():
                normalized_action = policy.select_action(batch)
                action_tensor = postprocessor.process_action(normalized_action)
            action = action_tensor.squeeze(0).detach().cpu().numpy()
            action = clamp_action(action)
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
                if not show_dual_preview(
                    wrist_preview=wrist_preview,
                    external_preview=external_preview,
                    mode=mode,
                    frame_index=frame_index + 1,
                    frame_count=frame_count,
                    execute=args.execute,
                ):
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
        if args.return_open_after:
            print(f"Return open reset -> {PLACE_RESET_OPEN_POS}")
            if args.execute:
                reset_follower(
                    arm,
                    reset_pos=PLACE_RESET_OPEN_POS,
                    speed=args.reset_speed,
                    acc=args.reset_acc,
                    stage_delay_s=args.reset_stage_delay_s,
                )
        if overruns:
            print(f"Loop overran target fps on {overruns}/{frame_count} frames.")
        wrist_camera.release()
        external_camera.release()
        if args.show:
            cv2.destroyAllWindows()
        if hasattr(arm, "_ser"):
            arm._ser.close()


if __name__ == "__main__":
    main()
