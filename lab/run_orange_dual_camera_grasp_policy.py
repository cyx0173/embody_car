#!/usr/bin/env python3
"""Run a dual-camera ACT policy that grasps a red-marked orange."""

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
from mark_one_bowl import YOLOSegMarker
from orange_grasp_config import (
    DEFAULT_DUAL_GRASP_POLICY_PATH,
    DEFAULT_FOLLOWER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    JOINT_MAP,
    MIN_DELTA_TICKS,
    ORANGE_FINAL_CLOSE_POS,
    ORANGE_POLICY_ACC,
    ORANGE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    RESET_BEFORE_POS,
    clamp,
    positions_to_array,
    sanitize_follower_positions,
)
from record_orange_dual_camera_dataset import RED_BGR
from record_orange_to_bowl_dual_camera_dataset import open_camera, read_dual_camera_frames
from run_orange_act_policy import (
    clamp_action,
    limit_action_step,
    read_follower_positions,
    reset_follower,
    send_action,
)
from run_orange_to_bowl_policy import REQUIRED_POLICY_FILES, choose_device


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

    config_path = policy_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        policy_config = json.load(f)
    if policy_config.get("type") is None:
        policy_config = {"type": "act", **policy_config}
        config_path.write_text(json.dumps(policy_config, indent=4) + "\n", encoding="utf-8")
        print(f"Patched policy config for local LeRobot compatibility: {config_path}")

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


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0


def show_dual_preview(
    *,
    wrist_rgb: np.ndarray,
    external_rgb: np.ndarray,
    mode: str,
    frame_index: int,
    frame_count: int,
    execute: bool,
    marker_meta: dict | None,
) -> bool:
    wrist_preview = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    external_preview = cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR)
    color = (0, 255, 0) if execute else (0, 255, 255)
    cv2.putText(
        wrist_preview,
        f"{mode} wrist {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    marker_text = "marker=none"
    if marker_meta:
        marker_text = (
            f"marker={marker_meta.get('class_name')} "
            f"conf={marker_meta.get('confidence') or 0.0:.2f}"
        )
    cv2.putText(
        external_preview,
        f"{mode} external {frame_index}/{frame_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        external_preview,
        marker_text,
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("orange_dual_camera_grasp_policy", cv2.hconcat([wrist_preview, external_preview]))
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the red-external-marker dual-camera orange grasp ACT policy."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_DUAL_GRASP_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Policy inference device. auto prefers MPS on Mac, then CUDA, then CPU.",
    )
    parser.add_argument("--reset-before", action="store_true")
    parser.add_argument("--reset-speed", type=int, default=1300)
    parser.add_argument("--reset-acc", type=int, default=55)
    parser.add_argument("--reset-stage-delay-s", type=float, default=0.9)
    parser.add_argument("--final-close", dest="final_close", action="store_true")
    parser.add_argument("--no-final-close", dest="final_close", action="store_false")
    parser.add_argument("--final-close-pos", type=int, default=ORANGE_FINAL_CLOSE_POS)
    parser.add_argument("--final-close-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--return-after-close",
        action="store_true",
        help="After final close, move joints 1-5 to reset while keeping the gripper closed.",
    )
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
    parser.add_argument("--execute", action="store_true", help="Actually send policy actions to the follower arm.")
    parser.set_defaults(final_close=True)
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
    print(f"Loading dual-camera orange grasp policy from {policy_path}")
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
        fps=int(round(args.fps)),
        name="wrist",
    )
    external_camera = open_camera(
        args.external_camera,
        width=args.width,
        height=args.height,
        fps=int(round(args.fps)),
        name="external",
    )
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
    print(f"Camera indexes: wrist={args.wrist_camera}, external={args.external_camera}")
    print(
        "External red marker: "
        f"target={args.marker_target}, choose={args.marker_choose}, model={args.marker_model}"
    )
    print("Press Ctrl+C to stop.")

    frame_count = max(1, int(round(args.duration_s * args.fps)))
    next_time = time.monotonic()
    overruns = 0
    try:
        for frame_index in range(frame_count):
            wrist_rgb, external_rgb = read_dual_camera_frames(
                wrist_camera=wrist_camera,
                external_camera=external_camera,
                args=args,
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
                "observation.images.wrist": image_to_tensor(wrist_rgb),
                "observation.images.external": image_to_tensor(external_rgb),
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
                print(
                    f"frame={frame_index + 1}/{frame_count} "
                    f"state={state.astype(int).tolist()} "
                    f"action={action.tolist()} "
                    f"marker={getattr(args, 'last_marker_meta', None)}"
                )

            if args.show:
                if not show_dual_preview(
                    wrist_rgb=wrist_rgb,
                    external_rgb=external_rgb,
                    mode=mode,
                    frame_index=frame_index + 1,
                    frame_count=frame_count,
                    execute=args.execute,
                    marker_meta=getattr(args, "last_marker_meta", None),
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
        wrist_camera.release()
        external_camera.release()
        if args.show:
            cv2.destroyAllWindows()
        if hasattr(arm, "_ser"):
            arm._ser.close()


if __name__ == "__main__":
    main()
