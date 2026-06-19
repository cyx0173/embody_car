#!/usr/bin/env python3
"""Run dual-camera grasp policy with hot-key takeover by the leader arm."""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from arm_control import ServoController
from leader_follower_teleop import LeaderFollowerTeleop
from orange_grasp_config import (
    DEFAULT_DUAL_GRASP_POLICY_PATH,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    LEADER_SPIKE_CONFIRM_FRAMES,
    LEADER_SPIKE_MAX_DELTA_TICKS,
    LOOP_INTERVAL_S,
    MIN_DELTA_TICKS,
    ORANGE_POLICY_ACC,
    ORANGE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
    SERVO_IDS,
    clamp_follower_target,
    normalize_servo_reading,
    parse_ids,
    positions_to_array,
    sanitize_follower_positions,
)


RED_BGR = (0, 0, 255)
REQUIRED_POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "train_config.json",
)


class RawTerminal:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self.fd: int | None = None
        self.old_settings = None

    def __enter__(self) -> "RawTerminal":
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc) -> None:
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _w, _e = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1)


def import_policy_runtime():
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.pipeline import DataProcessorPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing policy runtime dependency. Activate the environment that has "
            "torch and lerobot installed before running policy inference."
        ) from exc
    return torch, ACTPolicy, DataProcessorPipeline


def import_marker_runtime():
    try:
        from mark_one_bowl import YOLOSegMarker
    except ImportError as exc:
        raise RuntimeError(
            "Missing marker runtime dependency. Install ultralytics/opencv dependencies "
            "or run in the robot environment."
        ) from exc
    return YOLOSegMarker


def choose_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


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


def image_to_tensor(torch_module, rgb: np.ndarray):
    return torch_module.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0


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
    target_choice = getattr(args, "episode_target_bowl_choice", None)
    if marker is not None and target_choice is not None:
        external_bgr = cv2.cvtColor(external_image, cv2.COLOR_RGB2BGR)
        marker.choose = target_choice
        marked_bgr, meta = marker.infer(
            external_bgr,
            args.marker_target,
            return_meta=True,
            mode=args.marker_mode,
        )
        args.last_marker_meta = meta
        external_image = cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)
    return wrist_image, external_image


def read_follower_positions(arm: ServoController, servo_ids: tuple[int, ...]) -> dict[int, int]:
    positions: dict[int, int] = {}
    for servo_id in servo_ids:
        pos = arm.get_position(servo_id)
        if pos < 0:
            continue
        positions[servo_id] = normalize_servo_reading(int(pos))
    return positions


def clamp_action(action: np.ndarray) -> np.ndarray:
    clamped = action.copy()
    for idx, servo_id in enumerate(SERVO_IDS):
        clamped[idx] = clamp_follower_target(servo_id, int(round(clamped[idx])))
    return clamped.astype(np.int32)


def limit_action_step(
    action: np.ndarray,
    *,
    last_sent: dict[int, int],
    target_max_step: int,
    gripper_target_max_step: int,
) -> np.ndarray:
    limited = action.copy()
    for idx, servo_id in enumerate(SERVO_IDS):
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
    for idx, servo_id in enumerate(SERVO_IDS):
        target = int(action[idx])
        last = last_sent.get(servo_id)
        if last is not None and abs(target - last) < min_delta:
            continue
        last_sent[servo_id] = target
        if execute:
            arm.move_to(servo_id, target, speed=speed, acc=acc)


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
    cv2.imshow("dual_camera_grasp_hybrid_control", cv2.hconcat([wrist_preview, external_preview]))
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dual-camera ACT grasp with in-process leader-arm takeover. "
            "Press the takeover key during policy execution to switch to teleop."
        )
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_DUAL_GRASP_POLICY_PATH)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
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
    parser.add_argument("--takeover-key", default="t", help="Press this key to switch from policy to teleop.")
    parser.add_argument("--quit-key", default="q", help="Press this key to quit.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the follower arm.")

    parser.add_argument("--teleop-speed", type=int, default=1800)
    parser.add_argument("--teleop-acc", type=int, default=45)
    parser.add_argument("--teleop-loop-interval", type=float, default=LOOP_INTERVAL_S)
    parser.add_argument("--teleop-debug-targets", action="store_true")
    parser.add_argument("--teleop-debug-unchanged", action="store_true")
    parser.add_argument("--teleop-direct-raw", action="store_true")
    parser.add_argument("--teleop-leader-max-jump", type=int, default=LEADER_SPIKE_MAX_DELTA_TICKS)
    parser.add_argument("--teleop-spike-confirm-frames", type=int, default=LEADER_SPIKE_CONFIRM_FRAMES)
    parser.add_argument("--teleop-target-max-step", type=int, default=FOLLOWER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--teleop-gripper-target-max-step", type=int, default=GRIPPER_MAX_TARGET_STEP_TICKS)
    parser.add_argument(
        "--teleop-servo-ids",
        default=",".join(str(sid) for sid in SERVO_IDS),
        help=(
            "Comma-separated servo IDs controlled by leader after takeover. "
            "Use '6' for gripper-only takeover."
        ),
    )
    parser.add_argument(
        "--teleop-relative",
        action="store_true",
        help=(
            "Use relative/clutch teleop. At takeover, follower holds current pose; "
            "later leader movement deltas are added to that follower pose."
        ),
    )
    parser.add_argument(
        "--teleop-relative-gain",
        type=float,
        default=1.0,
        help="Gain applied to leader movement deltas in relative teleop.",
    )
    return parser


def make_teleop(args: argparse.Namespace, *, follower: ServoController) -> LeaderFollowerTeleop:
    return LeaderFollowerTeleop(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        speed=args.teleop_speed,
        acc=args.teleop_acc,
        min_delta=args.min_delta,
        dry_run=not args.execute,
        direct_raw=args.teleop_direct_raw,
        servo_ids=parse_ids(args.teleop_servo_ids),
        debug_targets=args.teleop_debug_targets,
        debug_unchanged=args.teleop_debug_unchanged,
        gripper_mode="linear",
        gripper_threshold=-1,
        leader_max_jump=args.teleop_leader_max_jump,
        spike_confirm_frames=args.teleop_spike_confirm_frames,
        target_max_step=args.teleop_target_max_step,
        gripper_target_max_step=args.teleop_gripper_target_max_step,
        loop_interval_s=args.teleop_loop_interval,
        follower=follower,
        close_follower=False,
        relative_mode=args.teleop_relative,
        relative_gain=args.teleop_relative_gain,
    )


def seed_teleop_targets(
    teleop: LeaderFollowerTeleop,
    *,
    arm: ServoController,
    policy_last_sent: dict[int, int],
) -> None:
    teleop_servo_ids = tuple(teleop.servo_ids)
    current_positions = read_follower_positions(arm, teleop_servo_ids)
    for servo_id in teleop_servo_ids:
        if servo_id in current_positions:
            teleop.last_sent[servo_id] = current_positions[servo_id]
        elif servo_id in policy_last_sent:
            teleop.last_sent[servo_id] = policy_last_sent[servo_id]


def switch_to_teleop(
    args: argparse.Namespace,
    *,
    arm: ServoController,
    policy_last_sent: dict[int, int],
) -> LeaderFollowerTeleop:
    teleop = make_teleop(args, follower=arm)
    seed_teleop_targets(teleop, arm=arm, policy_last_sent=policy_last_sent)
    return teleop


def main() -> None:
    args = build_arg_parser().parse_args()
    if len(args.takeover_key) != 1 or len(args.quit_key) != 1:
        raise ValueError("--takeover-key and --quit-key must be single characters.")
    if args.wrist_camera == args.external_camera:
        raise ValueError("wrist camera and external camera must use different indexes.")

    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    validate_policy_path(policy_path)

    torch, ACTPolicy, DataProcessorPipeline = import_policy_runtime()
    device = choose_device(torch, args.device)
    print(f"Loading dual-camera grasp policy from {policy_path}")
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
    YOLOSegMarker = import_marker_runtime()
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
    teleop: LeaderFollowerTeleop | None = None
    last_state: np.ndarray | None = None
    last_sent: dict[int, int] = {}
    mode = "POLICY"
    overruns = 0

    run_mode = "EXECUTE" if args.execute else "DRY RUN"
    frame_count = max(1, int(round(args.duration_s * args.fps)))
    print(f"{run_mode}: policy duration={args.duration_s}s fps={args.fps}")
    print(
        f"Press {args.takeover_key!r} to switch to leader teleop, "
        f"{args.quit_key!r} to quit, Ctrl+C to stop."
    )
    print(
        "Teleop config: "
        f"relative={args.teleop_relative} "
        f"servo_ids={args.teleop_servo_ids} "
        f"relative_gain={args.teleop_relative_gain}"
    )

    next_time = time.monotonic()
    with RawTerminal():
        try:
            frame_index = 0
            while True:
                key = read_key()
                if key == args.quit_key:
                    print("\nQuit requested.")
                    break
                if key == args.takeover_key and mode == "POLICY":
                    print("\nSwitching to TELEOP. Policy actions are stopped.")
                    mode = "TELEOP"
                    teleop = switch_to_teleop(args, arm=arm, policy_last_sent=last_sent)

                if mode == "TELEOP":
                    if teleop is None:
                        teleop = make_teleop(args, follower=arm)
                    teleop.step()
                    time.sleep(args.teleop_loop_interval)
                    continue

                if frame_index >= frame_count:
                    print("\nPolicy duration reached. Switching to TELEOP.")
                    mode = "TELEOP"
                    teleop = switch_to_teleop(args, arm=arm, policy_last_sent=last_sent)
                    continue

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
                    "observation.images.wrist": image_to_tensor(torch, wrist_rgb),
                    "observation.images.external": image_to_tensor(torch, external_rgb),
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
                frame_index += 1

                if frame_index % max(1, int(args.fps)) == 0:
                    print(
                        f"POLICY frame={frame_index}/{frame_count} "
                        f"state={state.astype(int).tolist()} action={action.tolist()}"
                    )

                if args.show:
                    if not show_dual_preview(
                        wrist_rgb=wrist_rgb,
                        external_rgb=external_rgb,
                        mode=run_mode,
                        frame_index=frame_index,
                        frame_count=frame_count,
                        execute=args.execute,
                        marker_meta=getattr(args, "last_marker_meta", None),
                    ):
                        print("\nPreview quit requested.")
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
            if overruns:
                print(f"Policy loop overran target fps on {overruns}/{frame_count} frames.")
            if teleop is not None:
                teleop.close()
            wrist_camera.release()
            external_camera.release()
            if args.show:
                cv2.destroyAllWindows()
            if hasattr(arm, "_ser"):
                arm._ser.close()


if __name__ == "__main__":
    main()
