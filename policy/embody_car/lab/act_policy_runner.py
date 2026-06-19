#!/usr/bin/env python3
"""Dual-camera ACT policy runner with manual runtime controls."""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
PROJECT_DIR = WORKSPACE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from robot_config import (
    DEFAULT_FOLLOWER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_FOLLOWER_CLOSE,
    GRIPPER_FOLLOWER_OPEN,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    MIN_DELTA_TICKS,
    POLICY_ACC,
    POLICY_DURATION_S,
    POLICY_FPS,
    POLICY_SPEED,
    SERVO_IDS,
    clamp_follower_target,
    normalize_servo_reading,
    positions_to_array,
    sanitize_follower_positions,
)
from servo_controller import ServoController
from support.marker import YOLOSegMarker


RED_BGR = (0, 0, 255)
DEFAULT_POLICY_PATH = PROJECT_DIR / "grasp_model"
DEFAULT_TRAJECTORY_DIR = PROJECT_DIR / "trajectory_logs"
DEFAULT_MANUAL_GRIPPER_CLOSE_POS = 580
DEFAULT_MANUAL_GRIPPER_MIN_POS = 430
DEFAULT_MANUAL_GRIPPER_SPEED = 3800
DEFAULT_MANUAL_GRIPPER_ACC = 120
DEFAULT_MANUAL_GRIPPER_HOLD_S = 2.0
DEFAULT_MANUAL_GRIPPER_SHAKE_AMP = 45
DEFAULT_MANUAL_GRIPPER_SHAKE_HZ = 7.0
DEFAULT_MANUAL_GRIPPER_STEP_TICKS = 220
DEFAULT_MANUAL_DOWN_SERVO = 2
DEFAULT_MANUAL_DOWN_OFFSET = 90
DEFAULT_NUDGE_STEP_TICKS = 45
DEFAULT_NUDGE_MAX_ABS_TICKS = 320
DEFAULT_EXTRA_NUDGE_STEP_TICKS = 35
DEFAULT_OPEN_GRIPPER_HOLD_S = 1.2
DEFAULT_OPEN_GRIPPER_SPEED = 3200
DEFAULT_OPEN_GRIPPER_ACC = 120
DEFAULT_OPEN_GRIPPER_STUTTER_STEPS = 4
DEFAULT_OPEN_GRIPPER_STUTTER_PAUSE_S = 0.06
DEFAULT_OPEN_GRIPPER_STEP_TICKS = 260
REQUIRED_POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.enabled = sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.old_settings = termios.tcgetattr(self.fd) if self.enabled else None
        if self.enabled:
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc) -> None:
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _w, _e = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if readable else None


def import_policy_runtime():
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.pipeline import DataProcessorPipeline
    except ImportError as exc:
        raise RuntimeError("Activate the environment with torch and lerobot installed.") from exc
    return torch, ACTPolicy, DataProcessorPipeline


def choose_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_marker_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def check_policy_path(policy_path: Path) -> None:
    missing = [name for name in REQUIRED_POLICY_FILES if not (policy_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Policy path is missing {missing}: {policy_path}")
    train_config = policy_path / "train_config.json"
    if train_config.exists():
        with train_config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Policy dataset repo_id: {cfg.get('dataset', {}).get('repo_id')}")


def image_to_tensor(torch_module, rgb: np.ndarray):
    array = np.ascontiguousarray(rgb)
    return torch_module.from_numpy(array).float().permute(2, 0, 1) / 255.0


def open_camera(index: int, *, width: int, height: int, fps: float, name: str) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open {name} camera index {index}.")
    return camera


def read_rgb(camera: cv2.VideoCapture, *, width: int, height: int, rotate_180: bool, name: str) -> np.ndarray:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read {name} camera frame.")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_follower_positions(arm: ServoController, servo_ids: tuple[int, ...]) -> dict[int, int]:
    positions: dict[int, int] = {}
    for servo_id in servo_ids:
        pos = arm.get_position(servo_id)
        if pos >= 0:
            positions[servo_id] = normalize_servo_reading(int(pos))
    return positions


def clamp_action(action: np.ndarray) -> np.ndarray:
    out = action.copy()
    for idx, servo_id in enumerate(SERVO_IDS):
        out[idx] = clamp_follower_target(servo_id, int(round(out[idx])))
    return out.astype(np.int32)


def limit_action_step(
    action: np.ndarray,
    *,
    last_sent: dict[int, int],
    target_max_step: int,
    gripper_target_max_step: int,
) -> np.ndarray:
    out = action.copy()
    for idx, servo_id in enumerate(SERVO_IDS):
        last = last_sent.get(servo_id)
        if last is None:
            continue
        max_step = gripper_target_max_step if servo_id == 6 else target_max_step
        if max_step <= 0:
            continue
        delta = int(out[idx]) - int(last)
        if abs(delta) > max_step:
            out[idx] = int(last) + (max_step if delta > 0 else -max_step)
    return out.astype(np.int32)


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


def clamp_manual_gripper_target(target: int, *, min_pos: int) -> int:
    return max(int(min_pos), min(2302, int(round(target))))


def apply_grasp_assist(action: np.ndarray, args: argparse.Namespace, elapsed_s: float) -> np.ndarray:
    assisted = action.copy()

    shake = int(round(args.manual_gripper_shake_amp * math.sin(2.0 * math.pi * args.manual_gripper_shake_hz * elapsed_s)))
    assisted[5] = clamp_manual_gripper_target(
        args.manual_gripper_close_pos + shake,
        min_pos=args.manual_gripper_min_pos,
    )

    down_servo = int(args.manual_down_servo)
    if down_servo in SERVO_IDS and args.manual_down_offset != 0:
        idx = SERVO_IDS.index(down_servo)
        ramp = min(1.0, max(0.0, elapsed_s / 0.35))
        assisted[idx] = clamp_follower_target(down_servo, int(round(assisted[idx] + args.manual_down_offset * ramp)))

    return assisted.astype(np.int32)


def clamp_offset(value: int, *, max_abs: int) -> int:
    return max(-int(max_abs), min(int(max_abs), int(value)))


class MarkerState:
    def __init__(self, marker: YOLOSegMarker, *, target: str, mode: str, every_n: int) -> None:
        self.marker = marker
        self.target = target
        self.mode = mode
        self.every_n = max(1, int(every_n))
        self.last_bbox: list[int] | None = None
        self.last_meta: dict | None = None

    def apply(self, external_rgb: np.ndarray, frame_index: int) -> np.ndarray:
        bgr = cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR)
        if frame_index % self.every_n == 0 or self.last_bbox is None:
            marked_bgr, meta = self.marker.infer(
                bgr,
                self.target,
                return_meta=True,
                mode=self.mode,
            )
            self.last_meta = meta
            self.last_bbox = meta.get("bbox_xyxy") if meta.get("found") else None
            return cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(self.marker.draw_bbox(bgr, self.last_bbox), cv2.COLOR_BGR2RGB)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class TrajectoryRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        root: Path,
        record_images: bool,
        metadata: dict,
    ) -> None:
        self.enabled = bool(enabled)
        self.record_images = bool(record_images)
        self.run_dir: Path | None = None
        self.jsonl_file = None
        if not self.enabled:
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = root.expanduser().resolve() / f"grasp_run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        if self.record_images:
            (self.run_dir / "wrist").mkdir()
            (self.run_dir / "external").mkdir()
        (self.run_dir / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        self.jsonl_file = (self.run_dir / "trajectory.jsonl").open("w", encoding="utf-8")
        print(f"Recording trajectory: {self.run_dir}")

    def record_frame(
        self,
        *,
        frame_index: int,
        elapsed_s: float,
        state: np.ndarray,
        action: np.ndarray,
        marker_meta: dict | None,
        manual_grasp_active: bool,
        manual_grasp_elapsed_s: float | None,
        wrist_rgb: np.ndarray,
        external_rgb: np.ndarray,
    ) -> None:
        if not self.enabled or self.jsonl_file is None:
            return

        entry = {
            "frame_index": int(frame_index),
            "elapsed_s": float(elapsed_s),
            "state": state.astype(float).tolist(),
            "action": action.astype(int).tolist(),
            "marker": json_safe(marker_meta),
            "manual_grasp_active": bool(manual_grasp_active),
            "manual_grasp_elapsed_s": manual_grasp_elapsed_s,
        }
        if self.record_images and self.run_dir is not None:
            wrist_name = f"{frame_index:06d}.jpg"
            external_name = f"{frame_index:06d}.jpg"
            cv2.imwrite(str(self.run_dir / "wrist" / wrist_name), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.run_dir / "external" / external_name), cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR))
            entry["wrist_image"] = f"wrist/{wrist_name}"
            entry["external_image"] = f"external/{external_name}"

        self.jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.jsonl_file is not None:
            self.jsonl_file.flush()
            self.jsonl_file.close()
            self.jsonl_file = None


def show_preview(
    *,
    wrist_rgb: np.ndarray,
    external_rgb: np.ndarray,
    mode: str,
    frame_index: int,
    frame_count: int,
    marker_meta: dict | None,
) -> str | None:
    wrist = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    external = cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(wrist, f"{mode} wrist {frame_index}/{frame_count}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    marker_text = "marker=none"
    if marker_meta:
        marker_text = f"marker={marker_meta.get('class_name')} conf={marker_meta.get('confidence') or 0.0:.2f}"
    cv2.putText(external, f"{mode} external {frame_index}/{frame_count}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(external, marker_text, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("grasp_clean", cv2.hconcat([wrist, external]))
    key = cv2.waitKey(1) & 0xFF
    if key == 255:
        return None
    try:
        return chr(key)
    except ValueError:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the default dual-camera red-marker ACT grasp policy.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=POLICY_FPS)
    parser.add_argument("--duration-s", type=float, default=POLICY_DURATION_S)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--speed", type=int, default=POLICY_SPEED)
    parser.add_argument("--acc", type=int, default=POLICY_ACC)
    parser.add_argument("--min-delta", type=int, default=MIN_DELTA_TICKS)
    parser.add_argument("--target-max-step", type=int, default=FOLLOWER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--gripper-target-max-step", type=int, default=GRIPPER_MAX_TARGET_STEP_TICKS)
    parser.add_argument("--state-max-jump", type=int, default=900)
    parser.add_argument("--state-range-margin", type=int, default=80)
    parser.add_argument("--no-wrist-rotate-180", action="store_true")
    parser.add_argument("--external-rotate-180", action="store_true")
    parser.add_argument("--marker-model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--marker-target", default="object")
    parser.add_argument("--marker-device", default="auto")
    parser.add_argument("--marker-conf", type=float, default=0.25)
    parser.add_argument("--marker-iou", type=float, default=0.7)
    parser.add_argument("--marker-alpha", type=float, default=0.20)
    parser.add_argument("--marker-thickness", type=int, default=6)
    parser.add_argument("--marker-mode", choices=("bbox", "mask", "both"), default="bbox")
    parser.add_argument("--marker-choose", choices=("largest", "highest_conf", "nearest_center", "leftmost", "rightmost"), default="largest")
    parser.add_argument("--marker-every-n", type=int, default=2, help="Run YOLO every N policy frames; cached bbox is drawn between detections.")
    parser.add_argument("--close-gripper-key", default="t")
    parser.add_argument("--next-stage-key", default="")
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--manual-gripper-close-pos", type=int, default=DEFAULT_MANUAL_GRIPPER_CLOSE_POS)
    parser.add_argument("--manual-gripper-min-pos", type=int, default=DEFAULT_MANUAL_GRIPPER_MIN_POS)
    parser.add_argument("--manual-gripper-speed", type=int, default=DEFAULT_MANUAL_GRIPPER_SPEED)
    parser.add_argument("--manual-gripper-acc", type=int, default=DEFAULT_MANUAL_GRIPPER_ACC)
    parser.add_argument("--manual-gripper-hold-s", type=float, default=DEFAULT_MANUAL_GRIPPER_HOLD_S)
    parser.add_argument("--manual-gripper-shake-amp", type=int, default=DEFAULT_MANUAL_GRIPPER_SHAKE_AMP)
    parser.add_argument("--manual-gripper-shake-hz", type=float, default=DEFAULT_MANUAL_GRIPPER_SHAKE_HZ)
    parser.add_argument("--manual-gripper-step", type=int, default=DEFAULT_MANUAL_GRIPPER_STEP_TICKS)
    parser.add_argument("--manual-down-servo", type=int, default=DEFAULT_MANUAL_DOWN_SERVO)
    parser.add_argument("--manual-down-offset", type=int, default=DEFAULT_MANUAL_DOWN_OFFSET)
    parser.add_argument("--nudge", dest="nudge_enabled", action="store_true", default=False)
    parser.add_argument("--nudge-step", type=int, default=DEFAULT_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-max", type=int, default=DEFAULT_NUDGE_MAX_ABS_TICKS)
    parser.add_argument("--nudge-left-key", default="a")
    parser.add_argument("--nudge-right-key", default="d")
    parser.add_argument("--nudge-forward-key", default="w")
    parser.add_argument("--nudge-back-key", default="s")
    parser.add_argument("--nudge-reset-key", default="x")
    parser.add_argument("--nudge-left-servo", type=int, default=1)
    parser.add_argument("--nudge-forward-servo", type=int, default=2)
    parser.add_argument("--nudge-back-servo", type=int, default=None)
    parser.add_argument("--nudge-left-delta", type=int, default=DEFAULT_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-forward-delta", type=int, default=DEFAULT_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-back-delta", type=int, default=None)
    parser.add_argument("--extra-nudge", dest="extra_nudge_enabled", action="store_true", default=False)
    parser.add_argument("--extra-nudge-step", type=int, default=DEFAULT_EXTRA_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-up-key", default="e")
    parser.add_argument("--nudge-down-key", default="c")
    parser.add_argument("--nudge-wrist-up-key", default="i")
    parser.add_argument("--nudge-wrist-down-key", default="k")
    parser.add_argument("--nudge-roll-left-key", default="j")
    parser.add_argument("--nudge-roll-right-key", default="l")
    parser.add_argument("--nudge-up-servo", type=int, default=3)
    parser.add_argument("--nudge-wrist-servo", type=int, default=4)
    parser.add_argument("--nudge-roll-servo", type=int, default=5)
    parser.add_argument("--nudge-up-delta", type=int, default=DEFAULT_EXTRA_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-wrist-delta", type=int, default=DEFAULT_EXTRA_NUDGE_STEP_TICKS)
    parser.add_argument("--nudge-roll-delta", type=int, default=DEFAULT_EXTRA_NUDGE_STEP_TICKS)
    parser.add_argument("--open-gripper-key", default="")
    parser.add_argument("--manual-open-pos", type=int, default=GRIPPER_FOLLOWER_OPEN)
    parser.add_argument("--manual-open-hold-s", type=float, default=DEFAULT_OPEN_GRIPPER_HOLD_S)
    parser.add_argument("--manual-open-speed", type=int, default=DEFAULT_OPEN_GRIPPER_SPEED)
    parser.add_argument("--manual-open-acc", type=int, default=DEFAULT_OPEN_GRIPPER_ACC)
    parser.add_argument("--manual-open-stutter-steps", type=int, default=DEFAULT_OPEN_GRIPPER_STUTTER_STEPS)
    parser.add_argument("--manual-open-stutter-pause-s", type=float, default=DEFAULT_OPEN_GRIPPER_STUTTER_PAUSE_S)
    parser.add_argument("--manual-open-step", type=int, default=DEFAULT_OPEN_GRIPPER_STEP_TICKS)
    parser.add_argument("--lock-gripper-until-open-key", action="store_true")
    parser.add_argument("--locked-gripper-pos", type=int, default=None)
    parser.add_argument("--quiet-controls", action="store_true")
    parser.add_argument("--show", dest="show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--execute", dest="execute", action="store_true", default=True)
    parser.add_argument("--dry-run", dest="execute", action="store_false")
    parser.add_argument("--record", action="store_true", help="Record state/action/marker trajectory to trajectory_logs.")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--record-images", action="store_true", help="Also save wrist/external images. This can slow down control.")
    return parser


class DualCameraGraspRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.wrist_camera == args.external_camera:
            raise ValueError("wrist camera and external camera must use different indexes.")
        self.args = args
        self.policy_path = args.policy_path.expanduser().resolve()
        self.torch = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.wrist_camera: cv2.VideoCapture | None = None
        self.external_camera: cv2.VideoCapture | None = None
        self.arm: ServoController | None = None
        self.marker_state: MarkerState | None = None
        self.last_state: np.ndarray | None = None
        self.last_sent: dict[int, int] = {}
        self.mode = "POLICY"
        self.overruns = 0
        self.manual_gripper_started = 0.0
        self.manual_gripper_until = 0.0
        self.manual_gripper_start_pos: int | None = None
        self.manual_gripper_target_pos: int | None = None
        self.manual_open_started = 0.0
        self.manual_open_until = 0.0
        self.manual_open_start_pos: int | None = None
        self.manual_open_target_pos: int | None = None
        self.manual_open_triggered = False
        self.locked_gripper_pos: int | None = None
        self.action_offsets: dict[int, int] = {}
        self.recorder: TrajectoryRecorder | None = None
        self.stop_reason = "done"

    def setup(self) -> None:
        args = self.args
        check_policy_path(self.policy_path)
        torch, ACTPolicy, DataProcessorPipeline = import_policy_runtime()
        self.torch = torch
        device = choose_device(torch, args.device)
        marker_device = choose_marker_device(torch, args.marker_device)

        print(f"Loading policy: {self.policy_path}")
        print(f"Inference device: {device}")
        print(f"Marker device: {marker_device}")

        self.policy = ACTPolicy.from_pretrained(self.policy_path, local_files_only=True, device=device)
        self.preprocessor = DataProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={"device_processor": {"device": device}},
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            overrides={"device_processor": {"device": device}},
        )
        self.policy.to(device)
        self.policy.eval()
        self.policy.reset()

        self.wrist_camera = open_camera(args.wrist_camera, width=args.width, height=args.height, fps=args.fps, name="wrist")
        self.external_camera = open_camera(args.external_camera, width=args.width, height=args.height, fps=args.fps, name="external")
        self.arm = ServoController(port=args.follower_port)

        marker = YOLOSegMarker(
            model=args.marker_model,
            device=marker_device,
            conf=args.marker_conf,
            iou=args.marker_iou,
            alpha=args.marker_alpha,
            thickness=args.marker_thickness,
            mode=args.marker_mode,
            choose=args.marker_choose,
            color_bgr=RED_BGR,
        )
        self.marker_state = MarkerState(marker, target=args.marker_target, mode=args.marker_mode, every_n=args.marker_every_n)
        self.recorder = TrajectoryRecorder(
            enabled=args.record,
            root=args.record_root,
            record_images=args.record_images,
            metadata={
                "policy_path": str(self.policy_path),
                "follower_port": args.follower_port,
                "wrist_camera": args.wrist_camera,
                "external_camera": args.external_camera,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "execute": args.execute,
                "marker_target": args.marker_target,
                "marker_mode": args.marker_mode,
                "marker_every_n": args.marker_every_n,
                "manual_gripper_close_pos": args.manual_gripper_close_pos,
                "manual_gripper_min_pos": args.manual_gripper_min_pos,
                "manual_gripper_hold_s": args.manual_gripper_hold_s,
                "manual_gripper_shake_amp": args.manual_gripper_shake_amp,
                "manual_gripper_shake_hz": args.manual_gripper_shake_hz,
                "manual_gripper_step": args.manual_gripper_step,
                "manual_down_servo": args.manual_down_servo,
                "manual_down_offset": args.manual_down_offset,
                "nudge_enabled": args.nudge_enabled,
                "nudge_step": args.nudge_step,
                "nudge_max": args.nudge_max,
                "extra_nudge_enabled": args.extra_nudge_enabled,
                "extra_nudge_step": args.extra_nudge_step,
                "open_gripper_key": args.open_gripper_key,
                "manual_open_pos": args.manual_open_pos,
                "manual_open_hold_s": args.manual_open_hold_s,
                "manual_open_stutter_steps": args.manual_open_stutter_steps,
                "manual_open_stutter_pause_s": args.manual_open_stutter_pause_s,
                "manual_open_step": args.manual_open_step,
                "lock_gripper_until_open_key": args.lock_gripper_until_open_key,
                "locked_gripper_pos": args.locked_gripper_pos,
            },
        )

    def add_action_offset(self, servo_id: int, delta: int, label: str) -> None:
        if servo_id not in SERVO_IDS:
            print(f"\nIgnored {label}: servo {servo_id} is not in {SERVO_IDS}.")
            return
        current = self.action_offsets.get(servo_id, 0)
        updated = clamp_offset(current + int(delta), max_abs=self.args.nudge_max)
        if updated == 0:
            self.action_offsets.pop(servo_id, None)
        else:
            self.action_offsets[servo_id] = updated
        if not self.args.quiet_controls:
            print(f"\nNudge {label}: servo{servo_id} offset={updated:+d} ticks")

    def reset_action_offsets(self) -> None:
        self.action_offsets.clear()
        if not self.args.quiet_controls:
            print("\nNudge offsets cleared.")

    def handle_nudge_key(self, key: str | None) -> bool:
        if not self.args.nudge_enabled or not key:
            return False
        if key == self.args.nudge_left_key:
            self.add_action_offset(self.args.nudge_left_servo, self.args.nudge_left_delta, "left")
            return True
        if key == self.args.nudge_right_key:
            self.add_action_offset(self.args.nudge_left_servo, -self.args.nudge_left_delta, "right")
            return True
        if key == self.args.nudge_forward_key:
            self.add_action_offset(self.args.nudge_forward_servo, self.args.nudge_forward_delta, "forward")
            return True
        if key == self.args.nudge_back_key:
            back_servo = self.args.nudge_back_servo
            if back_servo is None:
                self.add_action_offset(self.args.nudge_forward_servo, -self.args.nudge_forward_delta, "back")
            else:
                back_delta = self.args.nudge_back_delta
                if back_delta is None:
                    back_delta = self.args.nudge_forward_delta
                self.add_action_offset(back_servo, back_delta, "back")
            return True
        if key == self.args.nudge_reset_key:
            self.reset_action_offsets()
            return True
        if not self.args.extra_nudge_enabled:
            return False
        if key == self.args.nudge_up_key:
            self.add_action_offset(self.args.nudge_up_servo, self.args.nudge_up_delta, "up")
            return True
        if key == self.args.nudge_down_key:
            self.add_action_offset(self.args.nudge_up_servo, -self.args.nudge_up_delta, "down")
            return True
        if key == self.args.nudge_wrist_up_key:
            self.add_action_offset(self.args.nudge_wrist_servo, self.args.nudge_wrist_delta, "wrist up")
            return True
        if key == self.args.nudge_wrist_down_key:
            self.add_action_offset(self.args.nudge_wrist_servo, -self.args.nudge_wrist_delta, "wrist down")
            return True
        if key == self.args.nudge_roll_left_key:
            self.add_action_offset(self.args.nudge_roll_servo, self.args.nudge_roll_delta, "roll left")
            return True
        if key == self.args.nudge_roll_right_key:
            self.add_action_offset(self.args.nudge_roll_servo, -self.args.nudge_roll_delta, "roll right")
            return True
        return False

    def apply_action_offsets(self, action: np.ndarray) -> np.ndarray:
        if not self.action_offsets:
            return action
        adjusted = action.copy()
        for servo_id, offset in self.action_offsets.items():
            idx = SERVO_IDS.index(servo_id)
            adjusted[idx] = clamp_follower_target(servo_id, int(round(adjusted[idx] + offset)))
        return adjusted.astype(np.int32)

    def trigger_gripper_close(self) -> None:
        now = time.monotonic()
        self.manual_gripper_started = now
        self.manual_gripper_until = now + max(0.0, float(self.args.manual_gripper_hold_s))
        if self.args.lock_gripper_until_open_key:
            start = self.locked_gripper_pos
            if start is None:
                start = self.last_sent.get(6)
            if start is None and self.last_state is not None:
                start = int(self.last_state[5])
            if start is None:
                start = clamp_follower_target(6, int(self.args.manual_open_pos))

            step = max(1, int(self.args.manual_gripper_step))
            target = clamp_manual_gripper_target(
                int(start) - step,
                min_pos=self.args.manual_gripper_min_pos,
            )
            self.manual_gripper_start_pos = int(start)
            self.manual_gripper_target_pos = int(target)
            self.locked_gripper_pos = int(target)
            if not self.args.quiet_controls:
                print(f"\nManual gripper close step: servo6 {start} -> {target}")
            return

        target = clamp_manual_gripper_target(
            self.args.manual_gripper_close_pos,
            min_pos=self.args.manual_gripper_min_pos,
        )
        if self.arm is not None and self.args.execute:
            self.arm.move_to(
                6,
                target,
                speed=self.args.manual_gripper_speed,
                acc=self.args.manual_gripper_acc,
            )
            self.last_sent[6] = target
        if not self.args.quiet_controls:
            print(
                f"\nManual gripper close: servo6 -> {target} "
                f"speed={self.args.manual_gripper_speed} "
                f"shake=±{self.args.manual_gripper_shake_amp} "
                f"down=servo{self.args.manual_down_servo}{self.args.manual_down_offset:+d} "
                f"for {self.args.manual_gripper_hold_s:.2f}s; ACT continues."
            )

    def manual_close_target(self, elapsed_s: float) -> int:
        start = self.manual_gripper_start_pos
        target = self.manual_gripper_target_pos
        if target is None:
            target = clamp_manual_gripper_target(
                self.args.manual_gripper_close_pos,
                min_pos=self.args.manual_gripper_min_pos,
            )
        if start is None:
            return int(target)

        hold_s = max(0.001, float(self.args.manual_gripper_hold_s))
        progress = min(1.0, max(0.0, elapsed_s / hold_s))
        eased = 1.0 - (1.0 - progress) * (1.0 - progress)
        steps = max(1, int(self.args.manual_open_stutter_steps))
        stepped = math.ceil(eased * steps) / steps
        return clamp_manual_gripper_target(
            int(round(start + (int(target) - start) * stepped)),
            min_pos=self.args.manual_gripper_min_pos,
        )

    def trigger_gripper_open(self) -> None:
        now = time.monotonic()
        self.manual_open_started = now
        self.manual_open_until = now + max(0.0, float(self.args.manual_open_hold_s))
        self.manual_open_triggered = True
        start = self.last_sent.get(6)
        if start is None and self.last_state is not None:
            start = int(self.last_state[5])
        if self.args.lock_gripper_until_open_key and self.locked_gripper_pos is not None:
            start = int(self.locked_gripper_pos)

        full_open = clamp_follower_target(6, int(self.args.manual_open_pos))
        if start is None:
            start = full_open

        if self.args.lock_gripper_until_open_key:
            step = max(1, int(self.args.manual_open_step))
            target = min(full_open, int(start) + step)
            target = clamp_follower_target(6, target)
            self.locked_gripper_pos = target
        else:
            target = full_open

        self.manual_open_start_pos = int(start)
        self.manual_open_target_pos = int(target)
        if not self.args.quiet_controls:
            print(
                f"\nManual gripper open: servo6 -> {target} "
                f"speed={self.args.manual_open_speed} "
                f"stutter_steps={self.args.manual_open_stutter_steps} "
                f"for {self.args.manual_open_hold_s:.2f}s; ACT continues."
            )

    def manual_open_target(self, elapsed_s: float) -> int:
        start = self.manual_open_start_pos
        target = self.manual_open_target_pos
        if target is None:
            target = clamp_follower_target(6, int(self.args.manual_open_pos))
        if start is None:
            return int(target)

        hold_s = max(0.001, float(self.args.manual_open_hold_s))
        pause_s = max(0.0, float(self.args.manual_open_stutter_pause_s))
        active_s = max(0.001, hold_s - pause_s)
        progress = min(1.0, max(0.0, elapsed_s / active_s))
        eased = 1.0 - (1.0 - progress) * (1.0 - progress)
        steps = max(1, int(self.args.manual_open_stutter_steps))
        stepped = math.ceil(eased * steps) / steps
        return clamp_follower_target(6, int(round(start + (int(target) - start) * stepped)))

    def read_observation(self, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        args = self.args
        assert self.wrist_camera is not None
        assert self.external_camera is not None
        assert self.arm is not None
        assert self.marker_state is not None

        wrist_rgb = read_rgb(
            self.wrist_camera,
            width=args.width,
            height=args.height,
            rotate_180=not args.no_wrist_rotate_180,
            name="wrist",
        )
        external_rgb = read_rgb(
            self.external_camera,
            width=args.width,
            height=args.height,
            rotate_180=args.external_rotate_180,
            name="external",
        )
        external_rgb = self.marker_state.apply(external_rgb, frame_index)
        follower_positions = sanitize_follower_positions(
            read_follower_positions(self.arm, SERVO_IDS),
            last_state=self.last_state,
            max_jump=args.state_max_jump,
            range_margin=args.state_range_margin,
        )
        if self.last_state is None and len(follower_positions) < 6:
            raise RuntimeError(f"Could not read all follower positions: {follower_positions}")
        state = positions_to_array(follower_positions, fallback=self.last_state)
        return state, wrist_rgb, external_rgb

    def select_action(self, state: np.ndarray, wrist_rgb: np.ndarray, external_rgb: np.ndarray) -> np.ndarray:
        assert self.torch is not None
        assert self.policy is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None

        observation = {
            "observation.state": self.torch.from_numpy(state).float(),
            "observation.images.wrist": image_to_tensor(self.torch, wrist_rgb),
            "observation.images.external": image_to_tensor(self.torch, external_rgb),
        }
        batch = self.preprocessor.process_observation(observation)
        with self.torch.no_grad():
            normalized_action = self.policy.select_action(batch)
            action_tensor = self.postprocessor.process_action(normalized_action)
        action = clamp_action(action_tensor.squeeze(0).detach().cpu().numpy())
        action = self.apply_action_offsets(action)
        now = time.monotonic()
        if now < self.manual_gripper_until and not self.args.lock_gripper_until_open_key:
            action = apply_grasp_assist(action, self.args, now - self.manual_gripper_started)
        if self.args.lock_gripper_until_open_key:
            if self.locked_gripper_pos is None:
                if self.args.locked_gripper_pos is not None:
                    self.locked_gripper_pos = clamp_follower_target(6, int(self.args.locked_gripper_pos))
                else:
                    self.locked_gripper_pos = clamp_follower_target(6, int(round(state[5])))
                if not self.args.quiet_controls:
                    print(f"\nGripper locked until open key: servo6={self.locked_gripper_pos}")

            if now < self.manual_gripper_until:
                action[5] = self.manual_close_target(now - self.manual_gripper_started)
            elif now < self.manual_open_until:
                action[5] = self.manual_open_target(now - self.manual_open_started)
            else:
                action[5] = int(self.locked_gripper_pos)
        elif now < self.manual_open_until:
            action[5] = self.manual_open_target(now - self.manual_open_started)
        return limit_action_step(
            action,
            last_sent=self.last_sent,
            target_max_step=self.args.target_max_step,
            gripper_target_max_step=self.args.gripper_target_max_step,
        )

    def handle_runtime_key(self, key: str | None) -> str | None:
        args = self.args
        if not key:
            return None
        if key == args.quit_key:
            print("\nQuit requested.")
            return "quit"
        if args.next_stage_key and key == args.next_stage_key:
            if not args.quiet_controls:
                print("\nNext stage requested.")
            return "next"
        if args.close_gripper_key and key == args.close_gripper_key:
            self.trigger_gripper_close()
        elif args.open_gripper_key and key == args.open_gripper_key:
            self.trigger_gripper_open()
        else:
            self.handle_nudge_key(key)
        return None

    def run(self) -> str:
        self.setup()
        args = self.args
        assert self.arm is not None
        assert self.marker_state is not None

        frame_count = max(1, int(round(args.duration_s * args.fps)))
        next_time = time.monotonic()
        run_started = next_time
        print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: duration={args.duration_s}s fps={args.fps}")
        key_hints = [f"{args.quit_key}=quit"]
        if args.close_gripper_key:
            key_hints.append(f"{args.close_gripper_key}=close gripper while ACT continues")
        if args.open_gripper_key:
            key_hints.append(f"{args.open_gripper_key}=open gripper while ACT continues")
        if args.next_stage_key:
            key_hints.append(f"{args.next_stage_key}=next stage")
        if args.nudge_enabled:
            hint = (
                f"{args.nudge_forward_key}/{args.nudge_back_key}=forward/back "
                f"{args.nudge_left_key}/{args.nudge_right_key}=left/right "
                f"{args.nudge_reset_key}=clear offset"
            )
            if args.extra_nudge_enabled:
                hint += (
                    f" {args.nudge_up_key}/{args.nudge_down_key}=up/down "
                    f"{args.nudge_wrist_up_key}/{args.nudge_wrist_down_key}=wrist "
                    f"{args.nudge_roll_left_key}/{args.nudge_roll_right_key}=roll"
                )
            key_hints.append(hint)
        print("Keys: " + ", ".join(key_hints))

        with RawTerminal():
            try:
                for frame_index in range(frame_count):
                    reason = self.handle_runtime_key(read_key())
                    if reason is not None:
                        self.stop_reason = reason
                        break

                    state, wrist_rgb, external_rgb = self.read_observation(frame_index)
                    action = self.select_action(state, wrist_rgb, external_rgb)
                    send_action(
                        self.arm,
                        action,
                        last_sent=self.last_sent,
                        speed=args.speed,
                        acc=args.acc,
                        min_delta=args.min_delta,
                        execute=args.execute,
                    )
                    self.last_state = state
                    now = time.monotonic()
                    manual_active = now < self.manual_gripper_until
                    manual_elapsed = (now - self.manual_gripper_started) if manual_active else None
                    if self.recorder is not None:
                        self.recorder.record_frame(
                            frame_index=frame_index + 1,
                            elapsed_s=now - run_started,
                            state=state,
                            action=action,
                            marker_meta=self.marker_state.last_meta,
                            manual_grasp_active=manual_active,
                            manual_grasp_elapsed_s=manual_elapsed,
                            wrist_rgb=wrist_rgb,
                            external_rgb=external_rgb,
                        )

                    if not args.quiet_controls and (frame_index + 1) % max(1, int(args.fps)) == 0:
                        print(f"POLICY frame={frame_index + 1}/{frame_count} state={state.astype(int).tolist()} action={action.tolist()}")

                    if args.show:
                        preview_key = show_preview(
                            wrist_rgb=wrist_rgb,
                            external_rgb=external_rgb,
                            mode=self.mode,
                            frame_index=frame_index + 1,
                            frame_count=frame_count,
                            marker_meta=self.marker_state.last_meta,
                        )
                        reason = self.handle_runtime_key(preview_key)
                        if reason is not None:
                            self.stop_reason = reason
                            break

                    next_time += 1.0 / args.fps
                    sleep_s = next_time - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        self.overruns += 1
            except KeyboardInterrupt:
                print("\nStopped by user.")
                self.stop_reason = "quit"
            finally:
                self.close(frame_count)
        return self.stop_reason

    def close(self, frame_count: int) -> None:
        if self.overruns:
            print(f"Policy loop overran target fps on {self.overruns}/{frame_count} frames.")
        if self.wrist_camera is not None:
            self.wrist_camera.release()
        if self.external_camera is not None:
            self.external_camera.release()
        if self.args.show:
            cv2.destroyAllWindows()
        if self.recorder is not None:
            self.recorder.close()
        if self.arm is not None and hasattr(self.arm, "_ser"):
            self.arm._ser.close()


def main() -> None:
    DualCameraGraspRunner(build_arg_parser().parse_args()).run()


if __name__ == "__main__":
    main()
