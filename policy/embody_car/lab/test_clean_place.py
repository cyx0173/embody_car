#!/usr/bin/env python3
"""Test only the marked left/right bowl place policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from arm_control import ServoController
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    GRIPPER_FOLLOWER_OPEN,
    ORANGE_PLACE_POLICY_DURATION_S,
    ORANGE_POLICY_FPS,
    ORANGE_POLICY_SPEED,
)
from run_grasp_clean import DualCameraGraspRunner, build_arg_parser as build_runner_arg_parser

DEFAULT_PLACE_POLICY_PATH = PROJECT_DIR / "place_model"
PLACE_POLICY_KEYWORDS = ("place", "bowl")
DEFAULT_LOAD_CLOSE_POS = 580
DEFAULT_LOAD_GRIPPER_SPEED = 3200
DEFAULT_LOAD_GRIPPER_ACC = 120
DEFAULT_LOAD_SETTLE_S = 0.6
DEFAULT_GRIPPER_STUTTER_STEPS = 5
DEFAULT_GRIPPER_STUTTER_PAUSE_S = 0.08
DEFAULT_OPEN_STEP = 260


def move_gripper_stutter(
    arm: ServoController,
    *,
    start_pos: int,
    end_pos: int,
    speed: int,
    acc: int,
    steps: int,
    pause_s: float,
) -> None:
    steps = max(1, int(steps))
    for step in range(1, steps + 1):
        ratio = step / steps
        target = int(round(start_pos + (end_pos - start_pos) * ratio))
        arm.move_to(6, target, speed=speed, acc=acc)
        if pause_s > 0:
            time.sleep(pause_s)


def normalize_bowl(value: str) -> str:
    text = value.strip().lower()
    if text in {"l", "left", "leftmost"}:
        return "left"
    if text in {"r", "right", "rightmost"}:
        return "right"
    raise ValueError(f"target bowl must be left/right/l/r, got {value!r}")


def bowl_marker_choice(target_bowl: str) -> str:
    return "leftmost" if target_bowl == "left" else "rightmost"


def read_policy_repo_id(policy_path: Path) -> str:
    train_config_path = policy_path.expanduser().resolve() / "train_config.json"
    if not train_config_path.exists():
        return ""
    with train_config_path.open("r", encoding="utf-8") as f:
        train_config = json.load(f)
    return str(train_config.get("dataset", {}).get("repo_id") or "")


def looks_like_place_policy(repo_id: str) -> bool:
    text = repo_id.lower()
    return any(keyword in text for keyword in PLACE_POLICY_KEYWORDS)


def prepare_loaded_gripper(args: argparse.Namespace) -> None:
    if not args.prepare_gripper:
        return

    print(
        f"Opening gripper for loading: servo6 -> {args.load_open_pos} "
        f"speed={args.load_speed} acc={args.load_acc}"
    )
    if args.execute:
        arm = ServoController(port=args.follower_port)
        try:
            arm.move_to(6, args.load_open_pos, speed=args.load_speed, acc=args.load_acc)
            input("Put the object into the open gripper, then press Enter to close and start place ACT.")
            print(
                f"Closing gripper to hold object: servo6 -> {args.load_close_pos} "
                f"speed={args.load_speed} acc={args.load_acc} "
                f"stutter_steps={args.gripper_stutter_steps}"
            )
            move_gripper_stutter(
                arm,
                start_pos=args.load_open_pos,
                end_pos=args.load_close_pos,
                speed=args.load_speed,
                acc=args.load_acc,
                steps=args.gripper_stutter_steps,
                pause_s=args.gripper_stutter_pause_s,
            )
            if args.load_settle_s > 0:
                time.sleep(args.load_settle_s)
        finally:
            if hasattr(arm, "_ser"):
                arm._ser.close()
    else:
        input("[DRY RUN] Put the object into the gripper, then press Enter to continue.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test place_model only. The object should already be held by the gripper."
    )
    parser.add_argument("target_bowl", choices=("l", "r", "left", "right"))
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_PLACE_POLICY_PATH)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--external-camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=ORANGE_POLICY_FPS)
    parser.add_argument("--speed", type=int, default=ORANGE_POLICY_SPEED)
    parser.add_argument("--duration-s", type=float, default=ORANGE_PLACE_POLICY_DURATION_S)
    parser.add_argument("--marker-model", type=Path, default=BASE_DIR / "yolo11s.pt")
    parser.add_argument("--marker-device", default="auto")
    parser.add_argument("--marker-target", default="bowl")
    parser.add_argument("--nudge-step", type=int, default=45)
    parser.add_argument("--nudge-max", type=int, default=320)
    parser.add_argument("--open-key", default="t")
    parser.add_argument("--open-pos", type=int, default=GRIPPER_FOLLOWER_OPEN)
    parser.add_argument("--open-hold-s", type=float, default=1.2)
    parser.add_argument("--open-step", type=int, default=DEFAULT_OPEN_STEP)
    parser.add_argument("--prepare-gripper", dest="prepare_gripper", action="store_true", default=True)
    parser.add_argument("--no-prepare-gripper", dest="prepare_gripper", action="store_false")
    parser.add_argument("--load-open-pos", type=int, default=GRIPPER_FOLLOWER_OPEN)
    parser.add_argument("--load-close-pos", type=int, default=DEFAULT_LOAD_CLOSE_POS)
    parser.add_argument("--load-speed", type=int, default=DEFAULT_LOAD_GRIPPER_SPEED)
    parser.add_argument("--load-acc", type=int, default=DEFAULT_LOAD_GRIPPER_ACC)
    parser.add_argument("--load-settle-s", type=float, default=DEFAULT_LOAD_SETTLE_S)
    parser.add_argument("--gripper-stutter-steps", type=int, default=DEFAULT_GRIPPER_STUTTER_STEPS)
    parser.add_argument("--gripper-stutter-pause-s", type=float, default=DEFAULT_GRIPPER_STUTTER_PAUSE_S)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--strict-place-policy", action="store_true")
    parser.add_argument("--show", dest="show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--execute", dest="execute", action="store_true", default=True)
    parser.add_argument("--dry-run", dest="execute", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_bowl = normalize_bowl(args.target_bowl)
    policy_path = args.policy_path.expanduser().resolve()
    repo_id = read_policy_repo_id(policy_path)
    if repo_id and not looks_like_place_policy(repo_id):
        message = (
            "WARNING: policy metadata does not look like a place/bowl policy.\n"
            f"policy_path={policy_path}\n"
            f"dataset repo_id={repo_id}\n"
            "I will still run it because the metadata may be stale or copied."
        )
        if args.strict_place_policy:
            raise RuntimeError(message)
        print(message)

    runner_args = build_runner_arg_parser().parse_args([])
    runner_args.policy_path = policy_path
    runner_args.follower_port = args.follower_port
    runner_args.wrist_camera = args.wrist_camera
    runner_args.external_camera = args.external_camera
    runner_args.fps = args.fps
    runner_args.speed = args.speed
    runner_args.duration_s = args.duration_s
    runner_args.marker_target = args.marker_target
    runner_args.marker_choose = bowl_marker_choice(target_bowl)
    runner_args.marker_model = str(args.marker_model)
    runner_args.marker_device = args.marker_device
    runner_args.show = args.show
    runner_args.execute = args.execute
    runner_args.record = args.record
    runner_args.record_images = args.record_images and args.record
    runner_args.quiet_controls = True

    runner_args.close_gripper_key = ""
    runner_args.nudge_enabled = True
    runner_args.nudge_step = args.nudge_step
    runner_args.nudge_max = args.nudge_max
    runner_args.nudge_left_delta = args.nudge_step
    runner_args.nudge_forward_delta = args.nudge_step
    runner_args.open_gripper_key = args.open_key
    runner_args.manual_open_pos = args.open_pos
    runner_args.manual_open_hold_s = args.open_hold_s
    runner_args.manual_open_stutter_steps = args.gripper_stutter_steps
    runner_args.manual_open_stutter_pause_s = args.gripper_stutter_pause_s
    runner_args.manual_open_step = args.open_step
    runner_args.lock_gripper_until_open_key = True

    print(f"Testing place_model only: target_bowl={target_bowl}, marker_choose={runner_args.marker_choose}")
    print(f"Policy path: {policy_path}")
    if repo_id:
        print(f"Policy dataset repo_id: {repo_id}")
    print("Place controls during ACT: w/s=forward/back, a/d=left/right, x=clear offset, t=release gripper, q=quit.")
    if args.no_prompt:
        print("Skipping load prompt because --no-prompt was passed.")
    else:
        prepare_loaded_gripper(args)
    DualCameraGraspRunner(runner_args).run()


if __name__ == "__main__":
    main()
