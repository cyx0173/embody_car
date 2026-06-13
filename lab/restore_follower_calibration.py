#!/usr/bin/env python3
"""Restore follower servo calibration registers from a LeRobot-style JSON.

Default mode is a dry run. Pass --execute to write EEPROM/register values.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arm_control import ServoController
from orange_grasp_config import DEFAULT_FOLLOWER_PORT


DEFAULT_CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 1788,
        "range_min": 715,
        "range_max": 3466,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1706,
        "range_min": 822,
        "range_max": 3226,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": 1712,
        "range_min": 908,
        "range_max": 3123,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": 1345,
        "range_min": 845,
        "range_max": 3176,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 1900,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1313,
        "range_min": 1507,
        "range_max": 3026,
    },
}

ORANGE_GRIPPER_CALIBRATION = {
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1313,
        "range_min": 800,
        "range_max": 2302,
    },
}

ADDR_MIN_POSITION_LIMIT = 9
ADDR_MAX_POSITION_LIMIT = 11
ADDR_HOMING_OFFSET = 31
ADDR_OPERATING_MODE = 33
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56

SIGN_BIT_HOMING_OFFSET = 11


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    value = int(value)
    if value < 0:
        return abs(value) | (1 << int(sign_bit))
    return value


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    value = int(value)
    sign_mask = 1 << int(sign_bit)
    if value & sign_mask:
        return -(value & (sign_mask - 1))
    return value


def read_u16(arm: ServoController, servo_id: int, address: int) -> int | None:
    value = arm._send_read(servo_id, address, 2)
    if value < 0:
        return None
    return int(value)


def read_u8(arm: ServoController, servo_id: int, address: int) -> int | None:
    value = arm._send_read(servo_id, address, 1)
    if value < 0:
        return None
    return int(value)


def write_u16(arm: ServoController, servo_id: int, address: int, value: int) -> None:
    value = int(value)
    arm._send_write(servo_id, address, [value & 0xFF, (value >> 8) & 0xFF])
    time.sleep(0.04)


def write_u8(arm: ServoController, servo_id: int, address: int, value: int) -> None:
    arm._send_write(servo_id, address, [int(value) & 0xFF])
    time.sleep(0.04)


def load_calibration(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_CALIBRATION
    return json.loads(path.read_text(encoding="utf-8"))


def read_current(arm: ServoController, servo_id: int) -> dict[str, int | None]:
    raw_offset = read_u16(arm, servo_id, ADDR_HOMING_OFFSET)
    return {
        "range_min": read_u16(arm, servo_id, ADDR_MIN_POSITION_LIMIT),
        "range_max": read_u16(arm, servo_id, ADDR_MAX_POSITION_LIMIT),
        "homing_offset_raw": raw_offset,
        "homing_offset": None
        if raw_offset is None
        else decode_sign_magnitude(raw_offset, SIGN_BIT_HOMING_OFFSET),
        "mode": read_u8(arm, servo_id, ADDR_OPERATING_MODE),
        "lock": read_u8(arm, servo_id, ADDR_LOCK),
        "present_position": read_u16(arm, servo_id, ADDR_PRESENT_POSITION),
    }


def restore_servo(
    arm: ServoController,
    *,
    name: str,
    cfg: dict,
    execute: bool,
    unlock: bool,
    relock: bool,
) -> None:
    servo_id = int(cfg["id"])
    target_offset = int(cfg["homing_offset"])
    target_offset_raw = encode_sign_magnitude(target_offset, SIGN_BIT_HOMING_OFFSET)
    target_min = int(cfg["range_min"])
    target_max = int(cfg["range_max"])
    target_mode = int(cfg.get("drive_mode", 0))

    before = read_current(arm, servo_id)
    print(f"\n{name} servo {servo_id}")
    print(f"  before: {before}")
    print(
        "  target: "
        f"range_min={target_min}, range_max={target_max}, "
        f"homing_offset={target_offset} raw={target_offset_raw}, mode={target_mode}"
    )

    if not execute:
        return

    if unlock:
        write_u8(arm, servo_id, ADDR_LOCK, 0)
    write_u16(arm, servo_id, ADDR_MIN_POSITION_LIMIT, target_min)
    write_u16(arm, servo_id, ADDR_MAX_POSITION_LIMIT, target_max)
    write_u16(arm, servo_id, ADDR_HOMING_OFFSET, target_offset_raw)
    write_u8(arm, servo_id, ADDR_OPERATING_MODE, target_mode)
    if relock:
        write_u8(arm, servo_id, ADDR_LOCK, 1)

    after = read_current(arm, servo_id)
    print(f"  after:  {after}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore follower calibration registers.")
    parser.add_argument("--port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument(
        "--orange-gripper",
        action="store_true",
        help="Restore only servo 6 to the orange-grasp range: 800..2302.",
    )
    parser.add_argument("--only-ids", default="", help="Comma-separated servo IDs to restore; default is all.")
    parser.add_argument("--execute", action="store_true", help="Actually write registers.")
    parser.add_argument("--no-unlock", action="store_true", help="Do not write lock=0 before restoring.")
    parser.add_argument("--no-relock", action="store_true", help="Do not write lock=1 after restoring.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    calibration = ORANGE_GRIPPER_CALIBRATION if args.orange_gripper else load_calibration(args.calibration_json)
    only_ids = {
        int(part.strip())
        for part in args.only_ids.split(",")
        if part.strip()
    }
    if args.orange_gripper:
        only_ids = {6}

    print("Restore follower calibration")
    print(f"port={args.port}")
    print("mode=EXECUTE" if args.execute else "mode=DRY RUN")

    arm = ServoController(port=args.port)
    try:
        for name, cfg in calibration.items():
            servo_id = int(cfg["id"])
            if only_ids and servo_id not in only_ids:
                continue
            restore_servo(
                arm,
                name=name,
                cfg=cfg,
                execute=args.execute,
                unlock=not args.no_unlock,
                relock=not args.no_relock,
            )
    finally:
        if hasattr(arm, "_ser"):
            arm._ser.close()


if __name__ == "__main__":
    main()
