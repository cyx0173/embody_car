#!/usr/bin/env python3
"""Debug follower servo 6 gripper control.

Default mode only reads registers. Pass --execute to command open/close ticks.
"""

from __future__ import annotations

import argparse
import time

from arm_control import ServoController
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    GRIPPER_FOLLOWER_CLOSE,
    GRIPPER_FOLLOWER_OPEN,
    normalize_servo_reading,
)
from restore_follower_calibration import (
    ADDR_HOMING_OFFSET,
    ADDR_LOCK,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_POSITION,
    decode_sign_magnitude,
)


SERVO_ID = 6
SIGN_BIT_HOMING_OFFSET = 11
ADDR_GOAL_POSITION = 42


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


def read_status(arm: ServoController) -> dict[str, int | None]:
    offset_raw = read_u16(arm, SERVO_ID, ADDR_HOMING_OFFSET)
    present = read_u16(arm, SERVO_ID, ADDR_PRESENT_POSITION)
    return {
        "min_limit": read_u16(arm, SERVO_ID, ADDR_MIN_POSITION_LIMIT),
        "max_limit": read_u16(arm, SERVO_ID, ADDR_MAX_POSITION_LIMIT),
        "homing_offset_raw": offset_raw,
        "homing_offset": None
        if offset_raw is None
        else decode_sign_magnitude(offset_raw, SIGN_BIT_HOMING_OFFSET),
        "mode": read_u8(arm, SERVO_ID, ADDR_OPERATING_MODE),
        "lock": read_u8(arm, SERVO_ID, ADDR_LOCK),
        "goal_position": read_u16(arm, SERVO_ID, ADDR_GOAL_POSITION),
        "present_position": None if present is None else normalize_servo_reading(present),
    }


def move_and_report(
    arm: ServoController,
    *,
    target: int,
    speed: int,
    acc: int,
    settle_s: float,
) -> None:
    before = read_status(arm)
    print(f"\ntarget={target}")
    print(f"  before: {before}")
    arm.move_to(SERVO_ID, target, speed=speed, acc=acc)
    time.sleep(settle_s)
    after = read_status(arm)
    print(f"  after:  {after}")
    if after["present_position"] is not None:
        print(f"  error:  {after['present_position'] - int(target)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug follower gripper servo 6.")
    parser.add_argument("--port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--open", type=int, default=GRIPPER_FOLLOWER_OPEN)
    parser.add_argument("--close", type=int, default=GRIPPER_FOLLOWER_CLOSE)
    parser.add_argument("--speed", type=int, default=700)
    parser.add_argument("--acc", type=int, default=25)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    arm = ServoController(port=args.port)
    try:
        print(f"Follower gripper debug: port={args.port}, servo={SERVO_ID}")
        print(f"Initial status: {read_status(arm)}")
        if not args.execute:
            print("Dry run only. Add --execute to command open/close.")
            return
        print("Executing open -> close -> open. Be ready to cut power if it binds.")
        move_and_report(arm, target=args.open, speed=args.speed, acc=args.acc, settle_s=args.settle_s)
        move_and_report(arm, target=args.close, speed=args.speed, acc=args.acc, settle_s=args.settle_s)
        move_and_report(arm, target=args.open, speed=args.speed, acc=args.acc, settle_s=args.settle_s)
    finally:
        if hasattr(arm, "_ser"):
            arm._ser.close()


if __name__ == "__main__":
    main()
