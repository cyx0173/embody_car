#!/usr/bin/env python3
"""Read-only probe for Feetech servo registers.

This script does not write any servo register and does not move motors. It is
intended for diagnosing unexpected angle-limit / position-wrap behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arm_control import ServoController
from orange_grasp_config import DEFAULT_FOLLOWER_PORT, parse_ids


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "calibration_runs"

# Common Feetech/SCS register addresses. Names are intentionally conservative:
# different firmware variants sometimes document the same locations differently.
REGISTER_SPECS = {
    "model": (3, 2),
    "id": (5, 1),
    "baud": (6, 1),
    "min_angle_limit_candidate": (9, 2),
    "max_angle_limit_candidate": (11, 2),
    "max_temperature": (13, 1),
    "max_voltage": (14, 1),
    "min_voltage": (15, 1),
    "max_torque": (16, 2),
    "phase": (18, 1),
    "unloading_condition": (19, 1),
    "led_alarm_condition": (20, 1),
    "p_coefficient": (21, 1),
    "d_coefficient": (22, 1),
    "i_coefficient": (23, 1),
    "minimum_startup_force": (24, 2),
    "cw_dead_zone": (26, 1),
    "ccw_dead_zone": (27, 1),
    "protection_current": (28, 2),
    "angular_resolution": (30, 1),
    "offset_candidate": (31, 2),
    "mode": (33, 1),
    "torque_enable": (40, 1),
    "goal_position": (42, 2),
    "goal_time": (44, 2),
    "goal_speed": (46, 2),
    "lock": (55, 1),
    "present_position": (56, 2),
    "present_speed": (58, 2),
}


def read_register(arm: ServoController, servo_id: int, address: int, length: int) -> int | None:
    value = arm._send_read(servo_id, address, length)
    if value < 0:
        return None
    return int(value)


def dump_registers(arm: ServoController, servo_id: int, *, start: int, end: int) -> dict[int, int | None]:
    values: dict[int, int | None] = {}
    for address in range(start, end + 1):
        values[address] = read_register(arm, servo_id, address, 1)
        time.sleep(0.005)
    return values


def probe_servo(arm: ServoController, servo_id: int, *, dump_start: int, dump_end: int) -> dict:
    named: dict[str, int | None] = {}
    for name, (address, length) in REGISTER_SPECS.items():
        named[name] = read_register(arm, servo_id, address, length)
        time.sleep(0.005)

    return {
        "servo_id": servo_id,
        "named": named,
        "raw_1byte_dump": dump_registers(arm, servo_id, start=dump_start, end=dump_end),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only servo register probe.")
    parser.add_argument("--port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--ids", default="3,4,5")
    parser.add_argument("--dump-start", type=int, default=0)
    parser.add_argument("--dump-end", type=int, default=70)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    servo_ids = parse_ids(args.ids)
    arm = ServoController(port=args.port)
    results = []
    try:
        for servo_id in servo_ids:
            result = probe_servo(
                arm,
                servo_id,
                dump_start=args.dump_start,
                dump_end=args.dump_end,
            )
            results.append(result)
            named = result["named"]
            print(f"\nservo {servo_id}:")
            for key in (
                "model",
                "id",
                "min_angle_limit_candidate",
                "max_angle_limit_candidate",
                "offset_candidate",
                "mode",
                "torque_enable",
                "goal_position",
                "present_position",
            ):
                print(f"  {key}: {named.get(key)}")
    finally:
        if hasattr(arm, "_ser"):
            arm._ser.close()

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port": args.port,
        "results": results,
    }
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"servo_register_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved register probe: {output_path}")


if __name__ == "__main__":
    main()
