#!/usr/bin/env python3
"""Read raw bytes from Feetech servo control-table RAM/register addresses.

This is read-only: it sends INST_READ packets only and never moves motors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import serial

from orange_grasp_config import DEFAULT_FOLLOWER_PORT, parse_ids


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "calibration_runs"


def checksum(payload: list[int]) -> int:
    return (~sum(payload)) & 0xFF


def read_bytes(
    ser: serial.Serial,
    *,
    servo_id: int,
    address: int,
    length: int,
    settle_s: float,
) -> bytes | None:
    ser.reset_input_buffer()
    params = [int(address), int(length)]
    packet_len = 2 + len(params)
    payload = [int(servo_id), packet_len, 2] + params
    packet = [0xFF, 0xFF] + payload + [checksum(payload)]
    ser.write(bytes(packet))
    ser.flush()
    time.sleep(settle_s)
    response = ser.read(int(length) + 6)

    if len(response) < int(length) + 6:
        return None
    if response[0] != 0xFF or response[1] != 0xFF:
        return None
    if ((~sum(response[2:-1])) & 0xFF) != response[-1]:
        return None
    return bytes(response[5:-1])


def dump_servo(
    ser: serial.Serial,
    *,
    servo_id: int,
    start: int,
    length: int,
    chunk_size: int,
    settle_s: float,
) -> dict[int, int | None]:
    values: dict[int, int | None] = {}
    end = int(start) + int(length)
    address = int(start)
    while address < end:
        read_len = min(int(chunk_size), end - address)
        data = read_bytes(
            ser,
            servo_id=servo_id,
            address=address,
            length=read_len,
            settle_s=settle_s,
        )
        if data is None:
            for offset in range(read_len):
                values[address + offset] = None
        else:
            for offset, byte in enumerate(data):
                values[address + offset] = int(byte)
        address += read_len
        time.sleep(settle_s)
    return values


def print_dump(servo_id: int, values: dict[int, int | None], *, columns: int = 8) -> None:
    print(f"\nservo {servo_id}:")
    addresses = sorted(values)
    for row_start in range(0, len(addresses), columns):
        row_addresses = addresses[row_start : row_start + columns]
        label = f"{row_addresses[0]:03d}-{row_addresses[-1]:03d}"
        hex_values = []
        dec_values = []
        for address in row_addresses:
            value = values[address]
            if value is None:
                hex_values.append("--")
                dec_values.append("---")
            else:
                hex_values.append(f"{value:02X}")
                dec_values.append(f"{value:3d}")
        print(f"  {label}  hex: {' '.join(hex_values)}   dec: {' '.join(dec_values)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump raw servo SRAM/register bytes.")
    parser.add_argument("--port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--ids", default="1,2,3,4,5,6")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--start",
        type=int,
        default=40,
        help="Start address. 40 is the common RAM/control area; use 0 for full table.",
    )
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--settle-s", type=float, default=0.015)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    servo_ids = parse_ids(args.ids)
    print(
        f"port={args.port}, ids={servo_ids}, "
        f"range={args.start}..{args.start + args.length - 1}"
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port": args.port,
        "baudrate": args.baudrate,
        "start": args.start,
        "length": args.length,
        "servos": {},
    }
    with serial.Serial(args.port, args.baudrate, timeout=0.05) as ser:
        for servo_id in servo_ids:
            values = dump_servo(
                ser,
                servo_id=servo_id,
                start=args.start,
                length=args.length,
                chunk_size=args.chunk_size,
                settle_s=args.settle_s,
            )
            print_dump(servo_id, values)
            payload["servos"][str(servo_id)] = values

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"servo_sram_dump_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved SRAM/register dump: {output_path}")


if __name__ == "__main__":
    main()
