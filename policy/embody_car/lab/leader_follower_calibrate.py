#!/usr/bin/env python3
"""Read-only helper for recalibrating leader/follower joint extrema.

This script never sends movement commands. It only reads servo ticks so the
operator can manually move each joint to its safe endpoints and record them.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from arm_control import ServoController
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    SERVO_IDS,
    SERVO_TICKS_PER_TURN,
    normalize_servo_reading,
    parse_ids,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "calibration_runs"


@dataclass
class Sample:
    servo_id: int
    label: str
    leader: int | None
    follower: int | None
    timestamp_s: float


def read_servo(arm: ServoController | None, servo_id: int, *, samples: int, delay_s: float) -> int | None:
    if arm is None:
        return None

    values: list[int] = []
    for _ in range(max(1, samples)):
        pos = arm.get_position(servo_id)
        if pos >= 0:
            values.append(normalize_servo_reading(int(pos)))
        time.sleep(delay_s)

    if not values:
        return None
    return int(round(sum(values) / len(values)))


def read_positions(
    *,
    leader: ServoController | None,
    follower: ServoController | None,
    servo_ids: tuple[int, ...],
    samples: int,
    delay_s: float,
) -> tuple[dict[int, int | None], dict[int, int | None]]:
    leader_positions: dict[int, int | None] = {}
    follower_positions: dict[int, int | None] = {}
    for servo_id in servo_ids:
        leader_positions[servo_id] = read_servo(leader, servo_id, samples=samples, delay_s=delay_s)
        follower_positions[servo_id] = read_servo(follower, servo_id, samples=samples, delay_s=delay_s)
    return leader_positions, follower_positions


def format_positions(prefix: str, positions: dict[int, int | None], servo_ids: tuple[int, ...]) -> str:
    text = ", ".join(
        f"{servo_id}:{positions.get(servo_id) if positions.get(servo_id) is not None else -1}"
        for servo_id in servo_ids
    )
    return f"{prefix} {{{text}}}"


def watch_loop(
    *,
    leader: ServoController | None,
    follower: ServoController | None,
    servo_ids: tuple[int, ...],
    samples: int,
    delay_s: float,
    interval_s: float,
) -> None:
    print("Watching current ticks. Press Ctrl+C to stop.")
    try:
        while True:
            leader_positions, follower_positions = read_positions(
                leader=leader,
                follower=follower,
                servo_ids=servo_ids,
                samples=samples,
                delay_s=delay_s,
            )
            print(
                f"{format_positions('leader', leader_positions, servo_ids)} | "
                f"{format_positions('follower', follower_positions, servo_ids)}"
            )
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nStopped.")


def capture_extrema(
    *,
    leader: ServoController | None,
    follower: ServoController | None,
    servo_ids: tuple[int, ...],
    samples: int,
    delay_s: float,
) -> list[Sample]:
    print("Read-only calibration capture.")
    print("For each joint, manually move the arm(s) to the requested endpoint, then press Enter.")
    print("Use Ctrl+C if a joint is unsafe or you want to stop.")

    captured: list[Sample] = []
    for servo_id in servo_ids:
        print(f"\n=== Servo {servo_id} ===")
        for label in ("low", "mid", "high"):
            input(
                f"Move servo {servo_id} to the physical {label.upper()} reference "
                "on the connected arm(s), then press Enter..."
            )
            leader_pos = read_servo(leader, servo_id, samples=samples, delay_s=delay_s)
            follower_pos = read_servo(follower, servo_id, samples=samples, delay_s=delay_s)
            sample = Sample(
                servo_id=servo_id,
                label=label,
                leader=leader_pos,
                follower=follower_pos,
                timestamp_s=time.time(),
            )
            captured.append(sample)
            print(
                f"captured servo {servo_id} {label}: "
                f"leader={leader_pos if leader_pos is not None else -1}, "
                f"follower={follower_pos if follower_pos is not None else -1}"
            )
    return captured


def summarize_samples(samples: list[Sample]) -> None:
    print("\nSummary:")
    grouped: dict[int, dict[str, Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.servo_id, {})[sample.label] = sample

    for servo_id in sorted(grouped):
        values = grouped[servo_id]
        low = values.get("low")
        mid = values.get("mid")
        high = values.get("high")
        if low is None or high is None:
            continue
        print(f"\nservo {servo_id}:")
        print(
            "  leader: "
            f"low={low.leader if low.leader is not None else -1}, "
            f"mid={mid.leader if mid and mid.leader is not None else -1}, "
            f"high={high.leader if high.leader is not None else -1}"
        )
        print(
            "  follower: "
            f"low={low.follower if low.follower is not None else -1}, "
            f"mid={mid.follower if mid and mid.follower is not None else -1}, "
            f"high={high.follower if high.follower is not None else -1}"
        )
        if low.leader is not None and high.leader is not None and low.follower is not None and high.follower is not None:
            follower_wrap = should_mark_follower_wrap(
                low=low.follower,
                mid=mid.follower if mid else None,
                high=high.follower,
            )
            print("  candidate JOINT_MAP entry:")
            suffix = ', "follower_wrap": True' if follower_wrap else ""
            print(
                "  "
                f"{servo_id}: "
                "{"
                f"\"leader_min\": {low.leader}, "
                f"\"leader_max\": {high.leader}, "
                f"\"follower_min\": {low.follower}, "
                f"\"follower_max\": {high.follower}"
                f"{suffix}"
                "},"
            )
            if follower_wrap:
                span = (high.follower - low.follower) % SERVO_TICKS_PER_TURN
                print(f"  note: follower range wraps over 4096/0, span={span} ticks.")


def should_mark_follower_wrap(*, low: int, mid: int | None, high: int) -> bool:
    if high < low:
        return True
    if mid is None:
        return False
    simple_min = min(low, high)
    simple_max = max(low, high)
    return not (simple_min <= mid <= simple_max)


def save_samples(samples: list[Sample], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"leader_follower_extrema_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": [asdict(sample) for sample in samples],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only leader/follower extrema calibration helper."
    )
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--servo-ids", default=",".join(str(sid) for sid in SERVO_IDS))
    parser.add_argument("--leader-only", action="store_true")
    parser.add_argument("--follower-only", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Continuously print current ticks.")
    parser.add_argument("--capture", action="store_true", help="Interactively capture low/mid/high references.")
    parser.add_argument("--samples", type=int, default=3, help="Read this many samples and average them.")
    parser.add_argument("--sample-delay-s", type=float, default=0.02)
    parser.add_argument("--watch-interval-s", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.leader_only and args.follower_only:
        raise ValueError("Use only one of --leader-only or --follower-only.")

    servo_ids = parse_ids(args.servo_ids)
    leader = None
    follower = None
    try:
        if not args.follower_only:
            leader = ServoController(port=args.leader_port)
        if not args.leader_only:
            follower = ServoController(port=args.follower_port)

        if args.watch or not args.capture:
            watch_loop(
                leader=leader,
                follower=follower,
                servo_ids=servo_ids,
                samples=args.samples,
                delay_s=args.sample_delay_s,
                interval_s=args.watch_interval_s,
            )
            return

        samples = capture_extrema(
            leader=leader,
            follower=follower,
            servo_ids=servo_ids,
            samples=args.samples,
            delay_s=args.sample_delay_s,
        )
        summarize_samples(samples)
        output_path = save_samples(samples, args.output_dir)
        print(f"\nSaved calibration capture: {output_path}")
    finally:
        if follower is not None and hasattr(follower, "_ser"):
            follower._ser.close()
        if leader is not None and hasattr(leader, "_ser"):
            leader._ser.close()


if __name__ == "__main__":
    main()
