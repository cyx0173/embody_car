from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from Angle_config import SERVO_CALIBRATION
from arm_control import ServoController


DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_SERVO_IDS = (1, 2, 3, 4, 5, 6)
DEFAULT_PORT = "/dev/cu.usbmodem5AE60562991"
JOINT_DEBUG_STEPS = (5, 10, 25, 50, 100, 200)
ROTATE_CHOICES = ("none", "clockwise", "counterclockwise", "180")


@dataclass(frozen=True)
class ServoLimit:
    name: str
    low: int
    high: int


def build_servo_limits() -> dict[int, ServoLimit]:
    limits: dict[int, ServoLimit] = {}
    for name, cfg in SERVO_CALIBRATION.items():
        limits[int(cfg["id"])] = ServoLimit(
            name=name,
            low=int(cfg["range_min"]),
            high=int(cfg["range_max"]),
        )
    return limits


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def choose_test_target(position: int, limit: ServoLimit, step: int, margin: int) -> int:
    safe_low = limit.low + margin
    safe_high = limit.high - margin
    if safe_low >= safe_high:
        safe_low, safe_high = limit.low, limit.high

    if position + step <= safe_high:
        return position + step
    if position - step >= safe_low:
        return position - step
    midpoint = (safe_low + safe_high) // 2
    return clamp(midpoint, safe_low, safe_high)


def parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def test_servos(
    *,
    port: str,
    baudrate: int,
    servo_ids: Iterable[int],
    step: int,
    speed: int,
    acc: int,
    settle: float,
    margin: int,
) -> bool:
    limits = build_servo_limits()
    arm = ServoController(port=port, baudrate=baudrate)
    ok = True

    try:
        print("\n=== Servo test ===")
        print(f"port={port}, baudrate={baudrate}, step={step}, speed={speed}, acc={acc}")

        for servo_id in servo_ids:
            limit = limits.get(servo_id)
            if limit is None:
                print(f"[SKIP] servo {servo_id}: no configured limit")
                ok = False
                continue

            position = arm.get_position(servo_id)
            if position < 0:
                print(f"[FAIL] servo {servo_id} ({limit.name}): read position failed")
                ok = False
                continue

            if not (limit.low <= position <= limit.high):
                print(
                    f"[SKIP] servo {servo_id} ({limit.name}): "
                    f"position {position} outside configured range {limit.low}-{limit.high}"
                )
                ok = False
                continue

            target = choose_test_target(position, limit, step, margin)
            print(
                f"[MOVE] servo {servo_id} ({limit.name}): "
                f"{position} -> {target} -> {position}"
            )

            arm.move_to(servo_id, target, speed=speed, acc=acc)
            time.sleep(settle)
            target_readback = arm.get_position(servo_id)

            arm.move_to(servo_id, position, speed=speed, acc=acc)
            time.sleep(settle)
            return_readback = arm.get_position(servo_id)

            target_error = abs(target_readback - target) if target_readback >= 0 else None
            return_error = abs(return_readback - position) if return_readback >= 0 else None
            print(
                f"[READ] servo {servo_id}: target={target_readback} "
                f"(err={target_error}), return={return_readback} (err={return_error})"
            )

            if target_readback < 0 or return_readback < 0:
                ok = False

        print("Servo test finished.\n")
        return ok
    finally:
        arm._ser.close()


def move_servo_to_position(
    *,
    port: str,
    baudrate: int,
    servo_id: int,
    position: int,
    speed: int,
    acc: int,
    settle: float,
    no_clamp: bool,
) -> bool:
    limits = build_servo_limits()
    limit = limits.get(servo_id)
    target = position

    if limit is None:
        if not no_clamp:
            print(
                f"[FAIL] servo {servo_id}: no configured limit. "
                "Use --no-clamp if you really want to send this position."
            )
            return False
        limit_name = "unknown"
    else:
        limit_name = limit.name
        if no_clamp:
            if not (0 <= target <= 4095):
                print(f"[FAIL] servo {servo_id}: target {target} outside raw range 0-4095")
                return False
        else:
            clamped = clamp(target, limit.low, limit.high)
            if clamped != target:
                print(
                    f"[WARN] servo {servo_id} ({limit.name}): target {target} "
                    f"clamped to {clamped} by configured range {limit.low}-{limit.high}"
                )
            target = clamped

    arm = ServoController(port=port, baudrate=baudrate)
    try:
        print("\n=== Move one servo ===")
        current = arm.get_position(servo_id)
        print(f"servo {servo_id} ({limit_name}) current={current}, target={target}")
        if current < 0:
            print(f"[FAIL] servo {servo_id}: read position failed")
            return False

        arm.move_to(servo_id, target, speed=speed, acc=acc)
        time.sleep(settle)
        readback = arm.get_position(servo_id)
        error = abs(readback - target) if readback >= 0 else None
        print(f"servo {servo_id} readback={readback}, error={error}")
        return readback >= 0
    finally:
        arm._ser.close()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(w * height / h))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def draw_label(frame: np.ndarray, text: str, ok: bool = True) -> np.ndarray:
    color = (0, 220, 0) if ok else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def read_servo_positions(arm: ServoController, servo_ids: Iterable[int]) -> dict[int, int]:
    return {servo_id: arm.get_position(servo_id) for servo_id in servo_ids}


def next_step(current: int, direction: int) -> int:
    steps = list(JOINT_DEBUG_STEPS)
    if current not in steps:
        steps.append(current)
        steps.sort()
    index = steps.index(current)
    return steps[clamp(index + direction, 0, len(steps) - 1)]


def draw_joint_debug_panel(
    frame: np.ndarray,
    *,
    selected_servo: int,
    positions: dict[int, int],
    limits: dict[int, ServoLimit],
    step: int,
) -> np.ndarray:
    panel_height = 112
    y0 = frame.shape[0] - panel_height
    cv2.rectangle(frame, (0, y0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

    selected_limit = limits.get(selected_servo)
    selected_name = selected_limit.name if selected_limit else "unknown"
    selected_position = positions.get(selected_servo, -1)
    lines = [
        f"selected servo {selected_servo} ({selected_name})  pos={selected_position}  step={step}",
        "keys: 1-6 select | a/d move -/+ | [/ ] step | r reread | s save | q/esc quit",
    ]

    position_text = []
    for servo_id in sorted(positions):
        limit = limits.get(servo_id)
        name = limit.name if limit else "unknown"
        prefix = "*" if servo_id == selected_servo else " "
        position_text.append(f"{prefix}{servo_id}:{name}={positions[servo_id]}")
    lines.append("  ".join(position_text))

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, y0 + 28 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def show_cameras(
    *,
    left_index: int,
    right_index: int,
    width: int,
    height: int,
    seconds: float,
    save_path: str | None,
    left_rotation: str,
    right_rotation: str,
) -> bool:
    cap_left = open_camera(left_index, width, height)
    cap_right = open_camera(right_index, width, height)
    window_name = "Hardware camera check [left | right]"
    deadline = time.monotonic() + seconds if seconds > 0 else None
    frame_count = 0
    ok = True

    print("\n=== Camera preview ===")
    print(f"left={left_index}, right={right_index}, size={width}x{height}")
    print("Press q or ESC to quit. Press s to save one combined frame.")

    try:
        while True:
            cap_left.grab()
            cap_right.grab()
            ok_left, left = cap_left.retrieve()
            ok_right, right = cap_right.retrieve()

            if not ok_left or left is None:
                left = np.zeros((height, width, 3), dtype=np.uint8)
                draw_label(left, f"camera {left_index}: read failed", ok=False)
                ok = False
            else:
                left = rotate_frame(left, left_rotation)
                draw_label(left, f"camera {left_index}")

            if not ok_right or right is None:
                right = np.zeros((height, width, 3), dtype=np.uint8)
                draw_label(right, f"camera {right_index}: read failed", ok=False)
                ok = False
            else:
                right = rotate_frame(right, right_rotation)
                draw_label(right, f"camera {right_index}")

            preview_height = min(left.shape[0], right.shape[0], 540)
            left_preview = resize_to_height(left, preview_height)
            right_preview = resize_to_height(right, preview_height)
            combined = np.hstack([left_preview, right_preview])

            cv2.imshow(window_name, combined)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_path or "hardware_camera_check.png"
                cv2.imwrite(path, combined)
                print(f"Saved camera snapshot: {path}")

            if deadline is not None and time.monotonic() >= deadline:
                break

        print(f"Camera preview finished, frames={frame_count}.\n")
        return ok and frame_count > 0
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyWindow(window_name)


def debug_joints_with_cameras(
    *,
    port: str,
    baudrate: int,
    left_index: int,
    right_index: int,
    width: int,
    height: int,
    speed: int,
    acc: int,
    step: int,
    settle: float,
    save_path: str | None,
    left_rotation: str,
    right_rotation: str,
) -> bool:
    limits = build_servo_limits()
    servo_ids = list(DEFAULT_SERVO_IDS)
    selected_servo = servo_ids[0]
    positions: dict[int, int] = {}
    ok = True

    arm = ServoController(port=port, baudrate=baudrate)
    cap_left = open_camera(left_index, width, height)
    cap_right = open_camera(right_index, width, height)
    window_name = "Joint debug [left | right]"

    print("\n=== Joint debug with cameras ===")
    print(f"servo port={port}, left camera={left_index}, right camera={right_index}")
    print("Keys:")
    print("  1-6  select servo")
    print("  a/d  move selected servo by -step/+step")
    print("  [/]  decrease/increase step")
    print("  r    reread all servo positions")
    print("  s    save one combined frame")
    print("  q    quit")

    try:
        positions = read_servo_positions(arm, servo_ids)

        while True:
            cap_left.grab()
            cap_right.grab()
            ok_left, left = cap_left.retrieve()
            ok_right, right = cap_right.retrieve()

            if not ok_left or left is None:
                left = np.zeros((height, width, 3), dtype=np.uint8)
                draw_label(left, f"camera {left_index}: read failed", ok=False)
                ok = False
            else:
                left = rotate_frame(left, left_rotation)
                draw_label(left, f"camera {left_index}")

            if not ok_right or right is None:
                right = np.zeros((height, width, 3), dtype=np.uint8)
                draw_label(right, f"camera {right_index}: read failed", ok=False)
                ok = False
            else:
                right = rotate_frame(right, right_rotation)
                draw_label(right, f"camera {right_index}")

            preview_height = min(left.shape[0], right.shape[0], 540)
            left_preview = resize_to_height(left, preview_height)
            right_preview = resize_to_height(right, preview_height)
            combined = np.hstack([left_preview, right_preview])
            draw_joint_debug_panel(
                combined,
                selected_servo=selected_servo,
                positions=positions,
                limits=limits,
                step=step,
            )
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                selected_servo = int(chr(key))
                print(f"selected servo {selected_servo}")
                continue
            if key == ord("["):
                step = next_step(step, -1)
                print(f"step={step}")
                continue
            if key == ord("]"):
                step = next_step(step, 1)
                print(f"step={step}")
                continue
            if key == ord("r"):
                positions = read_servo_positions(arm, servo_ids)
                print(f"positions={positions}")
                continue
            if key == ord("s"):
                path = save_path or "joint_debug_camera_check.png"
                cv2.imwrite(path, combined)
                print(f"Saved camera snapshot: {path}")
                continue
            if key not in (ord("a"), ord("d")):
                continue

            direction = -1 if key == ord("a") else 1
            current = positions.get(selected_servo, -1)
            if current < 0:
                current = arm.get_position(selected_servo)
            limit = limits[selected_servo]
            target = clamp(current + direction * step, limit.low, limit.high)
            if target == current:
                print(
                    f"servo {selected_servo} already at limit near {current} "
                    f"({limit.low}-{limit.high})"
                )
                continue

            arm.move_to(selected_servo, target, speed=speed, acc=acc)
            time.sleep(settle)
            readback = arm.get_position(selected_servo)
            positions[selected_servo] = readback
            print(
                f"servo {selected_servo} ({limit.name}): "
                f"{current} -> {target}, readback={readback}"
            )

        return ok
    finally:
        arm._ser.close()
        cap_left.release()
        cap_right.release()
        cv2.destroyWindow(window_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test six servos and preview two cameras for the robot."
    )
    parser.add_argument("--skip-servos", action="store_true", help="Only run camera preview.")
    parser.add_argument("--skip-cameras", action="store_true", help="Only run servo test.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Servo serial port.")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Servo serial baudrate.")
    parser.add_argument(
        "--servo-ids",
        default=",".join(str(i) for i in DEFAULT_SERVO_IDS),
        help="Comma-separated servo IDs to test.",
    )
    parser.add_argument("--step", type=int, default=80, help="Servo test movement in ticks.")
    parser.add_argument("--speed", type=int, default=400, help="Servo movement speed.")
    parser.add_argument("--acc", type=int, default=25, help="Servo movement acceleration.")
    parser.add_argument("--settle", type=float, default=0.8, help="Seconds to wait after each move.")
    parser.add_argument("--margin", type=int, default=40, help="Keep this many ticks away from limits.")
    parser.add_argument(
        "--joint-debug",
        action="store_true",
        help="Open both cameras and interactively adjust servo positions.",
    )
    parser.add_argument(
        "--joint-step",
        type=int,
        default=25,
        help="Initial tick step for --joint-debug.",
    )
    parser.add_argument(
        "--joint-settle",
        type=float,
        default=0.15,
        help="Seconds to wait after each interactive joint move.",
    )
    parser.add_argument("--move-servo", type=int, default=None, help="Move one servo ID to a target tick.")
    parser.add_argument("--move-position", type=int, default=None, help="Target tick for --move-servo.")
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Do not clamp --move-position to the configured servo range.",
    )
    parser.add_argument(
        "--continue-after-move",
        action="store_true",
        help="After --move-servo, continue with the normal servo/camera checks.",
    )
    parser.add_argument("--left-camera", type=int, default=0, help="Left/base camera index.")
    parser.add_argument("--right-camera", type=int, default=1, help="Right/hand camera index.")
    parser.add_argument(
        "--left-rotation",
        choices=ROTATE_CHOICES,
        default="none",
        help="Rotation for the left camera preview.",
    )
    parser.add_argument(
        "--right-rotation",
        choices=ROTATE_CHOICES,
        default="clockwise",
        help="Rotation for the right camera preview. Default assumes camera 1 is on the hand.",
    )
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument(
        "--camera-seconds",
        type=float,
        default=0.0,
        help="Preview duration. 0 means run until q or ESC.",
    )
    parser.add_argument("--save-camera-path", default=None, help="Path used when pressing s.")
    args = parser.parse_args()

    all_ok = True

    if (args.move_servo is None) != (args.move_position is None):
        parser.error("--move-servo and --move-position must be used together.")

    if args.joint_debug:
        all_ok = debug_joints_with_cameras(
            port=args.port,
            baudrate=args.baudrate,
            left_index=args.left_camera,
            right_index=args.right_camera,
            width=args.camera_width,
            height=args.camera_height,
            speed=args.speed,
            acc=args.acc,
            step=args.joint_step,
            settle=args.joint_settle,
            save_path=args.save_camera_path,
            left_rotation=args.left_rotation,
            right_rotation=args.right_rotation,
        ) and all_ok
        if all_ok:
            print("Hardware check completed.")
        else:
            raise SystemExit("Hardware check completed with warnings or failures.")
        return

    if args.move_servo is not None:
        all_ok = move_servo_to_position(
            port=args.port,
            baudrate=args.baudrate,
            servo_id=args.move_servo,
            position=args.move_position,
            speed=args.speed,
            acc=args.acc,
            settle=args.settle,
            no_clamp=args.no_clamp,
        ) and all_ok
        if not args.continue_after_move:
            if all_ok:
                print("Hardware check completed.")
            else:
                raise SystemExit("Hardware check completed with warnings or failures.")
            return

    if not args.skip_servos:
        all_ok = test_servos(
            port=args.port,
            baudrate=args.baudrate,
            servo_ids=parse_ids(args.servo_ids),
            step=args.step,
            speed=args.speed,
            acc=args.acc,
            settle=args.settle,
            margin=args.margin,
        ) and all_ok

    if not args.skip_cameras:
        all_ok = show_cameras(
            left_index=args.left_camera,
            right_index=args.right_camera,
            width=args.camera_width,
            height=args.camera_height,
            seconds=args.camera_seconds,
            save_path=args.save_camera_path,
            left_rotation=args.left_rotation,
            right_rotation=args.right_rotation,
        ) and all_ok

    if all_ok:
        print("Hardware check completed.")
    else:
        raise SystemExit("Hardware check completed with warnings or failures.")


if __name__ == "__main__":
    main()
