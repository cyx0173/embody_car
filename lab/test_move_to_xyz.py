from __future__ import annotations

import argparse
import math
import time

from Angle_config import angles_to_ticks
from arm_control import ServoController
from solve_ik import forward_kinematics_check, position_to_arm


MOVE_ORDER = (2, 3, 4, 1, 5)
SERVO_OVERRIDES = {
    2: {"speed": 900, "acc": 45},
    4: {"speed": 400, "acc": 20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test moving the arm to a target xyz coordinate in the arm base frame."
    )
    parser.add_argument("x", type=float, help="target x in meters")
    parser.add_argument("y", type=float, help="target y in meters")
    parser.add_argument("z", type=float, help="target z in meters")
    parser.add_argument("--port", default="/dev/cu.usbmodem5AE60562991", help="servo serial port")
    parser.add_argument("--speed", type=int, default=800, help="servo move speed")
    parser.add_argument("--acc", type=int, default=40, help="servo acceleration")
    parser.add_argument("--delay", type=float, default=0.25, help="delay between servo commands")
    parser.add_argument("--tolerance", type=int, default=60, help="allowed final tick error")
    parser.add_argument("--wait-timeout", type=float, default=8.0, help="seconds to wait for each servo")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for each servo to reach target")
    parser.add_argument("--brake-on-exit", action="store_true", help="brake all servos before exiting")
    parser.add_argument("--yes", action="store_true", help="move without interactive confirmation")
    parser.add_argument("--dry-run", action="store_true", help="only print the IK result, do not move")
    return parser.parse_args()


def wait_until_servo_reaches(
    arm: ServoController,
    servo_id: int,
    target_tick: int,
    tolerance: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        current = arm.get_position(servo_id)
        error = abs(current - target_tick) if current >= 0 else 99999
        print(f"  等待舵机 {servo_id}: current={current}, target={target_tick}, error={error}")
        if current >= 0 and error <= tolerance:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.4)


def main() -> None:
    args = parse_args()
    target = (args.x, args.y, args.z)

    joints = position_to_arm(target)
    ticks = angles_to_ticks(joints)
    degrees = {name: math.degrees(rad) for name, rad in joints.items()}

    print("目标坐标 [m]:", target)
    print("关节角度 [deg]:")
    for name, deg in degrees.items():
        print(f"  {name}: {deg:.2f}")
    print("目标 tick:", ticks)
    forward_kinematics_check(joints, target)

    if args.dry_run:
        print("dry-run 模式：只计算，不移动机械臂")
        return

    if not args.yes:
        answer = input("确认移动机械臂？输入 y 后回车继续: ").strip().lower()
        if answer != "y":
            print("已取消移动")
            return

    arm = ServoController(port=args.port)
    move_failed = False
    print("移动前 tick:")
    for servo_id in MOVE_ORDER:
        print(f"  {servo_id}: {arm.get_position(servo_id)}")

    for servo_id in MOVE_ORDER:
        tick = ticks.get(servo_id)
        if tick is None:
            continue
        override = SERVO_OVERRIDES.get(servo_id, {})
        speed = override.get("speed", args.speed)
        acc = override.get("acc", args.acc)
        print(f"移动舵机 {servo_id} -> {tick}, speed={speed}, acc={acc}")
        arm.move_to(servo_id, tick, speed=speed, acc=acc)
        time.sleep(args.delay)
        if not args.no_wait:
            reached = wait_until_servo_reaches(
                arm,
                servo_id,
                tick,
                tolerance=args.tolerance,
                timeout_s=args.wait_timeout,
            )
            if not reached:
                move_failed = True
                print(f"警告：舵机 {servo_id} 没有到位，先停止后续移动")
                break

    print("移动后 tick:")
    for servo_id in MOVE_ORDER:
        print(f"  {servo_id}: {arm.get_position(servo_id)}")

    if args.brake_on_exit or move_failed:
        print("执行 brake_all()")
        arm.brake_all()


if __name__ == "__main__":
    main()
