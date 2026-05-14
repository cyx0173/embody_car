from __future__ import annotations

import time

import numpy as np

from arm_control import ServoController
from solve_ik import solve_ik


# ====== 修改这里就可以测试不同坐标 ======
# 单位: meter。坐标系: +x 向前/水平向右, +z 向上。
TARGETS = [
    (0.1, 0.1, 0.25),
]

SERIAL_PORT = "/dev/cu.usbmodem5AE60562991"
DRY_RUN = False
ONLY_MOVE_TO_SAFE_READY = False
RETURN_TO_SAFE_READY_BEFORE_EACH_TARGET = True
CONFIRM_BEFORE_EACH_TARGET = True

MOVE_SPEED = 500
MOVE_ACC = 20
SAFE_MOVE_SPEED = 800
SAFE_MOVE_ACC = 40
SAFE_MOVE_SERVO_OVERRIDES = {
    4: {"speed": 400, "acc": 20},
}
SERVO_DELAY_S = 0.25
SAFE_SERVO_DELAY_S = 0.6
SAFE_POSITION_TOLERANCE_TICKS = 80
SAFE_FOLD_TIMEOUT_S = 8.0
SAFE_READY_SETTLE_S = 4.0
FEEDBACK_READS = 20
FEEDBACK_INTERVAL_S = 0.5

# r 太小时在底座正上方，pan 无定义，而且机械臂会进入很别扭的折叠姿态。
MIN_RADIAL_M = 0.10

# 手动调整出来的安全初始姿态。
# 如果这个姿态仍然不舒服，优先只改这里的 tick。
SAFE_READY_TICKS = {
    1: 2222,
    2: 845,
    3: 3115,
    4: 898,
    5: 74,
}
SAFE_FOLD_ORDER = (3, 4)
SAFE_REST_ORDER = (2, 1, 5)
TARGET_MOVE_ORDER = (1, 2, 3, 4, 5)


def validate_target(target: np.ndarray) -> None:
    radial = float(np.hypot(target[0], target[1]))
    if radial < MIN_RADIAL_M:
        raise ValueError(
            f"target radial distance is too small: r={radial:.3f}m. "
            f"Please keep sqrt(x^2 + y^2) >= {MIN_RADIAL_M:.3f}m."
        )


def build_plan() -> list[tuple[np.ndarray, dict[int, int]]]:
    plan = []
    for raw_target in TARGETS:
        target = np.array(raw_target, dtype=float)
        validate_target(target)
        plan.append((target, solve_ik(target)))
    return plan


def read_positions(arm: ServoController) -> dict[int, int]:
    return {servo_id: arm.get_position(servo_id) for servo_id in range(1, 6)}


def move_ticks(
    arm: ServoController,
    ticks: dict[int, int],
    order: tuple[int, ...],
    speed: int,
    acc: int,
    delay_s: float = SERVO_DELAY_S,
    servo_overrides: dict[int, dict[str, int]] | None = None,
) -> None:
    for servo_id in order:
        tick = ticks.get(servo_id)
        if tick is None or tick < 0:
            continue
        override = (servo_overrides or {}).get(servo_id, {})
        move_speed = override.get("speed", speed)
        move_acc = override.get("acc", acc)
        print(f"移动舵机 {servo_id} -> {tick}, speed={move_speed}, acc={move_acc}")
        if not DRY_RUN:
            arm.move_to(servo_id, tick, speed=move_speed, acc=move_acc)
        time.sleep(delay_s)


def wait_and_print_feedback(arm: ServoController) -> None:
    if DRY_RUN:
        return
    print("\n等待到位并读取反馈：")
    for i in range(FEEDBACK_READS):
        time.sleep(FEEDBACK_INTERVAL_S)
        print(f"{i + 1:02d}: {read_positions(arm)}")


def wait_until_ticks(
    arm: ServoController,
    ticks: dict[int, int],
    order: tuple[int, ...],
    timeout_s: float,
    tolerance_ticks: int = SAFE_POSITION_TOLERANCE_TICKS,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        positions = {servo_id: arm.get_position(servo_id) for servo_id in order}
        is_ready = all(
            positions[servo_id] >= 0
            and abs(positions[servo_id] - ticks[servo_id]) <= tolerance_ticks
            for servo_id in order
        )
        print(f"等待舵机 {order} 到位: {positions}")
        if is_ready:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"舵机 {order} 未能在 {timeout_s:.1f}s 内到位: {positions}")
        time.sleep(0.5)


def move_to_safe_ready(arm: ServoController) -> None:
    print("\n先让 3/4 号回到折叠安全姿态")
    move_ticks(
        arm,
        SAFE_READY_TICKS,
        SAFE_FOLD_ORDER,
        speed=SAFE_MOVE_SPEED,
        acc=SAFE_MOVE_ACC,
        delay_s=SAFE_SERVO_DELAY_S,
        servo_overrides=SAFE_MOVE_SERVO_OVERRIDES,
    )
    wait_until_ticks(
        arm,
        SAFE_READY_TICKS,
        SAFE_FOLD_ORDER,
        timeout_s=SAFE_FOLD_TIMEOUT_S,
    )

    print("3/4 号已到位，再移动 2/1/5 回初始")
    move_ticks(
        arm,
        SAFE_READY_TICKS,
        SAFE_REST_ORDER,
        speed=SAFE_MOVE_SPEED,
        acc=SAFE_MOVE_ACC,
        delay_s=SAFE_SERVO_DELAY_S,
        servo_overrides=SAFE_MOVE_SERVO_OVERRIDES,
    )
    print(f"等待安全姿态稳定 {SAFE_READY_SETTLE_S:.1f}s")
    time.sleep(SAFE_READY_SETTLE_S)


def confirm(message: str) -> None:
    if CONFIRM_BEFORE_EACH_TARGET and not DRY_RUN:
        input(message)


def main() -> None:
    plan = [] if ONLY_MOVE_TO_SAFE_READY else build_plan()

    if ONLY_MOVE_TO_SAFE_READY:
        print("测试计划：只移动到折叠安全姿态")
    else:
        print("测试计划：")
        for target, ticks in plan:
            print(f"  target={target.tolist()} -> ticks={ticks}")

    if DRY_RUN:
        print("\nDRY_RUN=True，只计算不移动。")
        return

    arm = ServoController(port=SERIAL_PORT)
    try:
        print("\n移动前 tick：")
        print(read_positions(arm))

        if ONLY_MOVE_TO_SAFE_READY:
            confirm("确认周围安全后按回车移动到折叠安全姿态...")
            move_to_safe_ready(arm)
            wait_and_print_feedback(arm)
            return

        for target, ticks in plan:
            print(f"\n准备移动到目标坐标: {target.tolist()}")
            print(f"目标 tick: {ticks}")
            confirm("确认周围安全后按回车继续...")

            if RETURN_TO_SAFE_READY_BEFORE_EACH_TARGET:
                move_to_safe_ready(arm)

            print("\n移动到目标姿态")
            move_ticks(
                arm,
                ticks,
                TARGET_MOVE_ORDER,
                speed=MOVE_SPEED,
                acc=MOVE_ACC,
                delay_s=SERVO_DELAY_S,
            )
            wait_and_print_feedback(arm)
    finally:
        arm.brake_all()


if __name__ == "__main__":
    main()
