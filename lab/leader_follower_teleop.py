from __future__ import annotations

import argparse
import time

from arm_control import ServoController
from orange_grasp_config import (
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_PORT,
    FOLLOWER_MAX_TARGET_STEP_TICKS,
    GRIPPER_FOLLOWER_CLOSE,
    GRIPPER_FOLLOWER_OPEN,
    GRIPPER_LEADER_CLOSE,
    GRIPPER_LEADER_OPEN,
    GRIPPER_MAX_TARGET_STEP_TICKS,
    JOINT_MAP,
    LEADER_SPIKE_CONFIRM_FRAMES,
    LEADER_SPIKE_MAX_DELTA_TICKS,
    LOOP_INTERVAL_S,
    MIN_DELTA_TICKS,
    READ_RETRIES,
    READ_RETRY_DELAY_S,
    SERVO_IDS,
    SERVO_TICKS_PER_TURN,
    clamp,
    normalize_servo_reading,
    parse_ids,
)


class LeaderFollowerTeleop:
    def __init__(
        self,
        *,
        leader_port: str,
        follower_port: str,
        speed: int,
        acc: int,
        min_delta: int,
        dry_run: bool,
        direct_raw: bool,
        servo_ids: tuple[int, ...],
        debug_targets: bool,
        debug_unchanged: bool,
        gripper_mode: str,
        gripper_threshold: int,
        leader_max_jump: int,
        spike_confirm_frames: int,
        target_max_step: int,
        gripper_target_max_step: int,
    ) -> None:
        self.leader = ServoController(port=leader_port)
        self.follower = None if dry_run else ServoController(port=follower_port)
        self.speed = int(speed)
        self.acc = int(acc)
        self.min_delta = int(min_delta)
        self.dry_run = dry_run
        self.direct_raw = direct_raw
        self.servo_ids = servo_ids
        self.debug_targets = debug_targets
        self.debug_unchanged = debug_unchanged
        self.gripper_mode = gripper_mode
        self.gripper_threshold = int(gripper_threshold)
        self.leader_max_jump = int(leader_max_jump)
        self.spike_confirm_frames = max(1, int(spike_confirm_frames))
        self.target_max_step = int(target_max_step)
        self.gripper_target_max_step = int(gripper_target_max_step)
        self.last_sent: dict[int, int] = {}
        self.last_leader: dict[int, int] = {}
        self.pending_leader: dict[int, tuple[int, int]] = {}

    def run(self) -> None:
        print("Leader-follower teleop started. Press Ctrl+C to stop.")
        try:
            while True:
                commands = self._read_leader_positions()
                self._send_follower_positions(commands)
                time.sleep(LOOP_INTERVAL_S)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.close()

    def print_positions_loop(self) -> None:
        print("Printing leader/follower positions. Press Ctrl+C to stop.")
        try:
            while True:
                leader = self._read_leader_positions()
                follower = self._read_follower_positions()
                leader_text = ", ".join(
                    f"{sid}:{leader.get(sid, -1)}" for sid in self.servo_ids
                )
                follower_text = ", ".join(
                    f"{sid}:{follower.get(sid, -1)}" for sid in self.servo_ids
                )
                print(f"leader {{{leader_text}}} | follower {{{follower_text}}}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.close()

    def close(self) -> None:
        if self.follower is not None and hasattr(self.follower, "_ser"):
            self.follower._ser.close()
        if hasattr(self.leader, "_ser"):
            self.leader._ser.close()

    def _read_leader_positions(self) -> dict[int, int]:
        return self._read_positions(self.leader)

    def _read_follower_positions(self) -> dict[int, int]:
        if self.follower is None:
            return {}
        return self._read_positions(self.follower)

    def _read_positions(self, arm: ServoController) -> dict[int, int]:
        positions: dict[int, int] = {}
        for servo_id in self.servo_ids:
            pos = self._read_position_with_retries(arm, servo_id)
            if pos < 0:
                continue
            pos = self._normalize_servo_reading(int(pos))
            if arm is self.leader and not self._accept_leader_position(servo_id, pos):
                continue
            positions[servo_id] = pos
        return positions

    def _send_follower_positions(self, positions: dict[int, int]) -> None:
        for servo_id, pos in positions.items():
            target = self._clamp_follower_target(servo_id, pos)
            target = self._limit_target_step(servo_id, target)
            last = self.last_sent.get(servo_id)
            if last is not None and abs(target - last) < self.min_delta:
                if self.debug_targets and self.debug_unchanged:
                    print(self._format_debug_target(servo_id, pos, target, changed=False))
                continue
            self.last_sent[servo_id] = target
            if self.debug_targets:
                print(self._format_debug_target(servo_id, pos, target, changed=True))
            if self.dry_run:
                print(f"[DRY RUN] servo {servo_id}: {target}")
                continue
            if self.follower is None:
                continue
            self.follower.move_to(servo_id, target, speed=self.speed, acc=self.acc)

    def _clamp_follower_target(self, servo_id: int, pos: int) -> int:
        if self.direct_raw:
            target = int(pos)
        else:
            target = self._map_leader_to_follower(servo_id, int(pos))

        cfg = JOINT_MAP[servo_id]
        return clamp(target, cfg["follower_min"], cfg["follower_max"])

    def _map_leader_to_follower(self, servo_id: int, leader_pos: int) -> int:
        if servo_id == 6:
            return self._map_gripper(leader_pos)

        cfg = JOINT_MAP[servo_id]
        leader_min = int(cfg["leader_min"])
        leader_max = int(cfg["leader_max"])
        leader_pos = int(leader_pos)
        follower_min = int(cfg["follower_min"])
        follower_max = int(cfg["follower_max"])

        if leader_max == leader_min:
            return follower_min

        if cfg.get("wrap"):
            return self._map_wrapped_leader_to_follower(
                leader_pos=leader_pos,
                leader_min=leader_min,
                leader_max=leader_max,
                follower_min=follower_min,
                follower_max=follower_max,
            )

        ratio = (leader_pos - leader_min) / (leader_max - leader_min)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(follower_min + ratio * (follower_max - follower_min)))

    def _accept_leader_position(self, servo_id: int, pos: int) -> bool:
        last = self.last_leader.get(servo_id)
        if last is None:
            self.last_leader[servo_id] = pos
            self.pending_leader.pop(servo_id, None)
            return True

        delta = self._leader_position_delta(servo_id, last, pos)
        if delta <= self.leader_max_jump:
            self.last_leader[servo_id] = pos
            self.pending_leader.pop(servo_id, None)
            return True

        pending_pos, count = self.pending_leader.get(servo_id, (pos, 0))
        pending_delta = self._leader_position_delta(servo_id, pending_pos, pos)
        if pending_delta <= self.leader_max_jump:
            count += 1
        else:
            pending_pos = pos
            count = 1
        self.pending_leader[servo_id] = (pending_pos, count)

        if self.debug_targets:
            print(
                f"servo {servo_id}: rejected leader spike "
                f"last={last} raw={pos} delta={delta} confirm={count}/{self.spike_confirm_frames}"
            )

        if count >= self.spike_confirm_frames:
            self.last_leader[servo_id] = pending_pos
            self.pending_leader.pop(servo_id, None)
            if self.debug_targets:
                print(f"servo {servo_id}: accepted sustained leader jump -> {pending_pos}")
            return True

        return False

    def _leader_position_delta(self, servo_id: int, a: int, b: int) -> int:
        if JOINT_MAP.get(servo_id, {}).get("wrap") or servo_id == 6:
            return self._circular_distance(a, b)
        return abs(int(a) - int(b))

    def _limit_target_step(self, servo_id: int, target: int) -> int:
        last = self.last_sent.get(servo_id)
        if last is None:
            return target

        max_step = self.gripper_target_max_step if servo_id == 6 else self.target_max_step
        if max_step <= 0:
            return target

        delta = int(target) - int(last)
        if abs(delta) <= max_step:
            return target

        limited = int(last) + (max_step if delta > 0 else -max_step)
        if self.debug_targets:
            print(
                f"servo {servo_id}: limited follower target "
                f"requested={target} sent={limited} last={last}"
            )
        return limited

    def _normalize_servo_reading(self, pos: int) -> int:
        return normalize_servo_reading(pos)

    def _map_gripper(self, leader_pos: int) -> int:
        if self.gripper_mode == "binary":
            if self._is_gripper_closer_to_close(leader_pos):
                return GRIPPER_FOLLOWER_CLOSE
            return GRIPPER_FOLLOWER_OPEN

        leader_span = GRIPPER_LEADER_CLOSE - GRIPPER_LEADER_OPEN
        follower_span = GRIPPER_FOLLOWER_CLOSE - GRIPPER_FOLLOWER_OPEN
        if leader_span == 0:
            return GRIPPER_FOLLOWER_OPEN
        ratio = (leader_pos - GRIPPER_LEADER_OPEN) / leader_span
        ratio = max(0.0, min(1.0, ratio))
        return int(round(GRIPPER_FOLLOWER_OPEN + ratio * follower_span))

    def _is_gripper_closer_to_close(self, leader_pos: int) -> bool:
        if self.gripper_threshold >= 0:
            return leader_pos <= self.gripper_threshold
        distance_to_open = self._circular_distance(leader_pos, GRIPPER_LEADER_OPEN)
        distance_to_close = self._circular_distance(leader_pos, GRIPPER_LEADER_CLOSE)
        return distance_to_close < distance_to_open

    def _circular_distance(self, a: int, b: int) -> int:
        delta = abs((int(a) - int(b)) % SERVO_TICKS_PER_TURN)
        return min(delta, SERVO_TICKS_PER_TURN - delta)

    def _map_wrapped_leader_to_follower(
        self,
        *,
        leader_pos: int,
        leader_min: int,
        leader_max: int,
        follower_min: int,
        follower_max: int,
    ) -> int:
        span = (leader_max - leader_min) % SERVO_TICKS_PER_TURN
        delta = (leader_pos - leader_min) % SERVO_TICKS_PER_TURN

        if span == 0:
            return follower_min
        if delta > span:
            distance_to_min = min(delta, SERVO_TICKS_PER_TURN - delta)
            distance_to_max = min(delta - span, SERVO_TICKS_PER_TURN - (delta - span))
            delta = 0 if distance_to_min <= distance_to_max else span

        ratio = delta / span
        return int(round(follower_min + ratio * (follower_max - follower_min)))

    def _read_position_with_retries(self, arm: ServoController, servo_id: int) -> int:
        for _ in range(READ_RETRIES):
            pos = arm.get_position(servo_id)
            if pos >= 0:
                return int(pos)
            time.sleep(READ_RETRY_DELAY_S)
        return -1

    def _format_debug_target(
        self,
        servo_id: int,
        leader_pos: int,
        target: int,
        *,
        changed: bool,
    ) -> str:
        suffix = "" if changed else " unchanged"
        if servo_id != 6:
            return f"servo {servo_id}: leader={leader_pos} -> follower={target}{suffix}"

        distance_to_open = self._circular_distance(leader_pos, GRIPPER_LEADER_OPEN)
        distance_to_close = self._circular_distance(leader_pos, GRIPPER_LEADER_CLOSE)
        state = "close" if distance_to_close < distance_to_open else "open"
        return (
            f"servo 6: leader={leader_pos} -> follower={target} "
            f"state={state} d_open={distance_to_open} d_close={distance_to_close}{suffix}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple leader arm to follower arm teleoperation without cameras."
    )
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--speed", type=int, default=1800)
    parser.add_argument("--acc", type=int, default=45)
    parser.add_argument("--min-delta", type=int, default=MIN_DELTA_TICKS)
    parser.add_argument(
        "--servo-ids",
        default=",".join(str(sid) for sid in SERVO_IDS),
        help="Comma-separated servo IDs to teleoperate.",
    )
    parser.add_argument(
        "--debug-targets",
        action="store_true",
        help="Print leader readings and mapped follower targets when commands change.",
    )
    parser.add_argument(
        "--debug-unchanged",
        action="store_true",
        help="With --debug-targets, also print readings when the mapped target does not change.",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("binary", "linear"),
        default="linear",
        help="Use binary open/close mapping or continuous linear mapping for servo 6.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=int,
        default=-1,
        help=(
            "Leader 6 threshold for binary gripper mapping. "
            "Use -1 to choose open/close by circular distance."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--leader-max-jump",
        type=int,
        default=LEADER_SPIKE_MAX_DELTA_TICKS,
        help="Reject one-frame leader readings that jump more than this many ticks.",
    )
    parser.add_argument(
        "--spike-confirm-frames",
        type=int,
        default=LEADER_SPIKE_CONFIRM_FRAMES,
        help="Accept a large leader jump only after it persists for this many frames.",
    )
    parser.add_argument(
        "--target-max-step",
        type=int,
        default=FOLLOWER_MAX_TARGET_STEP_TICKS,
        help="Maximum follower target change per control loop for servos 1-5. Use 0 to disable.",
    )
    parser.add_argument(
        "--gripper-target-max-step",
        type=int,
        default=GRIPPER_MAX_TARGET_STEP_TICKS,
        help="Maximum follower target change per control loop for servo 6. Use 0 to disable.",
    )
    parser.add_argument(
        "--direct-raw",
        action="store_true",
        help="Copy leader raw ticks directly without JOINT_MAP calibration.",
    )
    parser.add_argument(
        "--print-positions",
        action="store_true",
        help="Only print leader/follower positions for calibration.",
    )
    args = parser.parse_args()

    teleop = LeaderFollowerTeleop(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        speed=args.speed,
        acc=args.acc,
        min_delta=args.min_delta,
        dry_run=args.dry_run,
        direct_raw=args.direct_raw,
        servo_ids=parse_ids(args.servo_ids),
        debug_targets=args.debug_targets,
        debug_unchanged=args.debug_unchanged,
        gripper_mode=args.gripper_mode,
        gripper_threshold=args.gripper_threshold,
        leader_max_jump=args.leader_max_jump,
        spike_confirm_frames=args.spike_confirm_frames,
        target_max_step=args.target_max_step,
        gripper_target_max_step=args.gripper_target_max_step,
    )
    if args.print_positions:
        teleop.print_positions_loop()
    else:
        teleop.run()


if __name__ == "__main__":
    main()
