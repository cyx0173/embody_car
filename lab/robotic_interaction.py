from __future__ import annotations

from ultralytics import YOLO
from interaction.so101 import RobotIKSolver
import numpy as np
from arm_control import ServoController
import json
import cv2
import time
import os
import sys
from pathlib import Path
from solve_ik import solve_ik

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = Path(__file__).resolve().parent


CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_SQUARE_CROP_SIZE = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT)
CAMERA_WORKING_WIDTH = CAMERA_SQUARE_CROP_SIZE // 2
CAMERA_WORKING_HEIGHT = CAMERA_SQUARE_CROP_SIZE // 2

BASE_YOLO_ROTATE_CODE = None
HAND_YOLO_ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE
HAND_CAMERA_ROTATE_CODE = None

# Triangulation returns a point relative to the base-camera optical center.
# This offset converts it into the arm-base coordinate frame. Tune the Z value
# if the arm consistently reaches too high/low for detected targets.
BASE_CAMERA_ORIGIN_IN_ARM_M = np.array([0.0, 0.0, 0.2], dtype=np.float64)

BASE_SEARCH_DEADZONE_PX = 120
BASE_SEARCH_STABLE_FRAMES = 1
BASE_SEARCH_TIMEOUT_S = 90.0
BASE_SEARCH_INTERVAL_S = 0.1
BASE_WHEEL_SEARCH_SPEED = 80
BASE_WHEEL_ALIGN_MIN_SPEED = 10
BASE_WHEEL_ALIGN_MAX_SPEED = 35
BASE_WHEEL_ALIGN_KP = 0.08
BASE_WHEEL_TURN_RIGHT_MODE = 1
BASE_WHEEL_TURN_LEFT_MODE = 0
BASE_WHEEL_STOP_MODE = 4
BASE_ALIGN_REVERSE_MARGIN_PX = 35
BASE_ALIGN_LOST_RESET_FRAMES = 8
BASE_ALIGN_REVERSE_MIN_CONF = 0.6
# Positive wheel speed uses BASE_WHEEL_TURN_RIGHT_MODE. Flip this if the target
# moves farther from center during real-machine alignment.
BASE_ALIGN_DIRECTION_SIGN = -1

SAFE_READY_TICKS = {
    1: 2222,
    2: 845,
    3: 3115,
    4: 898,
    5: 74,
}
DRY_RUN = False
SAFE_FOLD_ORDER = (3, 4)
SAFE_REST_ORDER = (2, 1, 5)
TARGET_MOVE_ORDER = (2, 3, 4, 1, 5)
TARGET_MOVE_ORDER_WITHOUT_WRIST = (2, 3, 1, 5)
TARGET_WRIST_FINAL_ORDER = (4,)
TARGET_SHOULDER_FIRST_ORDER = (2,)
TARGET_ELBOW_PREMOVE_ORDER = (3,)
SAFE_MOVE_SPEED = 800
SAFE_MOVE_ACC = 40
SAFE_MOVE_SERVO_OVERRIDES = {
    4: {"speed": 400, "acc": 20},
}

TARGET_MOVE_SPEED = 800
TARGET_MOVE_ACC = 35

TARGET_MOVE_SERVO_OVERRIDES = {
    2: {"speed": 950, "acc": 35},
    3: {"speed": 900, "acc": 35},
    4: {"speed": 750, "acc": 30},
}

SERVO_DELAY_S = 0.25
SMOOTH_MOVE_ENABLED = True
TARGET_SHOULDER_PREMOVE_ENABLED = True
TARGET_SHOULDER_PREMOVE_RATIO = 0.22
TARGET_SHOULDER_PREMOVE_MAX_TICKS = 280
TARGET_SHOULDER_PREMOVE_MIN_TICKS = 80
TARGET_SHOULDER_PREMOVE_TIMEOUT_S = 5.0
TARGET_ELBOW_PREMOVE_ENABLED = True
TARGET_ELBOW_PREMOVE_RATIO = 0.22
TARGET_ELBOW_PREMOVE_MAX_TICKS = 220
TARGET_ELBOW_PREMOVE_MIN_TICKS = 60
TARGET_ELBOW_PREMOVE_TIMEOUT_S = 5.0
TARGET_WRIST_ASSIST_ENABLED = True
TARGET_WRIST_ASSIST_TICK = 1000
TARGET_WRIST_ASSIST_TIMEOUT_S = 5.0
SMOOTH_STEP_TICKS = 140
SMOOTH_STEP_SERVO_OVERRIDES = {
    1: 180,
    2: 120,
    3: 130,
    4: 120,
    5: 180,
}
SMOOTH_STEP_DELAY_S = 0.04
TARGET_FINAL_SETTLE_S = 0.4
TARGET_FINAL_REISSUE_COUNT = 1
TARGET_FINAL_REISSUE_DELAY_S = 0.25
TARGET_SETTLE_TIMEOUT_S = 15.0
TARGET_SETTLE_INTERVAL_S = 0.3
TARGET_REISSUE_ERROR_TICKS = 120
ELBOW_STALL_RECOVERY_ENABLED = False
ELBOW_STALL_PROGRESS_TICKS = 20
ELBOW_STALL_RECOVERY_COUNT = 3
ELBOW_RECOVERY_WRIST_CENTER_TICK = 2049
ELBOW_RECOVERY_WRIST_ASSIST_TICKS = 300
ELBOW_RECOVERY_WAIT_S = 0.6

FEEDBACK_READS = 10
FEEDBACK_INTERVAL_S = 0.5
SAFE_POSITION_TOLERANCE_TICKS = 80
TARGET_POSITION_TOLERANCE_TICKS = 80
SAFE_FOLD_TIMEOUT_S = 8.0
TARGET_AXIS_TIMEOUT_S = 15.0
SAFE_READY_SETTLE_S = 4.0


def center_crop_and_resize_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    cropped = frame[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if (cropped.shape[1], cropped.shape[0]) == (target_w, target_h):
        return cropped
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


class RoboticInteraction:

    def __init__(
        self,
        xacro_path: str | None = None,
        camera_id: int = 1,
        base_camera_id: int = 0,
        model_path: str | None = None,
    ):
        if xacro_path is None:
            xacro_path = str(BASE_DIR / "interaction" / "so101_follower.urdf.xacro")
        if model_path is None:
            model_path = str(BASE_DIR / "yolo11s.pt")
        self.robot = RobotIKSolver(xacro_path)
        self.arm = ServoController()
        self.cap = cv2.VideoCapture(camera_id)
        self.base_cap = cv2.VideoCapture(base_camera_id)
        self._configure_camera(self.cap, "手部相机")
        self._configure_camera(self.base_cap, "底座相机")

        self.model = YOLO(model_path)
        self.scan_state = "RIGHT"
        self.state = "LOCATE"
        self.scan_speed = BASE_WHEEL_SEARCH_SPEED
        self.base_align_direction_sign = BASE_ALIGN_DIRECTION_SIGN
        self.last_base_align_abs_error: float | None = None
        self.last_base_align_speed: int | None = None
        self.base_align_lost_frames = 0
        self.last_target_conf: float | None = None

    def _configure_camera(self, cap: cv2.VideoCapture, name: str) -> None:
        if not cap.isOpened():
            raise RuntimeError(f"{name}打开失败")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_NATIVE_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_NATIVE_HEIGHT))

    def _preprocess_for_calib(self, frame: np.ndarray) -> np.ndarray:
        return center_crop_and_resize_frame(
            frame,
            (CAMERA_WORKING_WIDTH, CAMERA_WORKING_HEIGHT),
        )

    def _capture_hand_raw(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("手部相机读取失败")
        if HAND_CAMERA_ROTATE_CODE is not None:
            frame = cv2.rotate(frame, HAND_CAMERA_ROTATE_CODE)
        return frame

    def _capture_base_raw(self) -> np.ndarray:
        ret, frame = self.base_cap.read()
        if not ret or frame is None:
            raise RuntimeError("底座相机读取失败")
        return frame

    def _move_ticks(
        self,
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
                self.arm.move_to(servo_id, tick, speed=move_speed, acc=move_acc)
            time.sleep(delay_s)

    def _read_servo_positions(self, order: tuple[int, ...]) -> dict[int, int]:
        if DRY_RUN:
            return {}
        positions: dict[int, int] = {}
        for servo_id in order:
            positions[servo_id] = self.arm.get_position(servo_id)
            time.sleep(0.02)
        return positions

    def _move_ticks_smooth(
        self,
        ticks: dict[int, int],
        order: tuple[int, ...],
        speed: int,
        acc: int,
        servo_overrides: dict[int, dict[str, int]] | None = None,
    ) -> None:
        targets = {
            servo_id: int(ticks[servo_id])
            for servo_id in order
            if ticks.get(servo_id) is not None and int(ticks[servo_id]) >= 0
        }
        if not targets:
            return

        starts = self._read_servo_positions(tuple(targets.keys()))
        starts = {
            servo_id: starts.get(servo_id, -1)
            if starts.get(servo_id, -1) >= 0
            else targets[servo_id]
            for servo_id in targets
        }
        max_steps = 1
        for servo_id, target in targets.items():
            step_ticks = SMOOTH_STEP_SERVO_OVERRIDES.get(servo_id, SMOOTH_STEP_TICKS)
            max_steps = max(max_steps, int(np.ceil(abs(target - starts[servo_id]) / step_ticks)))

        print(f"平滑分段移动: steps={max_steps}, start={starts}, target={targets}")
        for step_idx in range(1, max_steps + 1):
            ratio = step_idx / max_steps
            print(f"  分段 {step_idx}/{max_steps}")
            for servo_id in order:
                if servo_id not in targets:
                    continue
                override = (servo_overrides or {}).get(servo_id, {})
                move_speed = override.get("speed", speed)
                move_acc = override.get("acc", acc)
                intermediate = int(round(starts[servo_id] + (targets[servo_id] - starts[servo_id]) * ratio))
                if not DRY_RUN:
                    self.arm.move_to(servo_id, intermediate, speed=move_speed, acc=move_acc)
            time.sleep(SMOOTH_STEP_DELAY_S)

        print("补发最终目标，帮助舵机克服末段误差")
        for _ in range(TARGET_FINAL_REISSUE_COUNT):
            self._move_ticks(
                targets,
                order,
                speed=speed,
                acc=acc,
                delay_s=0.05,
                servo_overrides=servo_overrides,
            )
            time.sleep(TARGET_FINAL_REISSUE_DELAY_S)
        time.sleep(TARGET_FINAL_SETTLE_S)

    def _recover_elbow_stall(
        self,
        targets: dict[int, int],
        positions: dict[int, int],
        speed: int,
        acc: int,
        servo_overrides: dict[int, dict[str, int]] | None = None,
    ) -> None:
        if 3 not in targets or 4 not in targets:
            return
        wrist_pos = positions.get(4, -1)
        if wrist_pos < 0:
            return

        if wrist_pos > ELBOW_RECOVERY_WRIST_CENTER_TICK:
            wrist_assist = max(ELBOW_RECOVERY_WRIST_CENTER_TICK, wrist_pos - ELBOW_RECOVERY_WRIST_ASSIST_TICKS)
        else:
            wrist_assist = min(ELBOW_RECOVERY_WRIST_CENTER_TICK, wrist_pos + ELBOW_RECOVERY_WRIST_ASSIST_TICKS)

        print(
            "检测到 3 号肘部轴卡滞，临时调整 4 号腕部减载: "
            f"4={wrist_pos}->{wrist_assist}, 3->{targets[3]}"
        )
        self._move_ticks(
            {4: wrist_assist},
            (4,),
            speed=speed,
            acc=acc,
            delay_s=ELBOW_RECOVERY_WAIT_S,
            servo_overrides=servo_overrides,
        )
        self._move_ticks(
            {3: targets[3]},
            (3,),
            speed=speed,
            acc=acc,
            delay_s=ELBOW_RECOVERY_WAIT_S,
            servo_overrides=servo_overrides,
        )

    def _settle_target_ticks(
        self,
        ticks: dict[int, int],
        order: tuple[int, ...],
        speed: int,
        acc: int,
        timeout_s: float = TARGET_SETTLE_TIMEOUT_S,
        tolerance_ticks: int = TARGET_POSITION_TOLERANCE_TICKS,
        reissue_error_ticks: int = TARGET_REISSUE_ERROR_TICKS,
        servo_overrides: dict[int, dict[str, int]] | None = None,
    ) -> dict[int, int]:
        targets = {
            servo_id: int(ticks[servo_id])
            for servo_id in order
            if ticks.get(servo_id) is not None and int(ticks[servo_id]) >= 0
        }
        if not targets or DRY_RUN:
            return {}

        deadline = time.monotonic() + timeout_s
        last_positions: dict[int, int] | None = None
        elbow_stall_count = 0
        while True:
            positions = self._read_servo_positions(tuple(targets.keys()))
            errors = {
                servo_id: targets[servo_id] - positions.get(servo_id, -1)
                for servo_id in targets
                if positions.get(servo_id, -1) >= 0
            }
            print(f"目标姿态等待: pos={positions}, err={errors}")

            arrived = (
                len(errors) == len(targets)
                and all(abs(error) <= tolerance_ticks for error in errors.values())
            )
            if arrived:
                print("目标姿态已到位")
                return positions

            if time.monotonic() >= deadline:
                print(f"目标姿态等待超时，最后反馈: {positions}")
                return positions

            elbow_error = errors.get(3)
            if (
                ELBOW_STALL_RECOVERY_ENABLED
                and elbow_error is not None
                and abs(elbow_error) > reissue_error_ticks
                and last_positions is not None
                and positions.get(3, -1) >= 0
                and last_positions.get(3, -1) >= 0
            ):
                elbow_progress = abs(positions[3] - last_positions[3])
                if elbow_progress <= ELBOW_STALL_PROGRESS_TICKS:
                    elbow_stall_count += 1
                else:
                    elbow_stall_count = 0
                if elbow_stall_count >= ELBOW_STALL_RECOVERY_COUNT:
                    elbow_stall_count = 0
                    self._recover_elbow_stall(
                        targets,
                        positions,
                        speed=speed,
                        acc=acc,
                        servo_overrides=servo_overrides,
                    )
            else:
                elbow_stall_count = 0

            retry_ticks = {
                servo_id: targets[servo_id]
                for servo_id, error in errors.items()
                if abs(error) > reissue_error_ticks
            }
            if retry_ticks:
                print(f"重发未到位舵机: {retry_ticks}")
                self._move_ticks(
                    retry_ticks,
                    order,
                    speed=speed,
                    acc=acc,
                    delay_s=0.04,
                    servo_overrides=servo_overrides,
                )
            last_positions = positions
            time.sleep(TARGET_SETTLE_INTERVAL_S)

    def _premove_axis(
        self,
        servo_id: int,
        ticks: dict[int, int],
        order: tuple[int, ...],
        label: str,
        ratio: float,
        max_ticks: int,
        min_ticks: int,
        timeout_s: float,
    ) -> None:
        if ticks.get(servo_id) is None or int(ticks[servo_id]) < 0:
            return

        target = int(ticks[servo_id])
        start = self._read_servo_positions(order).get(servo_id, -1)
        if start < 0:
            print(f"无法读取 {label} 当前位置，跳过预动作")
            return

        delta = target - start
        if abs(delta) < min_ticks:
            print(f"{label} 目标变化较小，跳过预动作，start={start}, target={target}")
            return

        premove_delta = int(round(delta * ratio))
        premove_delta = int(np.clip(premove_delta, -max_ticks, max_ticks))
        if premove_delta == 0:
            return

        premove_target = start + premove_delta
        premove_ticks = {servo_id: premove_target}

        print(
            f"\n预动作: {label} 先动一部分，"
            f"start={start}, premove={premove_target}, final={target}"
        )
        self._move_ticks_smooth(
            premove_ticks,
            order,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )
        self._settle_target_ticks(
            premove_ticks,
            order,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            timeout_s=timeout_s,
            tolerance_ticks=TARGET_REISSUE_ERROR_TICKS,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )

    def _move_wrist_to_assist(self) -> None:
        assist_ticks = {4: TARGET_WRIST_ASSIST_TICK}
        print(f"\n阶段 1: 先把 4 号腕部轴移动到辅助位 {TARGET_WRIST_ASSIST_TICK}")
        self._move_ticks_smooth(
            assist_ticks,
            TARGET_WRIST_FINAL_ORDER,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )
        self._settle_target_ticks(
            assist_ticks,
            TARGET_WRIST_FINAL_ORDER,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            timeout_s=TARGET_WRIST_ASSIST_TIMEOUT_S,
            tolerance_ticks=TARGET_REISSUE_ERROR_TICKS,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )

    def _move_target_with_joint_premoves(self, ticks: dict[int, int]) -> bool:
        if TARGET_WRIST_ASSIST_ENABLED:
            self._move_wrist_to_assist()

        if TARGET_SHOULDER_PREMOVE_ENABLED:
            self._premove_axis(
                2,
                ticks,
                TARGET_SHOULDER_FIRST_ORDER,
                "2 号肩部轴",
                TARGET_SHOULDER_PREMOVE_RATIO,
                TARGET_SHOULDER_PREMOVE_MAX_TICKS,
                TARGET_SHOULDER_PREMOVE_MIN_TICKS,
                TARGET_SHOULDER_PREMOVE_TIMEOUT_S,
            )

        if TARGET_ELBOW_PREMOVE_ENABLED:
            self._premove_axis(
                3,
                ticks,
                TARGET_ELBOW_PREMOVE_ORDER,
                "3 号肘部轴",
                TARGET_ELBOW_PREMOVE_RATIO,
                TARGET_ELBOW_PREMOVE_MAX_TICKS,
                TARGET_ELBOW_PREMOVE_MIN_TICKS,
                TARGET_ELBOW_PREMOVE_TIMEOUT_S,
            )

        print("\n阶段 2: 2/3/1/5 协同移动到最终目标，4 号保持辅助位")
        self._move_ticks_smooth(
            ticks,
            TARGET_MOVE_ORDER_WITHOUT_WRIST,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )

        print("\n阶段 3: 最后移动 4 号腕部轴到最终目标")
        self._move_ticks_smooth(
            ticks,
            TARGET_WRIST_FINAL_ORDER,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )
        return True

    def _wait_until_ticks(
        self,
        ticks: dict[int, int],
        order: tuple[int, ...],
        timeout_s: float,
        tolerance_ticks: int = SAFE_POSITION_TOLERANCE_TICKS,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            positions = {servo_id: self.arm.get_position(servo_id) for servo_id in order}
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

    def _move_to_safe_ready(self) -> None:
        print("先让 3/4 号回到折叠安全姿态")
        self._move_ticks(
            SAFE_READY_TICKS,
            SAFE_FOLD_ORDER,
            speed=SAFE_MOVE_SPEED,
            acc=SAFE_MOVE_ACC,
            delay_s=0.6,
            servo_overrides=SAFE_MOVE_SERVO_OVERRIDES,
        )
        self._wait_until_ticks(
            SAFE_READY_TICKS,
            SAFE_FOLD_ORDER,
            timeout_s=SAFE_FOLD_TIMEOUT_S,
        )
        print("3/4 号已到位，再移动 2/1/5 回初始")
        self._move_ticks(
            SAFE_READY_TICKS,
            SAFE_REST_ORDER,
            speed=SAFE_MOVE_SPEED,
            acc=SAFE_MOVE_ACC,
            delay_s=0.6,
            servo_overrides=SAFE_MOVE_SERVO_OVERRIDES,
        )
        print(f"等待安全姿态稳定 {SAFE_READY_SETTLE_S:.1f}s")
        time.sleep(SAFE_READY_SETTLE_S)

    def move_to_world_xyz(self, target_world_xyz: np.ndarray, return_to_ready: bool = False) -> None:
        print("[DEBUG] 使用平滑分段移动策略：从当前姿态逐步移动到目标")

        self.target_world_xyz = target_world_xyz
        self.last_ik_solution = solve_ik(self.target_world_xyz)

        print("目标坐标:", self.target_world_xyz)
        print("目标 tick:", self.last_ik_solution)

        if return_to_ready:
            self._move_to_safe_ready()

        print("\n移动到目标姿态")

        target_move_started = True
        if TARGET_WRIST_ASSIST_ENABLED or TARGET_SHOULDER_PREMOVE_ENABLED or TARGET_ELBOW_PREMOVE_ENABLED:
            target_move_started = self._move_target_with_joint_premoves(self.last_ik_solution)
        elif SMOOTH_MOVE_ENABLED:
            self._move_ticks_smooth(
                self.last_ik_solution,
                TARGET_MOVE_ORDER,
                speed=TARGET_MOVE_SPEED,
                acc=TARGET_MOVE_ACC,
                servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
            )
        else:
            self._move_ticks(
                self.last_ik_solution,
                TARGET_MOVE_ORDER,
                speed=TARGET_MOVE_SPEED,
                acc=TARGET_MOVE_ACC,
                delay_s=SERVO_DELAY_S,
                servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
            )

        if not target_move_started:
            return

        print("\n目标命令已下发，等待目标姿态稳定：")
        self._settle_target_ticks(
            self.last_ik_solution,
            TARGET_MOVE_ORDER,
            speed=TARGET_MOVE_SPEED,
            acc=TARGET_MOVE_ACC,
            servo_overrides=TARGET_MOVE_SERVO_OVERRIDES,
        )

    def interact(self, target: str):
        extrinsics = str(BASE_DIR / "calib" / "stereo_extrinsics.json")
        self.target_world_xyz = self.double_cap_locate(target, extrinsics)
        print(self.target_world_xyz)
        self.move_to_world_xyz(self.target_world_xyz)
        return self.target_world_xyz

    def detect_object(
        self,
        img: np.ndarray,
        target_class: str,
        save_path: str | None = None,
    ) -> tuple[float, float] | None:
        results = self.model(img, verbose=False)
        best_target: tuple[float, np.ndarray] | None = None
        visible_classes: list[str] = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue
                cls_name = self.model.names[int(box.cls[0])]
                visible_classes.append(f"{cls_name}:{conf:.2f}")
                if cls_name == target_class:
                    xyxy = box.xyxy[0].cpu().numpy()
                    if best_target is None or conf > best_target[0]:
                        best_target = (conf, xyxy)

        if best_target is None:
            self.last_target_conf = None
            if visible_classes:
                print(f"未检测到 {target_class}; 当前可见: {', '.join(visible_classes[:6])}")
            return None

        conf, xyxy = best_target
        self.last_target_conf = conf
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        cv2.rectangle(
            img,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            f"{target_class} {conf:.2f}",
            (int(xyxy[0]), int(xyxy[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.drawMarker(
            img,
            (int(cx), int(cy)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.putText(
            img,
            f"({cx:.1f}, {cy:.1f})",
            (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        print(f"检测到 {target_class}: conf={conf:.2f}, center=({cx:.1f}, {cy:.1f})")
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"检测结果已保存: {save_path}")
        return (float(cx), float(cy))

    def detect_object_with_yolo_rotation(
        self,
        img: np.ndarray,
        target_class: str,
        rotate_code: int | None = None,
        save_path: str | None = None,
    ) -> tuple[float, float] | None:
        if rotate_code is None:
            return self.detect_object(img, target_class, save_path=save_path)

        h, w = img.shape[:2]

        img_rot = cv2.rotate(img, rotate_code)
        pt_rot = self.detect_object(img_rot, target_class)

        if pt_rot is None:
            return None

        u_rot, v_rot = pt_rot

        if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            u = w - 1 - v_rot
            v = u_rot
            inv_rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif rotate_code == cv2.ROTATE_90_CLOCKWISE:
            u = v_rot
            v = h - 1 - u_rot
            inv_rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotate_code == cv2.ROTATE_180:
            u = w - 1 - u_rot
            v = h - 1 - v_rot
            inv_rotate_code = cv2.ROTATE_180
        else:
            raise ValueError(f"Unsupported rotate_code: {rotate_code}")

        img_annotated = img_rot.copy()
        cv2.drawMarker(
            img_annotated,
            (int(u_rot), int(v_rot)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.putText(
            img_annotated,
            f"({u:.1f}, {v:.1f})",
            (int(u_rot) + 10, int(v_rot) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if save_path:
            img_saved = cv2.rotate(img_annotated, inv_rotate_code)
            cv2.imwrite(save_path, img_saved)
            print(f"检测结果已保存: {save_path}")

        return float(u), float(v)
    
    def detect_target(
        self, target: str, save_dir: str | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
        os.makedirs(save_dir, exist_ok=True)
        frame_hand_raw = self._capture_hand_raw()
        frame_base_raw = self._capture_base_raw()

        frame_hand = self._preprocess_for_calib(frame_hand_raw)
        frame_base = self._preprocess_for_calib(frame_base_raw)

        base_save = os.path.join(save_dir, "base_detection.png") if save_dir else None
        hand_save = os.path.join(save_dir, "hand_detection.png") if save_dir else None

        pts_hand = self.detect_object_with_yolo_rotation(
            frame_hand,
            target,
            rotate_code=HAND_YOLO_ROTATE_CODE,
            save_path=hand_save,
        )

        pts_base = self.detect_object_with_yolo_rotation(
            frame_base,
            target,
            rotate_code=BASE_YOLO_ROTATE_CODE,
            save_path=base_save,
        )

        print("手部相机检测点 pts_hand @540x540:", pts_hand)
        print("底座相机检测点 pts_base @540x540:", pts_base)

        if pts_hand is None:
            raise RuntimeError(f"手部相机没有检测到目标: {target}")
        if pts_base is None:
            raise RuntimeError(f"底座相机没有检测到目标: {target}")

        return pts_base, pts_hand

    def _detect_base_target(self, target: str) -> tuple[float, float] | None:
        base_img_raw = self._capture_base_raw()
        base_img = self._preprocess_for_calib(base_img_raw)
        return self.detect_object_with_yolo_rotation(
            base_img,
            target,
            rotate_code=BASE_YOLO_ROTATE_CODE,
        )

    def _spin_base_by_wheels(self, speed: int) -> None:
        """Rotate the chassis so the fixed base camera sweeps the scene."""
        speed = int(speed)

        if speed == 0:
            self._stop_base_wheels()
            return

        self.scan_state = "RIGHT" if speed > 0 else "LEFT"
        self.arm.spin_wheel(speed)


    def _stop_base_wheels(self) -> None:
        self.arm.spin_wheel(0, acc=255)

    def _scan_with_base_camera(self) -> None:
        if self.scan_state == "RIGHT":
            self._spin_base_by_wheels(abs(self.scan_speed))
        else:
            self._spin_base_by_wheels(-abs(self.scan_speed))

    def _align_base_camera_to_target(self, base_uv: tuple[float, float]) -> bool:
        image_center_x = CAMERA_WORKING_WIDTH / 2.0
        error_x = float(base_uv[0]) - image_center_x
        abs_error_x = abs(error_x)
        if abs(error_x) <= BASE_SEARCH_DEADZONE_PX:
            self._stop_base_wheels()
            self.last_base_align_abs_error = None
            self.last_base_align_speed = None
            self.base_align_lost_frames = 0
            print(f"底部相机已看到目标并进入可定位区域，error_x={error_x:.1f}px")
            return True

        if (
            self.last_base_align_abs_error is not None
            and abs_error_x > self.last_base_align_abs_error + BASE_ALIGN_REVERSE_MARGIN_PX
            and self.last_target_conf is not None
            and self.last_target_conf >= BASE_ALIGN_REVERSE_MIN_CONF
        ):
            self.base_align_direction_sign *= -1
            print(
                "底部相机对准误差变大，反向修正: "
                f"last={self.last_base_align_abs_error:.1f}px, "
                f"current={abs_error_x:.1f}px, conf={self.last_target_conf:.2f}"
            )

        direction = self.base_align_direction_sign if error_x > 0 else -self.base_align_direction_sign
        align_speed = min(
            BASE_WHEEL_ALIGN_MAX_SPEED,
            max(BASE_WHEEL_ALIGN_MIN_SPEED, int(abs_error_x * BASE_WHEEL_ALIGN_KP)),
        )
        speed = int(direction * align_speed)
        self.scan_state = "RIGHT" if speed > 0 else "LEFT"
        print(
            f"底部相机对准中，target_x={base_uv[0]:.1f}, "
            f"error_x={error_x:.1f}, speed={speed}"
        )
        self.last_base_align_abs_error = abs_error_x
        self.last_base_align_speed = speed
        self.base_align_lost_frames = 0
        self._spin_base_by_wheels(speed)
        return False

    def locate_point(self, target: str) -> tuple[float, float]:
        """Find and roughly center the target by rotating the chassis wheels."""
        self.state = "LOCATE"

        stable_frames = 0
        deadline = time.monotonic() + BASE_SEARCH_TIMEOUT_S
        try:
            while self.state == "LOCATE":
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"底部相机没有在 {BASE_SEARCH_TIMEOUT_S:.1f}s 内找到目标: {target}")

                base_uv = self._detect_base_target(target)
                if base_uv is None:
                    print("底部相机初始没有检测到目标，开始扫视寻找...")
                    stable_frames = 0
                    if self.last_base_align_speed is not None:
                        self.base_align_lost_frames += 1
                        if self.base_align_lost_frames < BASE_ALIGN_LOST_RESET_FRAMES:
                            self._spin_base_by_wheels(self.last_base_align_speed)
                            time.sleep(BASE_SEARCH_INTERVAL_S)
                            continue
                        self.last_base_align_speed = None
                        self.base_align_lost_frames = 0

                    self._scan_with_base_camera()
                    time.sleep(BASE_SEARCH_INTERVAL_S)
                    continue

                if self._align_base_camera_to_target(base_uv):
                    stable_frames += 1
                    if stable_frames >= BASE_SEARCH_STABLE_FRAMES:
                        self.state = "TRACK"
                        print("底部相机稳定找到目标，准备进行双目定位")
                        return base_uv
                else:
                    stable_frames = 0

                time.sleep(BASE_SEARCH_INTERVAL_S)
        finally:
            self._stop_base_wheels()

        raise RuntimeError(f"底部相机没有找到目标: {target}")

    def convert_4d_to_3d(self, points_4d: np.ndarray) -> np.ndarray:
        points_4d = np.asarray(points_4d, dtype=np.float64).reshape(4, -1)
        w = float(points_4d[3, 0])
        if abs(w) < 1e-12:
            raise ValueError("三角化失败: homogeneous w is too close to zero")

        point_3d = points_4d[:3, 0] / w
        x_cam, y_cam, z_cam = point_3d
        point_from_base_camera = np.array([
            z_cam,    # 前方 -> X
            x_cam,   # 左方 -> Y
            -y_cam,   # 上方 -> Z
            ], dtype=np.float64)
        point_arm = BASE_CAMERA_ORIGIN_IN_ARM_M + point_from_base_camera
        print("三维坐标 relative to base camera [m]:", point_from_base_camera)
        print("三维坐标 in arm base frame [m]:", point_arm)
        return point_arm.reshape(3)

    def double_cap_locate(self, target: str, extrinsics_path: str) -> np.ndarray:
        self.locate_point(target)
        with open(extrinsics_path, "r") as f:
            ext = json.load(f)

        P1 = np.array(
            ext.get("P1_base_as_world", ext.get("P1_raw", ext["P1_rectification"])),
            dtype=np.float64,
        )
        P2 = np.array(
            ext.get("P2_hand_from_base", ext.get("P2_raw", ext["P2_rectification"])),
            dtype=np.float64,
        )

        save_dir = os.path.join(os.path.dirname(extrinsics_path), "detection_results")
        os.makedirs(save_dir, exist_ok=True)
        pts_base, pts_hand = self.detect_target(target, save_dir=save_dir)

        pts_base = np.array(pts_base, dtype=np.float64).reshape(2, 1)
        pts_hand = np.array(pts_hand, dtype=np.float64).reshape(2, 1)

        print("用于三角化的底座点 pts_base:", pts_base.ravel())
        print("用于三角化的手部点 pts_hand:", pts_hand.ravel())

        points_4d = cv2.triangulatePoints(P1, P2, pts_base, pts_hand)
        point_3d_base = self.convert_4d_to_3d(points_4d)

        print("三维坐标 in arm base frame [m]:", point_3d_base)
        return point_3d_base


if __name__ == "__main__":
    robot = RoboticInteraction()
    robot.interact("bottle")
