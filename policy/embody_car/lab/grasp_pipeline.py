from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from arm_control import ServoController
from camera import (
    BASE_YOLO_ROTATE_CODE,
    HAND_YOLO_ROTATE_CODE,
    CameraManager,
    center_crop_resize,
)


BASE_DIR = Path(__file__).resolve().parent

BASE_SEARCH_TIMEOUT_S = 60.0
LOOP_INTERVAL_S = 0.08

BASE_CENTER_DEADZONE_PX = 100
BASE_ALIGN_MIN_SPEED = 10
BASE_ALIGN_MAX_SPEED = 35
BASE_ALIGN_KP = 0.08
BASE_ALIGN_DIRECTION_SIGN = -1
BASE_APPROACH_CORRECTION_DEADZONE_PX = 25
BASE_APPROACH_CORRECTION_MIN_SPEED = 6
BASE_APPROACH_CORRECTION_MAX_SPEED = 28
BASE_APPROACH_SLOWDOWN_ERROR_PX = 180

WHEEL_SEARCH_SPEED = 150
WHEEL_APPROACH_SPEED = 170
CHASSIS_MOTOR_IDS = (7, 8, 9)
CHASSIS_X_SIGN = -1
CHASSIS_Y_SIGN = -1
CHASSIS_OMEGA_SIGN = 1

WRIST_READY_STABLE_FRAMES = 5
WRIST_READY_CENTER_X_DEADZONE_PX = 140
WRIST_READY_CENTER_Y_DEADZONE_PX = 120
WRIST_TARGET_OFFSET_X_PX = 40
WRIST_TARGET_OFFSET_Y_PX = -30
WRIST_ALIGN_DEADZONE_PX = 45
WRIST_ALIGN_MIN_SPEED = 8
WRIST_ALIGN_MAX_SPEED = 80
WRIST_ALIGN_KP = 0.12
WRIST_Y_ALIGN_KP = 0.35
WRIST_ALIGN_DIRECTION_SIGN = BASE_ALIGN_DIRECTION_SIGN
WRIST_SEEN_FORWARD_SPEED_RATIO = 0.7
WRIST_Y_ADJUST_SPEED_RATIO = 0.5
WRIST_MIN_CONF = 0.35
WRIST_MIN_BOX_AREA_RATIO = 0.07
WRIST_READY_MIN_BOX_OVERLAP_RATIO = 0.30

YOLO_CONF_THRESHOLD = 0.25
FRAME_MODE_CROP = "crop"
FRAME_MODE_LETTERBOX = "letterbox"
WRIST_SERVO_MODE_SEQUENTIAL = "sequential"
WRIST_SERVO_MODE_PROPORTIONAL = "proportional"
NAV_FRAME_WIDTH = 640
NAV_FRAME_HEIGHT = 480


def letterbox_resize(
    frame: np.ndarray,
    *,
    width: int = NAV_FRAME_WIDTH,
    height: int = NAV_FRAME_HEIGHT,
) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


@dataclass(frozen=True)
class TargetDetection:
    uv: tuple[float, float]
    conf: float
    xyxy: tuple[float, float, float, float]
    box_area_ratio: float
    source: str


class DryRunServoController:
    def __init__(self) -> None:
        self._last_wheel_command: tuple[str, int, int] | None = None

    def reset(self) -> None:
        print("[DRY RUN] arm.reset()")

    def set_mode(self, servo_id: int, mode: int) -> None:
        print(f"[DRY RUN] set_mode servo={servo_id}, mode={mode}")

    def _send_write(self, servo_id: int, address: int, data: list[int]) -> None:
        print(f"[DRY RUN] write servo={servo_id}, address={address}, data={data}")

    def spin_wheel(self, speed: int, acc: int = 50) -> None:
        command = ("spin_wheel", int(speed), int(acc))
        if command != self._last_wheel_command:
            print(f"[DRY RUN] spin_wheel speed={speed}, acc={acc}")
            self._last_wheel_command = command

    def move_wheel(self, mode: int, speed: int, acc: int = 50) -> None:
        command = ("move_wheel", int(mode), int(speed))
        if command != self._last_wheel_command:
            print(f"[DRY RUN] move_wheel mode={mode}, speed={speed}, acc={acc}")
            self._last_wheel_command = command

    def spin(self, servo_id: int, speed: int, acc: int = 50) -> None:
        print(f"[DRY RUN] spin servo={servo_id}, speed={speed}, acc={acc}")


class GraspPipeline:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        timeout_s: float = BASE_SEARCH_TIMEOUT_S,
        show: bool = False,
        dry_run: bool = False,
        approach_speed: int = WHEEL_APPROACH_SPEED,
        hand_camera_id: int = 3,
        base_camera_id: int = 0,
        frame_mode: str = FRAME_MODE_CROP,
        frame_width: int = NAV_FRAME_WIDTH,
        frame_height: int = NAV_FRAME_HEIGHT,
        wrist_target_offset_x_px: float = WRIST_TARGET_OFFSET_X_PX,
        wrist_target_offset_y_px: float = WRIST_TARGET_OFFSET_Y_PX,
        wrist_ready_center_x_deadzone_px: float = WRIST_READY_CENTER_X_DEADZONE_PX,
        wrist_ready_center_y_deadzone_px: float = WRIST_READY_CENTER_Y_DEADZONE_PX,
        wrist_ready_min_box_overlap_ratio: float = WRIST_READY_MIN_BOX_OVERLAP_RATIO,
        wrist_ready_position_mode: str = "center_or_overlap",
        wrist_preferred_box_area_ratio: float = 0.0,
        wrist_preferred_timeout_s: float = 0.0,
        wrist_min_box_visible_ratio: float = 0.0,
        wrist_visible_margin_px: float = 0.0,
        wrist_max_box_area_ratio: float = 0.0,
        wrist_min_conf: float = WRIST_MIN_CONF,
        wrist_min_box_area_ratio: float = WRIST_MIN_BOX_AREA_RATIO,
        wrist_visual_servo_mode: str = WRIST_SERVO_MODE_SEQUENTIAL,
        wheel_search_speed: int = WHEEL_SEARCH_SPEED,
        wrist_align_max_speed: int = WRIST_ALIGN_MAX_SPEED,
        wrist_align_min_speed: int = WRIST_ALIGN_MIN_SPEED,
    ) -> None:
        if model_path is None:
            model_path = str(BASE_DIR / "yolo11s.pt")

        self.camera = CameraManager(hand_id=hand_camera_id, base_id=base_camera_id)
        self.arm = DryRunServoController() if dry_run else ServoController()
        self.model = YOLO(model_path)
        self.timeout_s = float(timeout_s)
        self.show = show
        self.dry_run = dry_run
        self.approach_speed = int(approach_speed)
        if frame_mode not in (FRAME_MODE_CROP, FRAME_MODE_LETTERBOX):
            raise ValueError(f"Unsupported frame_mode: {frame_mode}")
        self.frame_mode = frame_mode
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.wrist_target_offset_x_px = float(wrist_target_offset_x_px)
        self.wrist_target_offset_y_px = float(wrist_target_offset_y_px)
        self.wrist_ready_center_x_deadzone_px = float(wrist_ready_center_x_deadzone_px)
        self.wrist_ready_center_y_deadzone_px = float(wrist_ready_center_y_deadzone_px)
        self.wrist_ready_min_box_overlap_ratio = float(wrist_ready_min_box_overlap_ratio)
        if wrist_ready_position_mode not in ("center_or_overlap", "center", "overlap"):
            raise ValueError(f"Unsupported wrist_ready_position_mode: {wrist_ready_position_mode}")
        self.wrist_ready_position_mode = wrist_ready_position_mode
        self.wrist_preferred_box_area_ratio = float(wrist_preferred_box_area_ratio)
        self.wrist_preferred_timeout_s = float(wrist_preferred_timeout_s)
        self.wrist_min_box_visible_ratio = float(wrist_min_box_visible_ratio)
        self.wrist_visible_margin_px = float(wrist_visible_margin_px)
        self.wrist_max_box_area_ratio = float(wrist_max_box_area_ratio)
        self.wrist_min_conf = float(wrist_min_conf)
        self.wrist_min_box_area_ratio = float(wrist_min_box_area_ratio)
        if wrist_visual_servo_mode not in (WRIST_SERVO_MODE_SEQUENTIAL, WRIST_SERVO_MODE_PROPORTIONAL):
            raise ValueError(f"Unsupported wrist_visual_servo_mode: {wrist_visual_servo_mode}")
        self.wrist_visual_servo_mode = wrist_visual_servo_mode
        self.wheel_search_speed = int(wheel_search_speed)
        self.wrist_align_max_speed = int(wrist_align_max_speed)
        self.wrist_align_min_speed = int(wrist_align_min_speed)
        self._last_chassis_command: tuple[int, int, int] | None = None
        self._wrist_preferred_started_at: float | None = None
        self._init_chassis_wheels()

        base_frame = self._read_base_frame()
        wrist_frame = self._read_wrist_frame()
        self.base_center_x = base_frame.shape[1] / 2.0
        self.base_center_y = base_frame.shape[0] / 2.0
        self.wrist_center_x = wrist_frame.shape[1] / 2.0
        self.wrist_center_y = wrist_frame.shape[0] / 2.0
        self.wrist_target_x = self.wrist_center_x + self.wrist_target_offset_x_px
        self.wrist_target_y = self.wrist_center_y + self.wrist_target_offset_y_px

    def run(self, target: str) -> None:
        detection = self.approach_until_wrist_ready(target)
        print(
            "Wrist target ready: "
            f"uv={detection.uv}, conf={detection.conf:.2f}, "
            f"area={detection.box_area_ratio:.3f}"
        )

    def approach_until_wrist_ready(
        self,
        target: str,
        *,
        wrist_target: str | None = None,
        reset_arm: bool = True,
    ) -> TargetDetection:
        wrist_target = wrist_target or target
        if reset_arm:
            self.arm.reset()
        no_target_deadline = time.monotonic() + self.timeout_s
        stable_frames = 0
        last_state = "init"
        last_wrist_detection: TargetDetection | None = None

        try:
            while True:
                if time.monotonic() >= no_target_deadline:
                    raise RuntimeError(
                        f"连续 {self.timeout_s:.1f}s 没有在 base/wrist 看到目标: "
                        f"base={target}, wrist={wrist_target}"
                    )

                base_frame = self._read_base_frame()
                wrist_frame = self._read_wrist_frame()
                base_det = self._detect_target(
                    base_frame,
                    target,
                    rotate_code=BASE_YOLO_ROTATE_CODE,
                    source="base",
                )
                wrist_det = self._detect_target(
                    wrist_frame,
                    wrist_target,
                    rotate_code=HAND_YOLO_ROTATE_CODE,
                    source="wrist",
                )

                if base_det is not None or wrist_det is not None:
                    no_target_deadline = time.monotonic() + self.timeout_s

                if wrist_det is None:
                    base_state, base_aligned = self._drive_base_alignment(
                        base_det,
                        stop_when_centered=False,
                    )
                    if not base_aligned:
                        last_state = base_state
                        stable_frames = 0
                    else:
                        self._move_forward_with_base_correction(base_det)
                        last_state = "base_guided_forward"
                        stable_frames = 0
                else:
                    last_state, stable_frames = self._drive_wrist_visual_servo(
                        wrist_det,
                        stable_frames,
                    )
                    if wrist_det is not None:
                        last_wrist_detection = wrist_det
                    if stable_frames >= WRIST_READY_STABLE_FRAMES:
                        self._stop_wheels()
                        if last_wrist_detection is None:
                            raise RuntimeError("wrist ready reached without a detection")
                        return last_wrist_detection

                if self.show:
                    if not self._show_debug(
                        base_frame,
                        wrist_frame,
                        base_det,
                        wrist_det,
                        last_state,
                        stable_frames,
                    ):
                        raise KeyboardInterrupt("user quit")

                time.sleep(LOOP_INTERVAL_S)
        finally:
            self._stop_wheels()

    def close(self) -> None:
        self._stop_wheels()
        self.camera.release()
        if hasattr(self.arm, "_ser"):
            self.arm._ser.close()
        if self.show:
            cv2.destroyAllWindows()

    def _drive_base_alignment(
        self,
        base_det: TargetDetection | None,
        *,
        stop_when_centered: bool = True,
    ) -> tuple[str, bool]:
        if base_det is None:
            self._spin_base(self.wheel_search_speed)
            return "base_search", False

        error_x = base_det.uv[0] - self.base_center_x
        if abs(error_x) <= BASE_CENTER_DEADZONE_PX:
            if stop_when_centered:
                self._stop_wheels()
            return "base_centered", True

        speed = self._signed_speed(
            error_x,
            kp=BASE_ALIGN_KP,
            min_speed=BASE_ALIGN_MIN_SPEED,
            max_speed=BASE_ALIGN_MAX_SPEED,
            direction_sign=BASE_ALIGN_DIRECTION_SIGN,
        )
        self._spin_base(speed)
        return "base_align", False

    def _drive_wrist_visual_servo(
        self,
        wrist_det: TargetDetection,
        stable_frames: int,
    ) -> tuple[str, int]:
        (
            clear_enough,
            centered_enough,
            overlap_enough,
            _,
            visible_enough,
            not_too_large,
        ) = self._wrist_ready_metrics(wrist_det)
        position_ready = self._wrist_position_ready(centered_enough, overlap_enough)

        if not not_too_large:
            self._wrist_preferred_started_at = None
            self._move_backward_for_wrist_fit()
            return "wrist_too_close", 0

        if clear_enough and position_ready and not visible_enough:
            self._wrist_preferred_started_at = None
            self._move_to_improve_wrist_visibility(wrist_det)
            return "wrist_fit_view", 0

        if not clear_enough or not position_ready:
            self._wrist_preferred_started_at = None
            self._move_with_wrist_correction(wrist_det, clear_enough)
            return "wrist_visual_servo", 0

        if self._should_keep_approaching_preferred_area(wrist_det):
            self._move_with_wrist_correction(wrist_det, clear_enough, force_forward=True)
            return "wrist_preferred_approach", 0

        self._stop_wheels()
        self._wrist_preferred_started_at = None
        return "wrist_ready", stable_frames + 1

    def _should_keep_approaching_preferred_area(self, detection: TargetDetection) -> bool:
        if self.wrist_preferred_box_area_ratio <= 0:
            return False
        if detection.box_area_ratio >= self.wrist_preferred_box_area_ratio:
            self._wrist_preferred_started_at = None
            return False

        now = time.monotonic()
        if self._wrist_preferred_started_at is None:
            self._wrist_preferred_started_at = now

        if self.wrist_preferred_timeout_s > 0:
            elapsed = now - self._wrist_preferred_started_at
            if elapsed >= self.wrist_preferred_timeout_s:
                return False

        return True

    def _wrist_ready_metrics(
        self,
        detection: TargetDetection,
    ) -> tuple[bool, bool, bool, float, bool, bool]:
        clear_enough = (
            detection.conf >= self.wrist_min_conf
            and detection.box_area_ratio >= self.wrist_min_box_area_ratio
        )
        not_too_large = (
            self.wrist_max_box_area_ratio <= 0
            or detection.box_area_ratio <= self.wrist_max_box_area_ratio
        )
        centered_enough = (
            abs(detection.uv[0] - self.wrist_target_x) <= self.wrist_ready_center_x_deadzone_px
            and abs(detection.uv[1] - self.wrist_target_y) <= self.wrist_ready_center_y_deadzone_px
        )
        overlap_ratio = self._wrist_ready_overlap_ratio(detection)
        overlap_enough = overlap_ratio >= self.wrist_ready_min_box_overlap_ratio
        visible_ratio = self._wrist_safe_visible_ratio(detection)
        visible_enough = visible_ratio >= self.wrist_min_box_visible_ratio
        return clear_enough, centered_enough, overlap_enough, overlap_ratio, visible_enough, not_too_large

    def _wrist_position_ready(self, centered_enough: bool, overlap_enough: bool) -> bool:
        if self.wrist_ready_position_mode == "center":
            return centered_enough
        if self.wrist_ready_position_mode == "overlap":
            return overlap_enough
        return centered_enough or overlap_enough

    def _wrist_ready_box(self) -> tuple[float, float, float, float]:
        return (
            self.wrist_target_x - self.wrist_ready_center_x_deadzone_px,
            self.wrist_target_y - self.wrist_ready_center_y_deadzone_px,
            self.wrist_target_x + self.wrist_ready_center_x_deadzone_px,
            self.wrist_target_y + self.wrist_ready_center_y_deadzone_px,
        )

    def _wrist_ready_overlap_ratio(self, detection: TargetDetection) -> float:
        return self._box_overlap_ratio(detection.xyxy, self._wrist_ready_box())

    def _wrist_safe_box(self) -> tuple[float, float, float, float]:
        margin = max(0.0, self.wrist_visible_margin_px)
        return (
            margin,
            margin,
            self.frame_width - margin,
            self.frame_height - margin,
        )

    def _wrist_safe_visible_ratio(self, detection: TargetDetection) -> float:
        if self.wrist_min_box_visible_ratio <= 0:
            return 1.0
        return self._box_overlap_ratio(detection.xyxy, self._wrist_safe_box())

    def _box_overlap_ratio(
        self,
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = max(1.0, min(area_a, area_b))
        return float(inter / denom)

    def _read_base_frame(self) -> np.ndarray:
        return self._prepare_nav_frame(self.camera.read_base_raw())

    def _read_wrist_frame(self) -> np.ndarray:
        return self._prepare_nav_frame(self.camera.read_hand_raw())

    def _prepare_nav_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.frame_mode == FRAME_MODE_LETTERBOX:
            return letterbox_resize(frame, width=self.frame_width, height=self.frame_height)
        return center_crop_resize(frame)

    def _detect_target(
        self,
        frame: np.ndarray,
        target: str,
        rotate_code: int | None = None,
        source: str = "base",
    ) -> TargetDetection | None:
        detect_frame = cv2.rotate(frame, rotate_code) if rotate_code is not None else frame
        results = self.model(detect_frame, verbose=False)
        target_name = target.strip().lower()

        best: tuple[float, np.ndarray] | None = None
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < YOLO_CONF_THRESHOLD:
                    continue
                cls_name = self.model.names[int(box.cls[0])].lower()
                if cls_name != target_name:
                    continue
                xyxy_rot = box.xyxy[0].cpu().numpy()
                if best is None or conf > best[0]:
                    best = (conf, xyxy_rot)

        if best is None:
            return None

        conf, xyxy_rot = best
        xyxy = self._map_xyxy_to_original(frame, xyxy_rot, rotate_code)
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        frame_area = float(frame.shape[0] * frame.shape[1])
        return TargetDetection(
            uv=(float(cx), float(cy)),
            conf=conf,
            xyxy=(float(x1), float(y1), float(x2), float(y2)),
            box_area_ratio=float(box_area / frame_area),
            source=source,
        )

    def _map_xyxy_to_original(
        self,
        frame: np.ndarray,
        xyxy_rot: np.ndarray,
        rotate_code: int | None,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in xyxy_rot]
        corners = [
            self._map_point_to_original(frame, x1, y1, rotate_code),
            self._map_point_to_original(frame, x2, y1, rotate_code),
            self._map_point_to_original(frame, x2, y2, rotate_code),
            self._map_point_to_original(frame, x1, y2, rotate_code),
        ]
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _map_point_to_original(
        self,
        frame: np.ndarray,
        u_rot: float,
        v_rot: float,
        rotate_code: int | None,
    ) -> tuple[float, float]:
        if rotate_code is None:
            return u_rot, v_rot

        h, w = frame.shape[:2]
        if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return w - 1 - v_rot, u_rot
        if rotate_code == cv2.ROTATE_90_CLOCKWISE:
            return v_rot, h - 1 - u_rot
        if rotate_code == cv2.ROTATE_180:
            return w - 1 - u_rot, h - 1 - v_rot
        raise ValueError(f"Unsupported rotate_code: {rotate_code}")

    def _signed_speed(
        self,
        error_x: float,
        *,
        kp: float,
        min_speed: int,
        max_speed: int,
        direction_sign: int,
    ) -> int:
        magnitude = int(np.clip(abs(error_x) * kp, min_speed, max_speed))
        direction = direction_sign if error_x > 0 else -direction_sign
        return int(direction * magnitude)

    def _spin_base(self, speed: int) -> None:
        self._drive_chassis(0, 0, int(speed))

    def _move_forward(self) -> None:
        self._drive_chassis(0, self.approach_speed, 0)

    def _move_forward_with_base_correction(
        self,
        base_det: TargetDetection | None,
    ) -> None:
        if base_det is None:
            self._move_forward()
            return

        error_x = base_det.uv[0] - self.base_center_x
        omega = 0
        forward_speed = self.approach_speed

        if abs(error_x) > BASE_APPROACH_CORRECTION_DEADZONE_PX:
            omega = self._signed_speed(
                error_x,
                kp=BASE_ALIGN_KP,
                min_speed=BASE_APPROACH_CORRECTION_MIN_SPEED,
                max_speed=BASE_APPROACH_CORRECTION_MAX_SPEED,
                direction_sign=BASE_ALIGN_DIRECTION_SIGN,
            )
            if abs(error_x) >= BASE_APPROACH_SLOWDOWN_ERROR_PX:
                forward_speed = int(round(self.approach_speed * 0.6))

        self._drive_chassis(0, forward_speed, omega)

    def _move_with_wrist_correction(
        self,
        wrist_det: TargetDetection,
        clear_enough: bool,
        *,
        force_forward: bool = False,
    ) -> None:
        if self.wrist_visual_servo_mode == WRIST_SERVO_MODE_PROPORTIONAL:
            self._move_with_wrist_proportional(wrist_det, clear_enough, force_forward=force_forward)
            return

        error_x = wrist_det.uv[0] - self.wrist_target_x
        error_y = wrist_det.uv[1] - self.wrist_target_y

        if abs(error_x) > WRIST_ALIGN_DEADZONE_PX:
            speed = self._signed_speed(
                error_x,
                kp=WRIST_ALIGN_KP,
                min_speed=self.wrist_align_min_speed,
                max_speed=self.wrist_align_max_speed,
                direction_sign=WRIST_ALIGN_DIRECTION_SIGN,
            )
            self._spin_base(speed)
            return

        if force_forward or not clear_enough or abs(error_y) > self.wrist_ready_center_y_deadzone_px:
            forward_speed = int(
                round(self.approach_speed * WRIST_SEEN_FORWARD_SPEED_RATIO)
            )
            if clear_enough or force_forward:
                forward_speed = int(
                    round(self.approach_speed * WRIST_Y_ADJUST_SPEED_RATIO)
                )
            if not force_forward and error_y > self.wrist_ready_center_y_deadzone_px:
                forward_speed = -forward_speed
            self._drive_chassis(0, forward_speed, 0)
            return

        self._stop_wheels()

    def _move_with_wrist_proportional(
        self,
        wrist_det: TargetDetection,
        clear_enough: bool,
        *,
        force_forward: bool = False,
    ) -> None:
        error_x = wrist_det.uv[0] - self.wrist_target_x
        error_y = wrist_det.uv[1] - self.wrist_target_y
        omega = 0
        vy = 0

        if abs(error_x) > WRIST_ALIGN_DEADZONE_PX:
            omega = self._signed_speed(
                error_x,
                kp=WRIST_ALIGN_KP,
                min_speed=self.wrist_align_min_speed,
                max_speed=self.wrist_align_max_speed,
                direction_sign=WRIST_ALIGN_DIRECTION_SIGN,
            )

        if force_forward:
            vy = int(round(self.approach_speed * WRIST_Y_ADJUST_SPEED_RATIO))
        elif not clear_enough:
            vy = int(round(self.approach_speed * WRIST_SEEN_FORWARD_SPEED_RATIO))
        elif abs(error_y) > self.wrist_ready_center_y_deadzone_px:
            y_speed = self._signed_speed(
                error_y,
                kp=WRIST_Y_ALIGN_KP,
                min_speed=self.wrist_align_min_speed,
                max_speed=int(round(self.approach_speed * WRIST_Y_ADJUST_SPEED_RATIO)),
                direction_sign=1,
            )
            vy = -y_speed

        if omega == 0 and vy == 0:
            self._stop_wheels()
            return

        self._drive_chassis(0, vy, omega)

    def _move_backward_for_wrist_fit(self) -> None:
        backward_speed = -int(round(self.approach_speed * WRIST_Y_ADJUST_SPEED_RATIO))
        self._drive_chassis(0, backward_speed, 0)

    def _move_to_improve_wrist_visibility(self, wrist_det: TargetDetection) -> None:
        x1, y1, x2, y2 = wrist_det.xyxy
        sx1, sy1, sx2, sy2 = self._wrist_safe_box()
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)

        if box_w >= (sx2 - sx1) or box_h >= (sy2 - sy1):
            self._move_backward_for_wrist_fit()
            return

        error_x = wrist_det.uv[0] - self.wrist_target_x
        error_y = wrist_det.uv[1] - self.wrist_target_y
        has_center_error = (
            abs(error_x) > WRIST_ALIGN_DEADZONE_PX
            or abs(error_y) > self.wrist_ready_center_y_deadzone_px
        )
        if has_center_error:
            self._move_with_wrist_correction(wrist_det, clear_enough=True)
            return

        self._move_backward_for_wrist_fit()

    def _stop_wheels(self) -> None:
        self._drive_chassis(0, 0, 0, acc=255)

    def _drive_chassis(self, vx: float, vy: float, omega: float, acc: int = 50) -> None:
        """
        Three-wheel omni chassis kinematics.

        vx: lateral motion, positive means right in the reference teleop script.
        vy: forward motion, positive means forward in the reference teleop script.
        omega: positive means counterclockwise in the reference teleop script.

        The real chassis has front/back and left/right reversed relative to that script,
        so x/y signs are configurable at the constants above.
        """
        vx = float(vx) * CHASSIS_X_SIGN
        vy = float(vy) * CHASSIS_Y_SIGN
        omega = float(omega) * CHASSIS_OMEGA_SIGN

        v1 = -vx + omega
        v2 = 0.5 * vx - 0.866 * vy + omega
        v3 = 0.5 * vx + 0.866 * vy + omega
        command = tuple(int(round(speed)) for speed in (v1, v2, v3))
        if command != self._last_chassis_command and any(abs(speed) > 0 for speed in command):
            print(
                "chassis command "
                f"vx={vx:.1f}, vy={vy:.1f}, omega={omega:.1f} -> "
                f"{dict(zip(CHASSIS_MOTOR_IDS, command))}"
            )
        self._last_chassis_command = command
        for servo_id, speed in zip(CHASSIS_MOTOR_IDS, command):
            self.arm.spin(servo_id, speed, acc=acc)

    def _init_chassis_wheels(self) -> None:
        for servo_id in CHASSIS_MOTOR_IDS:
            self.arm.set_mode(servo_id, 1)
            self.arm._send_write(servo_id, 40, [1])
            self.arm.spin(servo_id, 0)
            time.sleep(0.01)

    def _show_debug(
        self,
        base_frame: np.ndarray,
        wrist_frame: np.ndarray,
        base_det: TargetDetection | None,
        wrist_det: TargetDetection | None,
        state: str,
        stable_frames: int,
    ) -> bool:
        base_vis = self._draw_debug_frame(
            base_frame.copy(),
            base_det,
            center=(self.base_center_x, self.base_center_y),
            state=state,
            stable_frames=stable_frames,
            name="base",
        )
        wrist_vis = self._draw_debug_frame(
            wrist_frame.copy(),
            wrist_det,
            center=(self.wrist_center_x, self.wrist_center_y),
            target=(self.wrist_target_x, self.wrist_target_y),
            state=state,
            stable_frames=stable_frames,
            name="wrist",
        )
        cv2.imshow("Grasp Pipeline - Base", base_vis)
        cv2.imshow("Grasp Pipeline - Wrist", wrist_vis)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def _draw_debug_frame(
        self,
        frame: np.ndarray,
        detection: TargetDetection | None,
        *,
        center: tuple[float, float],
        state: str,
        stable_frames: int,
        name: str,
        target: tuple[float, float] | None = None,
    ) -> np.ndarray:
        cx, cy = center
        cv2.drawMarker(
            frame,
            (int(cx), int(cy)),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        if name == "wrist":
            tx, ty = target if target is not None else center
            cv2.drawMarker(
                frame,
                (int(tx), int(ty)),
                (0, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                markerSize=24,
                thickness=2,
            )
            cv2.rectangle(
                frame,
                (
                    int(tx - self.wrist_ready_center_x_deadzone_px),
                    int(ty - self.wrist_ready_center_y_deadzone_px),
                ),
                (
                    int(tx + self.wrist_ready_center_x_deadzone_px),
                    int(ty + self.wrist_ready_center_y_deadzone_px),
                ),
                (255, 180, 0),
                1,
            )
            if self.wrist_min_box_visible_ratio > 0:
                sx1, sy1, sx2, sy2 = self._wrist_safe_box()
                cv2.rectangle(
                    frame,
                    (int(sx1), int(sy1)),
                    (int(sx2), int(sy2)),
                    (0, 220, 80),
                    1,
                )
        if detection is not None:
            x1, y1, x2, y2 = detection.xyxy
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )
            cv2.drawMarker(
                frame,
                (int(detection.uv[0]), int(detection.uv[1])),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            label = (
                f"{detection.source} conf={detection.conf:.2f} "
                f"area={detection.box_area_ratio:.3f}"
            )
            if name == "wrist":
                (
                    clear,
                    centered,
                    overlap,
                    overlap_ratio,
                    visible,
                    not_too_large,
                ) = self._wrist_ready_metrics(detection)
                position_ready = self._wrist_position_ready(centered, overlap)
                ready = clear and position_ready and visible and not_too_large
                preferred_ready = (
                    self.wrist_preferred_box_area_ratio <= 0
                    or detection.box_area_ratio >= self.wrist_preferred_box_area_ratio
                )
                visible_ratio = self._wrist_safe_visible_ratio(detection)
                label += (
                    f" overlap={overlap_ratio:.2f} "
                    f"vis={visible_ratio:.2f} "
                    f"clear={'Y' if clear else 'N'} "
                    f"center={'Y' if centered else 'N'} "
                    f"fit={'Y' if visible else 'N'} "
                    f"max={'Y' if not_too_large else 'N'} "
                    f"mode={self.wrist_ready_position_mode} "
                    f"ctrl={self.wrist_visual_servo_mode} "
                    f"ready={'Y' if ready else 'N'} "
                    f"pref={'Y' if preferred_ready else 'N'}"
                )
            cv2.putText(
                frame,
                label,
                (int(x1), max(24, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{name} state={state} stable={stable_frames}/{WRIST_READY_STABLE_FRAMES}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        return frame


def run_wheel_test(
    *,
    vx: float,
    vy: float,
    omega: float,
    duration_s: float,
    dry_run: bool,
) -> None:
    arm = DryRunServoController() if dry_run else ServoController()
    try:
        for servo_id in CHASSIS_MOTOR_IDS:
            arm.set_mode(servo_id, 1)
            arm._send_write(servo_id, 40, [1])
            arm.spin(servo_id, 0)
            time.sleep(0.01)
        print(
            f"Testing chassis vx={vx}, vy={vy}, omega={omega}, "
            f"duration={duration_s:.2f}s"
        )
        vx = float(vx) * CHASSIS_X_SIGN
        vy = float(vy) * CHASSIS_Y_SIGN
        omega = float(omega) * CHASSIS_OMEGA_SIGN
        v1 = -vx + omega
        v2 = 0.5 * vx - 0.866 * vy + omega
        v3 = 0.5 * vx + 0.866 * vy + omega
        for servo_id, speed in zip(CHASSIS_MOTOR_IDS, (v1, v2, v3)):
            arm.spin(servo_id, int(round(speed)))
        time.sleep(duration_s)
    finally:
        for servo_id in CHASSIS_MOTOR_IDS:
            arm.spin(servo_id, 0, acc=255)
        if hasattr(arm, "_ser"):
            arm._ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move the chassis until a target is centered and clear in the wrist camera."
    )
    parser.add_argument("--target", default=None, help="YOLO class name, for example: bottle or cup.")
    parser.add_argument("--model", default=str(BASE_DIR / "yolo11s.pt"), help="YOLO model path.")
    parser.add_argument("--timeout", type=float, default=BASE_SEARCH_TIMEOUT_S)
    parser.add_argument("--show", action="store_true", help="Show base and wrist debug windows.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send wheel commands.")
    parser.add_argument("--hand-camera-id", type=int, default=3, help="OpenCV index for the wrist camera.")
    parser.add_argument("--base-camera-id", type=int, default=0, help="OpenCV index for the base camera.")
    parser.add_argument(
        "--frame-mode",
        choices=(FRAME_MODE_CROP, FRAME_MODE_LETTERBOX),
        default=FRAME_MODE_CROP,
        help="Navigation image preprocessing. crop keeps the old square view; letterbox preserves aspect ratio.",
    )
    parser.add_argument("--frame-width", type=int, default=NAV_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=NAV_FRAME_HEIGHT)
    parser.add_argument(
        "--approach-speed",
        type=int,
        default=WHEEL_APPROACH_SPEED,
        help="Chassis forward vy speed used when approaching the target.",
    )
    parser.add_argument("--wrist-target-offset-x", type=float, default=WRIST_TARGET_OFFSET_X_PX)
    parser.add_argument("--wrist-target-offset-y", type=float, default=WRIST_TARGET_OFFSET_Y_PX)
    parser.add_argument("--wrist-ready-deadzone-x", type=float, default=WRIST_READY_CENTER_X_DEADZONE_PX)
    parser.add_argument("--wrist-ready-deadzone-y", type=float, default=WRIST_READY_CENTER_Y_DEADZONE_PX)
    parser.add_argument("--wrist-ready-min-overlap", type=float, default=WRIST_READY_MIN_BOX_OVERLAP_RATIO)
    parser.add_argument(
        "--wrist-ready-position-mode",
        choices=("center_or_overlap", "center", "overlap"),
        default="center_or_overlap",
    )
    parser.add_argument("--wrist-preferred-box-area", type=float, default=0.0)
    parser.add_argument("--wrist-preferred-timeout-s", type=float, default=0.0)
    parser.add_argument("--wrist-min-box-visible-ratio", type=float, default=0.0)
    parser.add_argument("--wrist-visible-margin-px", type=float, default=0.0)
    parser.add_argument("--wrist-max-box-area", type=float, default=0.0)
    parser.add_argument("--wrist-min-conf", type=float, default=WRIST_MIN_CONF)
    parser.add_argument("--wrist-min-box-area", type=float, default=WRIST_MIN_BOX_AREA_RATIO)
    parser.add_argument(
        "--wrist-visual-servo-mode",
        choices=(WRIST_SERVO_MODE_SEQUENTIAL, WRIST_SERVO_MODE_PROPORTIONAL),
        default=WRIST_SERVO_MODE_SEQUENTIAL,
    )
    parser.add_argument("--wheel-search-speed", type=int, default=WHEEL_SEARCH_SPEED)
    parser.add_argument("--wrist-align-max-speed", type=int, default=WRIST_ALIGN_MAX_SPEED)
    parser.add_argument("--wrist-align-min-speed", type=int, default=WRIST_ALIGN_MIN_SPEED)
    parser.add_argument("--wheel-test", action="store_true", help="Only test chassis vx/vy/omega briefly.")
    parser.add_argument("--wheel-test-vx", type=float, default=0.0)
    parser.add_argument("--wheel-test-vy", type=float, default=0.0)
    parser.add_argument("--wheel-test-omega", type=float, default=0.0)
    parser.add_argument("--wheel-test-duration", type=float, default=0.8)
    args = parser.parse_args()

    if args.wheel_test:
        run_wheel_test(
            vx=args.wheel_test_vx,
            vy=args.wheel_test_vy,
            omega=args.wheel_test_omega,
            duration_s=args.wheel_test_duration,
            dry_run=args.dry_run,
        )
        return

    if not args.target:
        parser.error("--target is required unless --wheel-test is used.")

    pipeline = GraspPipeline(
        model_path=args.model,
        timeout_s=args.timeout,
        show=args.show,
        dry_run=args.dry_run,
        approach_speed=args.approach_speed,
        hand_camera_id=args.hand_camera_id,
        base_camera_id=args.base_camera_id,
        frame_mode=args.frame_mode,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        wrist_target_offset_x_px=args.wrist_target_offset_x,
        wrist_target_offset_y_px=args.wrist_target_offset_y,
        wrist_ready_center_x_deadzone_px=args.wrist_ready_deadzone_x,
        wrist_ready_center_y_deadzone_px=args.wrist_ready_deadzone_y,
        wrist_ready_min_box_overlap_ratio=args.wrist_ready_min_overlap,
        wrist_ready_position_mode=args.wrist_ready_position_mode,
        wrist_preferred_box_area_ratio=args.wrist_preferred_box_area,
        wrist_preferred_timeout_s=args.wrist_preferred_timeout_s,
        wrist_min_box_visible_ratio=args.wrist_min_box_visible_ratio,
        wrist_visible_margin_px=args.wrist_visible_margin_px,
        wrist_max_box_area_ratio=args.wrist_max_box_area,
        wrist_min_conf=args.wrist_min_conf,
        wrist_min_box_area_ratio=args.wrist_min_box_area,
        wrist_visual_servo_mode=args.wrist_visual_servo_mode,
        wheel_search_speed=args.wheel_search_speed,
        wrist_align_max_speed=args.wrist_align_max_speed,
        wrist_align_min_speed=args.wrist_align_min_speed,
    )
    try:
        pipeline.run(args.target)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
