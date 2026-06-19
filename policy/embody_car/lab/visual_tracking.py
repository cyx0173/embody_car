from pathlib import Path
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from arm_control import ServoController
from camera import CameraManager

BASE_DIR = Path(__file__).resolve().parent
SEARCH_AXIS_ID = 1
SEARCH_SPEED = 200
SEARCH_MOVE_SPEED = int(os.getenv("TRACK_SEARCH_MOVE_SPEED", "450"))
SEARCH_MOVE_ACC = int(os.getenv("TRACK_SEARCH_MOVE_ACC", "20"))
SEARCH_LEFT_TICK = int(os.getenv("TRACK_SEARCH_LEFT_TICK", "1100"))
SEARCH_RIGHT_TICK = int(os.getenv("TRACK_SEARCH_RIGHT_TICK", "3000"))
SEARCH_EDGE_TOLERANCE_TICKS = 90
BASE_SEARCH_INTERVAL_S = float(os.getenv("TRACK_BASE_SEARCH_INTERVAL_S", "0.08"))
BASE_ALIGN_DEADZONE_RATIO = 0.18
BASE_ALIGN_MIN_DEADZONE_PX = 120
BASE_ALIGN_STABLE_FRAMES = 1
BASE_ALIGN_MIN_SPEED = 40
BASE_ALIGN_MAX_SPEED = 140
BASE_ALIGN_KP = 0.35
BASE_ALIGN_DIRECTION_SIGN = 1
BASE_COARSE_ALIGN_ENABLED = os.getenv("TRACK_BASE_COARSE_ALIGN", "0") == "1"
TRACK_DEADZONE_RATIO = 0.08
TRACK_MIN_DEADZONE_PX = 60
TRACK_KP = float(os.getenv("TRACK_KP", "0.45"))
TRACK_MAX_SPEED = int(os.getenv("TRACK_MAX_SPEED", "650"))
TRACK_LOST_MAX_FRAMES = 30
TRACK_CONF_THRESHOLD = 0.2
TRACK_WRIST_FLEX_SIGN = int(os.getenv("TRACK_WRIST_FLEX_SIGN", "1"))
TRACK_WRIST_ROLL_SIGN = int(os.getenv("TRACK_WRIST_ROLL_SIGN", "1"))
HAND_LOST_SCAN_START_FRAMES = int(os.getenv("HAND_LOST_SCAN_START_FRAMES", "3"))
HAND_LOST_SCAN_SPEED_5 = int(os.getenv("HAND_LOST_SCAN_SPEED_5", "420"))
HAND_LOST_SCAN_SPEED_4 = int(os.getenv("HAND_LOST_SCAN_SPEED_4", "120"))
HAND_LOST_SCAN_SWITCH_FRAMES = int(os.getenv("HAND_LOST_SCAN_SWITCH_FRAMES", "12"))
BASE_FALLBACK_KP = float(os.getenv("TRACK_BASE_FALLBACK_KP", "0.25"))
BASE_FALLBACK_MAX_SPEED = int(os.getenv("TRACK_BASE_FALLBACK_MAX_SPEED", "220"))
WRIST_FLEX_TRACK_TICKS = 1285
WRIST_ROLL_TRACK_TICKS = 69
TRACK_LOOP_INTERVAL_S = float(os.getenv("TRACK_LOOP_INTERVAL_S", "0.01"))
SHOW_TRACKING_WINDOW = os.getenv("TRACK_SHOW_WINDOW", "0") == "1"
TRACK_DRY_RUN = os.getenv("TRACK_DRY_RUN", "0") == "1"

SERVO_LIMITS = {
    1: (715, 3466),
    4: (845, 3176),
    5: (0, 4095),
}
SERVO_LIMIT_MARGIN_TICKS = 80

#干运行 用于调试
class DryRunServoController:
    def __init__(self) -> None:
        self.positions = {
            1: 2000,
            4: WRIST_FLEX_TRACK_TICKS,
            5: WRIST_ROLL_TRACK_TICKS,
        }
        self.last_speeds: dict[int, int] = {}

    def spin(self, servo_id: int, speed: int, acc: int = 50) -> None:
        speed = int(speed)
        if self.last_speeds.get(servo_id) == speed:
            return
        self.last_speeds[servo_id] = speed
        print(f"[DRY RUN] spin servo {servo_id}: speed={speed}, acc={acc}")

    def move_to(self, servo_id: int, pos: int, speed: int = 4000, acc: int = 50) -> None:
        pos = int(pos)
        self.positions[servo_id] = pos
        self.last_speeds[servo_id] = 0
        print(f"[DRY RUN] move_to servo {servo_id}: pos={pos}, speed={speed}, acc={acc}")

    def brake(self, servo_id: int) -> None:
        self.spin(servo_id, 0, acc=255)

    def brake_all(self) -> None:
        for servo_id in (1, 4, 5):
            self.brake(servo_id)

    def get_position(self, servo_id: int) -> int:
        return self.positions.get(servo_id, 2000)


class VisualTracking:
    def __init__(
        self,
        camera: CameraManager,
        model_path: str | None = None,
    ) -> None:
        if model_path is None:
            model_path = str(BASE_DIR / "yolo11s.pt")
        self.camera = camera
        self.model = YOLO(model_path)
        self.arm = DryRunServoController() if TRACK_DRY_RUN else ServoController()

        frame = self._capture()
        self.center_x = frame.shape[1] // 2
        self.center_y = frame.shape[0] // 2

        base_frame = self._capture_base()
        self.base_center_x = base_frame.shape[1] // 2
        self.base_align_deadzone = max(
            BASE_ALIGN_MIN_DEADZONE_PX,
            int(base_frame.shape[1] * BASE_ALIGN_DEADZONE_RATIO),
        )

        self.track_deadzone_x = max(
            TRACK_MIN_DEADZONE_PX,
            int(frame.shape[1] * TRACK_DEADZONE_RATIO),
        )
        self.track_deadzone_y = max(
            TRACK_MIN_DEADZONE_PX,
            int(frame.shape[0] * TRACK_DEADZONE_RATIO),
        )
        self.kp_wrist = TRACK_KP
        self.max_lost_allow = TRACK_LOST_MAX_FRAMES

        self.scan_state = "RIGHT"
        self.scan_speed = SEARCH_SPEED
        self.scan_target_tick: int | None = None
        self.hand_scan_state = "RIGHT"
        self.running = False
        self.is_tracking = False
        self.arm4_tracking_orignal_angle = WRIST_FLEX_TRACK_TICKS
        self.arm5_tracking_orignal_angle = WRIST_ROLL_TRACK_TICKS

    def run(self, target_class: str):
        self.running = True

        stable_base_frames = 0
        try:
            while self.running:
                base_img = self._capture_base()
                base_uv = self.detect_object(base_img, target_class)
                if SHOW_TRACKING_WINDOW:
                    cv2.circle(
                        base_img,
                        (int(self.base_center_x), int(base_img.shape[0] / 2)),
                        8,
                        (255, 0, 0),
                        -1,
                    )
                    cv2.imshow("Base Camera Search", base_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
 
                if base_uv is None:
                    stable_base_frames = 0
                    self._search_move()
                    time.sleep(BASE_SEARCH_INTERVAL_S)
                    continue

                if BASE_COARSE_ALIGN_ENABLED:
                    if self._align_base_camera(base_uv):
                        stable_base_frames += 1
                        if stable_base_frames >= BASE_ALIGN_STABLE_FRAMES:
                            self.scan_state = "RIGHT"
                            self.arm.brake(SEARCH_AXIS_ID)
                            self._handle_tracking(target_class)
                            stable_base_frames = 0
                    else:
                        stable_base_frames = 0
                else:
                    self.scan_state = "RIGHT"
                    self.arm.brake(SEARCH_AXIS_ID)
                    self._handle_tracking(target_class)
                    stable_base_frames = 0

                time.sleep(TRACK_LOOP_INTERVAL_S)

        finally:
            self.arm.brake(SEARCH_AXIS_ID)
            self.arm.brake(4)
            self.arm.brake(5)
            self.release_cameras()
            if SHOW_TRACKING_WINDOW:
                cv2.destroyAllWindows()

    def track(self, target_class: str):
        self.running = True
        self.run(target_class)

    def stop(self):
        self.running = False
        self.arm.brake(SEARCH_AXIS_ID)
        self.arm.brake(4)
        self.arm.brake(5)

    def release_cameras(self) -> None:
        """No-op: cameras are shared with other components, don't release here."""
        pass

    def _search_move(self):
        self.arm.brake(4)
        self.arm.brake(5)

        pos = self.arm.get_position(SEARCH_AXIS_ID)
        if pos < 0:
            self.arm.spin(SEARCH_AXIS_ID, self.scan_speed)
            return

        if self.scan_target_tick is None:
            self.scan_target_tick = SEARCH_RIGHT_TICK if pos < (SEARCH_LEFT_TICK + SEARCH_RIGHT_TICK) / 2 else SEARCH_LEFT_TICK
            self.scan_state = "RIGHT" if self.scan_target_tick == SEARCH_RIGHT_TICK else "LEFT"

        if abs(pos - self.scan_target_tick) <= SEARCH_EDGE_TOLERANCE_TICKS:
            self.scan_target_tick = SEARCH_LEFT_TICK if self.scan_target_tick == SEARCH_RIGHT_TICK else SEARCH_RIGHT_TICK
            self.scan_state = "RIGHT" if self.scan_target_tick == SEARCH_RIGHT_TICK else "LEFT"

        self.arm.move_to(
            SEARCH_AXIS_ID,
            self.scan_target_tick,
            speed=SEARCH_MOVE_SPEED,
            acc=SEARCH_MOVE_ACC,
        )

    def _align_base_camera(self, base_uv: tuple[float, float]) -> bool:
        error_x = float(base_uv[0]) - self.base_center_x
        if abs(error_x) <= self.base_align_deadzone:
            self.arm.brake(SEARCH_AXIS_ID)
            return True

        direction = BASE_ALIGN_DIRECTION_SIGN if error_x > 0 else -BASE_ALIGN_DIRECTION_SIGN
        speed = int(np.clip(abs(error_x) * BASE_ALIGN_KP, BASE_ALIGN_MIN_SPEED, BASE_ALIGN_MAX_SPEED))
        speed *= direction

        speed, _blocked = self._speed_with_tick_limit(SEARCH_AXIS_ID, speed)
        if speed == 0:
            self.arm.brake(SEARCH_AXIS_ID)
            self.scan_state = "LEFT" if direction > 0 else "RIGHT"
            return False

        self.scan_state = "RIGHT" if speed > 0 else "LEFT"
        self.arm.spin(SEARCH_AXIS_ID, speed)
        return False

    def _speed_with_tick_limit(self, servo_id: int, speed: int) -> tuple[int, bool]:
        if speed == 0:
            return 0, False

        limits = SERVO_LIMITS.get(servo_id)
        if limits is None:
            return speed, False

        pos = self.arm.get_position(servo_id)
        if pos < 0:
            return speed, False

        lo, hi = limits
        if pos <= lo + SERVO_LIMIT_MARGIN_TICKS and speed < 0:
            print(f"舵机 {servo_id} 接近下限 pos={pos}, 阻止 speed={speed}")
            return 0, True
        if pos >= hi - SERVO_LIMIT_MARGIN_TICKS and speed > 0:
            print(f"舵机 {servo_id} 接近上限 pos={pos}, 阻止 speed={speed}")
            return 0, True
        return speed, False

    def _base_fallback_speed(self, err_x: float) -> int:
        if abs(err_x) <= self.track_deadzone_x:
            return 0
        speed = int(np.clip(err_x * BASE_FALLBACK_KP, -BASE_FALLBACK_MAX_SPEED, BASE_FALLBACK_MAX_SPEED))
        speed *= BASE_ALIGN_DIRECTION_SIGN
        speed, _blocked = self._speed_with_tick_limit(SEARCH_AXIS_ID, speed)
        return speed

    def _tracking_speed(self, error_px: float, deadzone_px: int) -> int:
        if abs(error_px) <= deadzone_px:
            return 0
        return int(np.clip(error_px * self.kp_wrist, -TRACK_MAX_SPEED, TRACK_MAX_SPEED))

    def _scan_hand_camera_when_lost(self, lost_frames: int) -> None:
        if lost_frames < HAND_LOST_SCAN_START_FRAMES:
            self.arm.brake(4)
            self.arm.brake(5)
            return

        if lost_frames % HAND_LOST_SCAN_SWITCH_FRAMES == 0:
            self.hand_scan_state = "LEFT" if self.hand_scan_state == "RIGHT" else "RIGHT"

        speed5 = HAND_LOST_SCAN_SPEED_5 if self.hand_scan_state == "RIGHT" else -HAND_LOST_SCAN_SPEED_5
        speed5, _blocked5 = self._speed_with_tick_limit(5, speed5)
        if speed5 == 0:
            self.hand_scan_state = "LEFT" if self.hand_scan_state == "RIGHT" else "RIGHT"
            speed5 = HAND_LOST_SCAN_SPEED_5 if self.hand_scan_state == "RIGHT" else -HAND_LOST_SCAN_SPEED_5
            speed5, _blocked5 = self._speed_with_tick_limit(5, speed5)

        speed4 = 0
        if lost_frames >= HAND_LOST_SCAN_SWITCH_FRAMES:
            speed4 = HAND_LOST_SCAN_SPEED_4 if (lost_frames // HAND_LOST_SCAN_SWITCH_FRAMES) % 2 == 0 else -HAND_LOST_SCAN_SPEED_4
            speed4, _blocked4 = self._speed_with_tick_limit(4, speed4)

        self.arm.spin(4, speed4)
        self.arm.spin(5, speed5)

    def _handle_tracking(self, target_class: str):
        lost_frames = 0
        self.is_tracking = True
        self.arm.move_to(4, self.arm4_tracking_orignal_angle)
        self.arm.move_to(5, self.arm5_tracking_orignal_angle)
        try:
            while self.running:
                img = self._capture()
                current_uv = self.detect_object(img, target_class)
                if current_uv is None:
                    lost_frames += 1
                    self._scan_hand_camera_when_lost(lost_frames)
                    if lost_frames > self.max_lost_allow:
                        return
                else:
                    lost_frames = 0
                    self.hand_scan_state = "RIGHT"
                    cx, cy = current_uv
                    err_x = cx - self.center_x
                    err_y = cy - self.center_y

                    raw_v5 = TRACK_WRIST_ROLL_SIGN * self._tracking_speed(err_x, self.track_deadzone_x)
                    raw_v4 = TRACK_WRIST_FLEX_SIGN * self._tracking_speed(err_y, self.track_deadzone_y)
                    v5, wrist_roll_blocked = self._speed_with_tick_limit(5, raw_v5)
                    v4, wrist_flex_blocked = self._speed_with_tick_limit(4, raw_v4)
                    v1 = self._base_fallback_speed(err_x) if wrist_roll_blocked and raw_v5 != 0 else 0
                    self.arm.spin(SEARCH_AXIS_ID, v1)
                    self.arm.spin(4, v4)
                    self.arm.spin(5, v5)

                if SHOW_TRACKING_WINDOW:
                    cv2.circle(img, (int(self.center_x), int(self.center_y)), 8, (255, 0, 0), -1)
                    cv2.imshow("Tracking System", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                time.sleep(TRACK_LOOP_INTERVAL_S)
        finally:
            self.is_tracking = False
            self.arm.brake(4)
            self.arm.brake(5)


    def _capture(self) -> np.ndarray:
        return self.camera.read_hand_raw()

    def _capture_base(self) -> np.ndarray:
        return self.camera.read_base_raw()

    def detect_object(self, img: np.ndarray, target_class: str) -> tuple[float, float] | None:
        results = self.model(img, verbose=False)
        best_target: tuple[float, np.ndarray] | None = None
        visible_classes: list[str] = []
        target_name = target_class.strip().lower()
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < TRACK_CONF_THRESHOLD:
                    continue
                cls_name = self.model.names[int(box.cls[0])]
                visible_classes.append(f"{cls_name}:{conf:.2f}")
                if cls_name.lower() == target_name:
                    xyxy = box.xyxy[0].cpu().numpy()
                    if best_target is None or conf > best_target[0]:
                        best_target = (conf, xyxy)

        if best_target is None:
            return None

        conf, xyxy = best_target
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])),
                      (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
        cv2.putText(img, f"{target_class} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return (float(cx), float(cy))


if __name__ == "__main__":
    os.environ.setdefault("TRACK_SHOW_WINDOW", "1")
    SHOW_TRACKING_WINDOW = os.getenv("TRACK_SHOW_WINDOW", "0") == "1"
    tracker = VisualTracking()
    time.sleep(2)
    tracker.track(target_class="bottle")
