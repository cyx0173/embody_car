from pathlib import Path
import cv2
import numpy as np
import time
from ultralytics import YOLO
from arm_control import ServoController
from Angle_config import ArmManager

BASE_DIR = Path(__file__).resolve().parent

HAND_CAMERA_ID = 1
BASE_CAMERA_ID = 0
SEARCH_AXIS_ID = 1
SEARCH_SPEED = 200
BASE_ALIGN_DEADZONE_RATIO = 0.18
BASE_ALIGN_MIN_DEADZONE_PX = 120
BASE_ALIGN_STABLE_FRAMES = 1
BASE_ALIGN_MIN_SPEED = 40
BASE_ALIGN_MAX_SPEED = 140
BASE_ALIGN_KP = 0.35
BASE_ALIGN_DIRECTION_SIGN = 1
TRACK_DEADZONE_RATIO = 0.08
TRACK_MIN_DEADZONE_PX = 60
TRACK_KP = 0.2
TRACK_MAX_SPEED = 300
TRACK_LOST_MAX_FRAMES = 30
WRIST_FLEX_TRACK_TICKS = 1285
WRIST_ROLL_TRACK_TICKS = 69
TRACK_LOOP_INTERVAL_S = 0.02


class VisualTracking:
    def __init__(
        self,
        model_path: str | None = None,
        camera_id: int = HAND_CAMERA_ID,
        base_camera_id: int = BASE_CAMERA_ID,
    ) -> None:
        if model_path is None:
            model_path = str(BASE_DIR / "yolo11s.pt")
        self.cap = cv2.VideoCapture(camera_id)
        self.base_cap = cv2.VideoCapture(base_camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"手部相机打开失败: camera_id={camera_id}")
        if not self.base_cap.isOpened():
            raise RuntimeError(f"底座相机打开失败: base_camera_id={base_camera_id}")
        self.model = YOLO(model_path)
        self.arm = ServoController()
        self.arm_manager = ArmManager()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头，无法获取分辨率")
        self.center_x = frame.shape[1] // 2
        self.center_y = frame.shape[0] // 2

        ret, base_frame = self.base_cap.read()
        if not ret:
            raise RuntimeError("无法读取底座相机，无法获取分辨率")
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
        self.running = False
        self.is_tracking = False
        self.arm4_tracking_orignal_angle = WRIST_FLEX_TRACK_TICKS
        self.arm5_tracking_orignal_angle = WRIST_ROLL_TRACK_TICKS

    def run(self, target_class: str):
        self.running = True
        print(f"🚀 开始任务：正在寻找 {target_class}...")

        stable_base_frames = 0
        try:
            while self.running:
                base_img = self._capture_base()
                base_uv = self.detect_object(base_img, target_class)
 
                if base_uv is None:
                    stable_base_frames = 0
                    self._search_move()
                    time.sleep(TRACK_LOOP_INTERVAL_S)
                    continue

                if self._align_base_camera(base_uv):
                    stable_base_frames += 1
                    if stable_base_frames >= BASE_ALIGN_STABLE_FRAMES:
                        self.scan_state = "RIGHT"
                        self.arm.brake(SEARCH_AXIS_ID)
                        self._handle_tracking(target_class)
                        stable_base_frames = 0
                else:
                    stable_base_frames = 0

                time.sleep(TRACK_LOOP_INTERVAL_S)

        finally:
            self.arm.brake_all()
            self.cap.release()
            self.base_cap.release()
            cv2.destroyAllWindows()

    def track(self, target_class: str):
        self.running = True
        self.run(target_class)

    def _search_move(self):
        self.arm.brake(4)
        self.arm.brake(5)

        if self.scan_state == "RIGHT":
            speed = self.scan_speed
            next_state = "LEFT"
            danger = "right"
        else:
            speed = -self.scan_speed
            next_state = "RIGHT"
            danger = "left"

        is_safe, danger_side = self.arm_manager.safe_detect(SEARCH_AXIS_ID, self.arm)
        if not is_safe and danger_side == danger:
            self.arm.brake(SEARCH_AXIS_ID)
            self.scan_state = next_state
            return

        self.arm.spin(SEARCH_AXIS_ID, speed)

    def _align_base_camera(self, base_uv: tuple[float, float]) -> bool:
        error_x = float(base_uv[0]) - self.base_center_x
        if abs(error_x) <= self.base_align_deadzone:
            self.arm.brake(SEARCH_AXIS_ID)
            print(
                "底座相机粗对准完成，"
                f"error_x={error_x:.1f}px, deadzone={self.base_align_deadzone}px"
            )
            return True

        direction = BASE_ALIGN_DIRECTION_SIGN if error_x > 0 else -BASE_ALIGN_DIRECTION_SIGN
        speed = int(np.clip(abs(error_x) * BASE_ALIGN_KP, BASE_ALIGN_MIN_SPEED, BASE_ALIGN_MAX_SPEED))
        speed *= direction

        is_safe, danger_side = self.arm_manager.safe_detect(SEARCH_AXIS_ID, self.arm)
        if not is_safe and ((danger_side == "right" and speed > 0) or (danger_side == "left" and speed < 0)):
            self.arm.brake(SEARCH_AXIS_ID)
            self.scan_state = "LEFT" if speed > 0 else "RIGHT"
            return False

        self.scan_state = "RIGHT" if speed > 0 else "LEFT"
        print(f"底座相机粗对准中，target_x={base_uv[0]:.1f}, error_x={error_x:.1f}, speed={speed}")
        self.arm.spin(SEARCH_AXIS_ID, speed)
        return False

    def _speed_with_limit(self, servo_id: int, speed: int) -> int:
        if speed == 0:
            return 0

        safe, danger = self.arm_manager.safe_detect(servo_id, self.arm)
        if not safe and ((danger == "right" and speed > 0) or (danger == "left" and speed < 0)):
            return 0
        return speed

    def _tracking_speed(self, error_px: float, deadzone_px: int) -> int:
        if abs(error_px) <= deadzone_px:
            return 0
        return int(np.clip(error_px * self.kp_wrist, -TRACK_MAX_SPEED, TRACK_MAX_SPEED))

    def _handle_tracking(self, target_class: str): 
        print(f"🎯 锁定目标开启精细移动{target_class}")
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
                    self.arm.brake(4)
                    self.arm.brake(5)
                    if lost_frames > self.max_lost_allow:
                        print("⚠️ 目标连续丢失过久，退出追踪模式，恢复扫视")
                        return
                else:
                    lost_frames = 0
                    cx, cy = current_uv
                    err_x = cx - self.center_x
                    err_y = cy - self.center_y

                    v5 = self._speed_with_limit(5, self._tracking_speed(err_x, self.track_deadzone_x))
                    v4 = self._speed_with_limit(4, self._tracking_speed(err_y, self.track_deadzone_y))

                    self.arm.spin(4, v4)
                    self.arm.spin(5, v5)

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
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return frame
    
    def _capture_base(self) -> np.ndarray:
        ret, frame = self.base_cap.read()
        if not ret:
            raise RuntimeError("Base camera read failed")
        return frame
        

    def detect_object(self, img: np.ndarray, target_class: str) -> tuple[float, float] | None:
        results = self.model(img, verbose=False)
        best_target: tuple[float, np.ndarray] | None = None
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue
                cls_name = self.model.names[int(box.cls[0])]
                if cls_name == target_class:
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
    tracker = VisualTracking()
    time.sleep(2)
    tracker.track(target_class="person")
