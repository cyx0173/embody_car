from __future__ import annotations

import time

import cv2
import numpy as np
from ultralytics import YOLO

from Angle_config import ArmManager
from arm_control import ServoController


class VisualTracking:
    TARGET_ALIASES = {
        "person": "person",
        "人": "person",
        "行人": "person",
        "人体": "person",
        "bottle": "bottle",
        "瓶子": "bottle",
        "水瓶": "bottle",
        "cup": "cup",
        "杯子": "cup",
        "水杯": "cup",
        "chair": "chair",
        "椅子": "chair",
        "cell phone": "cell phone",
        "cellphone": "cell phone",
        "mobile phone": "cell phone",
        "phone": "cell phone",
        "手机": "cell phone",
    }

    SUPPORTED_TARGETS = ("person", "bottle", "cup", "chair", "cell phone")

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        camera_id: int = 0,
        base_camera_id: int = 1,
        servo_port: str | None = None,
        min_conf: float = 0.3,
        search_speed: int = 200,
        deadzone: int = 60,
        coarse_deadzone: int = 80,
        max_lost_allow: int = 30,
    ) -> None:
        self.model_path = model_path
        self.camera_id = camera_id
        self.base_camera_id = base_camera_id
        self.servo_port = servo_port
        self.min_conf = min_conf
        self.search_speed = int(search_speed)
        self.deadzone = int(deadzone)
        self.coarse_deadzone = int(coarse_deadzone)
        self.max_lost_allow = int(max_lost_allow)

        self.kp_wrist = 0.2
        self.coarse_kp = 0.35
        self.max_tracking_speed = 300
        self.max_fine_align_speed = 180
        self.max_coarse_speed = 260
        self.min_coarse_speed = 80
        self.base_handoff_offset_x = 0
        self.base_align_sign = 1
        self.coarse_lost_allow = 8
        self.coarse_lost_hold_frames = 3
        self.coarse_flip_margin = 18
        self.fine_align_deadzone = max(80, self.deadzone + 30)
        self.coarse_align_confirm_frames = 2
        self.arm_acquire_confirm_frames = 2
        self.arm_acquire_max_miss_frames = 90
        self.arm_acquire_phase_frames = 12
        self.arm_acquire_yaw_speed = 120
        self.arm_acquire_pitch_speed = 80
        self.fine_align_confirm_frames = 3
        self.fine_lost_allow = 20

        self.scan_state = "RIGHT"
        self.running = False
        self.arm4_tracking_original_angle = 1285
        self.arm5_tracking_original_angle = 69
        self.arm_manager = ArmManager()

        self.model = None
        self.cap = None
        self.base_cap = None
        self.arm = None

        self.center_x = 0
        self.center_y = 0
        self.base_center_x = 0
        self.base_center_y = 0
        self._tracking_pose_ready = False
        self._arm_acquire_phase = 0
        self._arm_acquire_phase_frame = 0
        self._prev_coarse_err_x = None
        self._coarse_wrong_dir_count = 0

    def track(self, target_class: str, current_image=None):
        del current_image

        normalized_target = self._normalize_target(target_class)
        if normalized_target is None:
            readable = " / ".join(self.SUPPORTED_TARGETS)
            print(f"❌ 暂不支持追踪目标: {target_class}，当前支持: {readable}")
            return

        try:
            self.run(normalized_target)
        except KeyboardInterrupt:
            print("\n已手动停止视觉追踪。")
        except Exception as exc:
            print(f"❌ 视觉追踪异常: {exc}")
        finally:
            self.running = False
            self._cleanup_resources()

    def run(self, target_class: str):
        self._initialize_resources()
        self.running = True
        self.scan_state = "RIGHT"
        self._tracking_pose_ready = False
        lost_frames = 0
        coarse_aligned_frames = 0
        coarse_lost_frames = 0
        arm_acquire_seen_frames = 0
        arm_acquire_miss_frames = 0
        fine_aligned_frames = 0
        fine_lost_frames = 0
        state = "SEARCH"

        print(f"🚀 开始任务：正在寻找 {target_class} ...")

        while self.running:
            if state == "SEARCH":
                base_img = self._capture_base()
                base_uv = self.detect_object(base_img, target_class, stage="base")
                self._draw_base_guides(base_img)
                cv2.imshow("Base Search", base_img)
                if self._user_requested_stop():
                    break

                if base_uv is None:
                    self._search_move()
                else:
                    self.arm.brake(1)
                    coarse_aligned_frames = 0
                    coarse_lost_frames = 0
                    self._reset_coarse_align_feedback()
                    print("👀 底座相机发现目标，开始粗对准")
                    state = "COARSE_ALIGN"
                continue

            if state == "COARSE_ALIGN":
                base_img = self._capture_base()
                base_uv = self.detect_object(base_img, target_class, stage="base")
                self._draw_base_guides(base_img)
                cv2.imshow("Base Search", base_img)
                if self._user_requested_stop():
                    break

                if base_uv is None:
                    coarse_lost_frames += 1
                    self.arm.brake(1)
                    if coarse_lost_frames > self.coarse_lost_allow:
                        state = self._recover_to_search("⚠️ 粗对准阶段目标丢失，恢复扫视")
                    continue

                coarse_lost_frames = 0
                aligned = self._coarse_align(base_uv)
                if aligned is None:
                    state = self._recover_to_search("⚠️ 粗对准触发底座限位，恢复扫视")
                    continue

                if aligned:
                    coarse_aligned_frames += 1
                    if coarse_aligned_frames >= self.coarse_align_confirm_frames:
                        self.arm.brake(1)
                        self._tracking_pose_ready = False
                        self._reset_arm_acquire_search()
                        arm_acquire_seen_frames = 0
                        arm_acquire_miss_frames = 0
                        fine_aligned_frames = 0
                        fine_lost_frames = 0
                        print("🎯 粗对准完成，切换到臂上摄像头接管搜索")
                        state = "ARM_ACQUIRE"
                else:
                    coarse_aligned_frames = 0
                continue

            if state == "ARM_ACQUIRE":
                if not self._tracking_pose_ready:
                    self._prepare_tracking_pose()

                img = self._capture()
                current_uv = self.detect_object(img, target_class, stage="arm")
                cv2.imshow("Tracking System", img)
                if self._user_requested_stop():
                    break

                if current_uv is None:
                    arm_acquire_seen_frames = 0
                    arm_acquire_miss_frames += 1
                    self._arm_acquire_search_step()
                    if arm_acquire_miss_frames > self.arm_acquire_max_miss_frames:
                        state = self._recover_to_search("⚠️ 臂上摄像头接管失败，恢复扫视")
                    continue

                arm_acquire_miss_frames = 0
                arm_acquire_seen_frames += 1
                self.arm.brake(4)
                self.arm.brake(5)
                if arm_acquire_seen_frames >= self.arm_acquire_confirm_frames:
                    fine_aligned_frames = 0
                    fine_lost_frames = 0
                    print("🎯 臂上摄像头已接管目标，开始细对准")
                    state = "FINE_ALIGN"
                continue

            if state == "FINE_ALIGN":
                if not self._tracking_pose_ready:
                    self._prepare_tracking_pose()

                img = self._capture()
                current_uv = self.detect_object(img, target_class, stage="arm")
                cv2.imshow("Tracking System", img)
                if self._user_requested_stop():
                    break

                if current_uv is None:
                    fine_lost_frames += 1
                    fine_aligned_frames = 0
                    self.arm.brake(4)
                    self.arm.brake(5)
                    if fine_lost_frames > self.fine_lost_allow:
                        state = self._recover_to_search("⚠️ 细对准阶段臂上摄像头丢失目标，恢复扫视")
                    continue

                fine_lost_frames = 0
                fine_aligned = self._fine_align(current_uv)
                if fine_aligned:
                    fine_aligned_frames += 1
                    if fine_aligned_frames >= self.fine_align_confirm_frames:
                        lost_frames = 0
                        print("🎯 细对准完成，进入连续跟踪")
                        state = "TRACK"
                else:
                    fine_aligned_frames = 0
                continue

            if state == "TRACK":
                if not self._tracking_pose_ready:
                    self._prepare_tracking_pose()

                img = self._capture()
                current_uv = self.detect_object(img, target_class, stage="arm")
                cv2.imshow("Tracking System", img)
                if self._user_requested_stop():
                    break

                if current_uv is None:
                    lost_frames += 1
                    self.arm.brake(4)
                    self.arm.brake(5)
                    if lost_frames > self.max_lost_allow:
                        state = self._recover_to_search("⚠️ 目标丢失过久，退出精追并恢复扫视")
                    continue

                lost_frames = 0
                self._apply_tracking_control(current_uv)
                continue

            raise RuntimeError(f"未知追踪状态: {state}")

    def _initialize_resources(self):
        self.model = self.model or YOLO(self.model_path)

        self.base_cap, base_frame = self._open_camera(self.base_camera_id, "base")
        self.cap, frame = self._open_camera(self.camera_id, "tracking")

        self.center_x = frame.shape[1] // 2
        self.center_y = frame.shape[0] // 2
        self.base_center_x = base_frame.shape[1] // 2
        self.base_center_y = base_frame.shape[0] // 2

        if self.servo_port:
            self.arm = ServoController(port=self.servo_port)
        else:
            self.arm = ServoController()
        self.arm.reset()

    def _cleanup_resources(self):
        if self.arm is not None:
            try:
                self.arm.brake_all()
            except Exception as exc:
                print(f"[cleanup] 刹车失败: {exc}")
            try:
                self.arm.close()
            except Exception:
                pass
            self.arm = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.base_cap is not None:
            self.base_cap.release()
            self.base_cap = None

        cv2.destroyAllWindows()

    def _open_camera(self, camera_id: int, camera_name: str):
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 {camera_name} 摄像头: {camera_id}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = None
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, frame
            time.sleep(0.1)

        cap.release()
        raise RuntimeError(f"无法读取 {camera_name} 摄像头画面: {camera_id}")

    def _normalize_target(self, target_class: str | None) -> str | None:
        if target_class is None:
            return None

        key = str(target_class).strip().lower()
        if not key:
            return None

        if key in self.TARGET_ALIASES:
            return self.TARGET_ALIASES[key]

        return key if key in self.SUPPORTED_TARGETS else None

    def _search_move(self):
        self.arm.brake(4)
        self.arm.brake(5)

        _, danger = self.arm_manager.safe_detect(1, self.arm)
        if self.scan_state == "RIGHT":
            self.arm.spin(1, self.search_speed)
            if danger == "right":
                self.arm.brake(1)
                self.scan_state = "LEFT"
        elif self.scan_state == "LEFT":
            self.arm.spin(1, -self.search_speed)
            if danger == "left":
                self.arm.brake(1)
                self.scan_state = "RIGHT"

    def _coarse_align(self, base_uv: tuple[float, float]) -> bool | None:
        err_x = base_uv[0] - self._base_handoff_center_x()
        if abs(err_x) <= self.coarse_deadzone:
            self.arm.brake(1)
            self._prev_coarse_err_x = None
            self._coarse_wrong_dir_count = 0
            return True

        if self._prev_coarse_err_x is not None:
            same_side = np.sign(err_x) == np.sign(self._prev_coarse_err_x)
            got_worse = abs(err_x) > abs(self._prev_coarse_err_x) + self.coarse_flip_margin
            got_better = abs(err_x) < abs(self._prev_coarse_err_x) - self.coarse_flip_margin

            if same_side and got_worse:
                self._coarse_wrong_dir_count += 1
                if self._coarse_wrong_dir_count >= 2:
                    self.base_align_sign *= -1
                    self._coarse_wrong_dir_count = 0
                    print(f"🔄 粗对准方向已修正: {self.base_align_sign:+d}")
            elif got_better:
                self._coarse_wrong_dir_count = 0

        speed = int(self.base_align_sign * err_x * self.coarse_kp)
        if abs(speed) < self.min_coarse_speed:
            speed = self.min_coarse_speed if err_x > 0 else -self.min_coarse_speed
            speed *= self.base_align_sign
        speed = int(np.clip(speed, -self.max_coarse_speed, self.max_coarse_speed))
        speed = self._guard_axis_speed(axis_id=1, speed=speed)
        if speed == 0:
            self.arm.brake(1)
            return None
        self._prev_coarse_err_x = err_x
        self.arm.spin(1, speed)
        return False

    def _base_handoff_center_x(self) -> float:
        return self.base_center_x + self.base_handoff_offset_x

    def _draw_base_guides(self, img: np.ndarray):
        handoff_x = int(self._base_handoff_center_x())
        center_y = int(self.base_center_y)
        deadzone = int(self.coarse_deadzone)
        h, w = img.shape[:2]

        cv2.line(img, (handoff_x, 0), (handoff_x, h), (0, 255, 255), 1)
        cv2.line(img, (0, center_y), (w, center_y), (0, 255, 255), 1)
        cv2.rectangle(
            img,
            (max(0, handoff_x - deadzone), 0),
            (min(w - 1, handoff_x + deadzone), h - 1),
            (255, 255, 0),
            1,
        )

    def _reset_arm_acquire_search(self):
        self._arm_acquire_phase = 0
        self._arm_acquire_phase_frame = 0

    def _reset_coarse_align_feedback(self):
        self._prev_coarse_err_x = None
        self._coarse_wrong_dir_count = 0

    def _arm_acquire_search_step(self):
        phases = (
            (0, self.arm_acquire_yaw_speed),
            (0, -self.arm_acquire_yaw_speed),
            (self.arm_acquire_pitch_speed, 0),
            (0, self.arm_acquire_yaw_speed),
            (0, -self.arm_acquire_yaw_speed),
            (-self.arm_acquire_pitch_speed, 0),
        )

        v4, v5 = phases[self._arm_acquire_phase]
        v4 = self._guard_axis_speed(axis_id=4, speed=v4)
        v5 = self._guard_axis_speed(axis_id=5, speed=v5)

        self.arm.spin(4, v4)
        self.arm.spin(5, v5)

        self._arm_acquire_phase_frame += 1
        if self._arm_acquire_phase_frame >= self.arm_acquire_phase_frames:
            self._arm_acquire_phase = (self._arm_acquire_phase + 1) % len(phases)
            self._arm_acquire_phase_frame = 0

    def _fine_align(self, current_uv: tuple[float, float]) -> bool:
        cx, cy = current_uv
        err_x = cx - self.center_x
        err_y = cy - self.center_y

        if abs(err_x) <= self.fine_align_deadzone and abs(err_y) <= self.fine_align_deadzone:
            self.arm.brake(4)
            self.arm.brake(5)
            return True

        v5 = int(err_x * self.kp_wrist) if abs(err_x) > self.fine_align_deadzone else 0
        v4 = int(err_y * self.kp_wrist) if abs(err_y) > self.fine_align_deadzone else 0

        v5 = int(np.clip(v5, -self.max_fine_align_speed, self.max_fine_align_speed))
        v4 = int(np.clip(v4, -self.max_fine_align_speed, self.max_fine_align_speed))
        v4 = self._guard_axis_speed(axis_id=4, speed=v4)
        v5 = self._guard_axis_speed(axis_id=5, speed=v5)

        self.arm.spin(4, v4)
        self.arm.spin(5, v5)
        return False

    def _prepare_tracking_pose(self):
        self.arm.move_to(4, self.arm4_tracking_original_angle)
        self.arm.move_to(5, self.arm5_tracking_original_angle)
        time.sleep(0.2)
        self._tracking_pose_ready = True

    def _apply_tracking_control(self, current_uv: tuple[float, float]):
        cx, cy = current_uv
        err_x = cx - self.center_x
        err_y = cy - self.center_y

        v5 = int(err_x * self.kp_wrist) if abs(err_x) > self.deadzone else 0
        v4 = int(err_y * self.kp_wrist) if abs(err_y) > self.deadzone else 0

        v5 = int(np.clip(v5, -self.max_tracking_speed, self.max_tracking_speed))
        v4 = int(np.clip(v4, -self.max_tracking_speed, self.max_tracking_speed))
        v4 = self._guard_axis_speed(axis_id=4, speed=v4)
        v5 = self._guard_axis_speed(axis_id=5, speed=v5)

        self.arm.spin(4, v4)
        self.arm.spin(5, v5)

    def _guard_axis_speed(self, axis_id: int, speed: int) -> int:
        if speed == 0:
            return 0

        safe, danger = self.arm_manager.safe_detect(axis_id, self.arm)
        if safe:
            return speed

        if (danger == "right" and speed > 0) or (danger == "left" and speed < 0):
            return 0
        return speed

    def _recover_to_search(self, reason: str) -> str:
        print(reason)
        self.arm.brake(1)
        self.arm.brake(4)
        self.arm.brake(5)
        self.scan_state = "RIGHT"
        self._tracking_pose_ready = False
        self._reset_arm_acquire_search()
        self._reset_coarse_align_feedback()
        return "SEARCH"

    def _user_requested_stop(self) -> bool:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False
            return True
        return False

    def _capture(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("tracking camera 未初始化")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return frame

    def _capture_base(self) -> np.ndarray:
        if self.base_cap is None:
            raise RuntimeError("base camera 未初始化")

        ret, frame = self.base_cap.read()
        if not ret:
            raise RuntimeError("Base camera read failed")
        return frame

    def detect_object(self, img: np.ndarray, target_class: str, stage: str = "arm") -> tuple[float, float] | None:
        results = self.model(img, verbose=False)
<<<<<<< HEAD
        best_match = None
        img_h, img_w = img.shape[:2]
        center_x = img_w / 2
        center_y = img_h / 2

        for result in results:
            for box in result.boxes:
=======
        for r in results:
            for box in r.boxes:
>>>>>>> origin/cyx
                conf = float(box.conf[0])
                if conf < self.min_conf:
                    continue

                cls_name = self.model.names[int(box.cls[0])]
                if cls_name != target_class:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2

                if stage == "base":
                    norm_dx = abs(obj_center_x - center_x) / max(center_x, 1.0)
                    norm_dy = abs(obj_center_y - center_y) / max(center_y, 1.0)
                    center_bonus = max(0.0, 1.0 - 0.7 * norm_dx - 0.3 * norm_dy)
                    score = 0.65 * conf + 0.35 * center_bonus
                else:
                    score = conf

                if best_match is None or score > best_match[0]:
                    best_match = (score, conf, xyxy, cls_name)

        if best_match is None:
            return None

        _, conf, xyxy, cls_name = best_match
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"{cls_name} {conf:.2f}",
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return float(cx), float(cy)


if __name__ == "__main__":
    tracker = VisualTracking()
    tracker.track(target_class="person")
