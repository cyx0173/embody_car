import cv2
import numpy as np
import time
from ultralytics import YOLO
from arm_control import ServoController
from Angle_config import ArmManager

class VisualTracking:
    def __init__(self, model_path: str = "yolov8s.pt", camera_id: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.model = YOLO(model_path)
        self.arm = ServoController()
        self.arm_manager = ArmManager()
        self.arm.reset()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头，无法获取分辨率")
        self.center_x = frame.shape[1] // 2
        self.center_y = frame.shape[0] // 2
        self.deadzone = 200
        self.kp_wrist = 1.2

        self.scan_state = "RIGHT"
        self.scan_speed = 200
        self.running = False
        self.is_tracking = False # 标记当前是否正处于“锁定追踪”状态

    def run(self, target_class: str):
        self.running = True
        print(f"🚀 开始任务：正在寻找 {target_class}...")

        try:
            while self.running:
                img = self._capture()
                target_uv = self.detect_object(img, target_class)
 
                if target_uv is None:
                    is_safe, danger = self.arm_manager.safe_detect(1, self.arm)
                    #print(f"1号电机位置: {pos1}")
                    if self.scan_state == "RIGHT":
                        self.arm.spin(1, self.scan_speed)
                        if danger == "right":
                            self.arm.brake(1)
                            self.scan_state = "LEFT"
                    elif self.scan_state == "LEFT":
                        self.arm.spin(1, -self.scan_speed)
                        if danger == "left":
                            self.arm.brake(1)
                            self.scan_state = "RIGHT"
                    self._handle_searching()
                else:
                    self.scan_state = "RIGHT"
                    self.arm.brake(1)
                    self._handle_tracking(target_uv)

                cv2.imshow("Tracking System", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        finally:
            self.arm.brake_all()
            self.cap.release()
            cv2.destroyAllWindows()

    def track(self, target_class: str):
        self.running = True
        self.run(target_class)

    def stop(self):
        self.running = False

    def _handle_searching(self):
        self.arm.brake(4)
        self.arm.brake(5)

    def _calculate_speed(self, current_val, target_val, kp):
        error = current_val - target_val
        if abs(error) <= self.deadzone:
            return 0
        return int(error * kp)

    def _handle_tracking(self, target_class: str): # 删除了冗余的 first_target_uv
        print(f"🎯 锁定目标：{target_class}，开启眼颈协同模式")
        lost_frames = 0
        max_lost_allow = 30
        last_v4, last_v5, last_v1 = 0, 0, 0 
        kp_base = 0.05 
        while True:
            img = self._capture()
            current_uv = self.detect_object(img, target_class)
            
            if current_uv is None:
                lost_frames += 1
                if lost_frames > max_lost_allow:
                    print("⚠️ 目标丢失过久，退出追踪模式，恢复扫视")
                    self.arm.brake_all()
                    return 
                v4, v5, v1 = last_v4, last_v5, last_v1
            else:
                lost_frames = 0
                cx, cy = current_uv
                v5 = self._calculate_speed(cx, self.center_x, self.kp_wrist)
                v4 = self._calculate_speed(cy, self.center_y, self.kp_wrist)
                raw_5 = self.arm.get_position(5)
                if raw_5 != -1:
                    center_5 = self.arm_manager.axes[5].c
                    wrist_deviation = self.arm_manager._get_dist(center_5, raw_5)
                    if abs(wrist_deviation) > 50:
                        v1 = int(wrist_deviation * kp_base)
                    else:
                        v1 = 0
                else:
                    v1 = 0
            safe4, danger4 = self.arm_manager.safe_detect(4, self.arm)
            if not safe4 and ((danger4 == "right" and v4 > 0) or (danger4 == "left" and v4 < 0)):
                v4 = 0

            safe5, danger5 = self.arm_manager.safe_detect(5, self.arm)
            if not safe5 and ((danger5 == "right" and v5 > 0) or (danger5 == "left" and v5 < 0)):
                v5 = 0
                
            safe1, danger1 = self.arm_manager.safe_detect(1, self.arm)
            if not safe1 and ((danger1 == "right" and v1 > 0) or (danger1 == "left" and v1 < 0)):
                v1 = 0
            if current_uv:
                print(f"指令发速 -> v1(底座):{v1:4} | v4(上下):{v4:4} | v5(左右):{v5:4} | 手腕偏移:{wrist_deviation if 'wrist_deviation' in locals() else 0}")

            self.arm.spin(4, v4)
            self.arm.spin(5, v5)
            self.arm.spin(1, v1)
            
            last_v4, last_v5, last_v1 = v4, v5, v1

            cv2.imshow("Tracking System", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def _capture(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return frame

    def detect_object(self, img: np.ndarray, target_class: str) -> tuple[float, float] | None:
        results = self.model(img, verbose=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.8:
                    continue
                cls_name = self.model.names[int(box.cls[0])]
                print(cls_name)
                if cls_name == target_class:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(img, cls_name, (int(xyxy[0]), int(xyxy[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return (float(cx), float(cy))
        return None


if __name__ == "__main__":
    tracker = VisualTracking()
    time.sleep(2)
    tracker.track(target_class="person")
