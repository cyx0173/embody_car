from tkinter import Frame
import cv2
import numpy as np
import time
from ultralytics import YOLO
from arm_control import ServoController
from Angle_config import ArmManager

class VisualTracking:
    def __init__(self, model_path: str = "yolo11n.pt", camera_id: int = 0, base_camera_id: int = 1) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.base_cap = cv2.VideoCapture(base_camera_id)
        self.model = YOLO(model_path)
        self.arm = ServoController()
        self.arm_manager = ArmManager()
        self.arm.reset()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头，无法获取分辨率")
        self.center_x = frame.shape[1] // 2
        self.center_y = frame.shape[0] // 2
        self.deadzone = 60
        self.kp_wrist = 0.2
        self.max_lost_allow = 30

        self.scan_state = "RIGHT"
        self.scan_speed = 200
        self.running = False
        self.is_tracking = False # 标记当前是否正处于"锁定追踪"状态
        self.arm4_tracking_orignal_angle =1285
        self.arm5_tracking_orignal_angle = 69

    def run(self, target_class: str):
        self.running = True
        print(f"🚀 开始任务：正在寻找 {target_class}...")

        try:
            while self.running:
                base_img = self._capture_base()
                base_uv = self.detect_object(base_img, target_class)
 
                if base_uv is None:
                    self._search_move()
                else:
                    #这里可以补个函数 更好的定位
        
                    self.scan_state = "RIGHT"
                    self.arm.brake(1)
                    self._handle_tracking(target_class)

        finally:
            self.arm.brake_all()
            self.cap.release()
            cv2.destroyAllWindows()

    def track(self, target_class: str):
        self.running = True
        self.run(target_class)

    def _search_move(self):
        self.arm.brake(4)
        self.arm.brake(5)

        is_safe, danger = self.arm_manager.safe_detect(1, self.arm)
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

    def _handle_tracking(self, target_class: str): 
        print(f"🎯 锁定目标开启精细移动{target_class}")
        lost_frames = 0
        self.arm.move_to(4, self.arm4_tracking_orignal_angle)
        self.arm.move_to(5, self.arm5_tracking_orignal_angle)
        while True:
            img = self._capture()
            current_uv = self.detect_object(img, target_class)
            if current_uv is None:
                lost_frames += 1
                if lost_frames > self.max_lost_allow:
                    print("⚠️ 目标丢失过久，退出追踪模式，恢复扫视")
                    self.arm.brake_all()
                    return 
            else:
                cx, cy = current_uv
                err_x = cx - self.center_x
                err_y = cy - self.center_y
                if abs(err_x) > self.deadzone:  v5 = int(err_x * self.kp_wrist)  
                else: v5 = 0
                if abs(err_y) > self.deadzone:  v4 = int(err_y * self.kp_wrist)  
                else: v4 = 0

                max_tracking_speed = 300 
                v5 = np.clip(v5, -max_tracking_speed, max_tracking_speed)
                v4 = np.clip(v4, -max_tracking_speed, max_tracking_speed)
                safe4, danger4 = self.arm_manager.safe_detect(4, self.arm)
                if not safe4 and ((danger4 == "right" and v4 > 0) or (danger4 == "left" and v4 < 0)):
                    v4 = 0

                safe5, danger5 = self.arm_manager.safe_detect(5, self.arm)
                if not safe5 and ((danger5 == "right" and v5 > 0) or (danger5 == "left" and v5 < 0)):
                    v5 = 0

                self.arm.spin(4, v4)
                self.arm.spin(5, v5)

            cv2.imshow("Tracking System", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break


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
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
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
