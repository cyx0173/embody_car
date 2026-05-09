from __future__ import annotations
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from interaction.so101 import RobotIKSolver
import numpy as np
from arm_control import ServoController
import json
import cv2
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class RoboticInteraction:
    """机器人交互模块"""

    def __init__(self,
        xacro_path: str | Path | None = None,
        camera_id: int = 0, base_camera_id: int = 1,
        model_path: str = "yolo11n.pt"
        ):
        self.robot = RobotIKSolver(xacro_path)
        self.arm = ServoController()
        self.cap = cv2.VideoCapture(camera_id)
        self.base_cap = cv2.VideoCapture(base_camera_id)
        self.model = YOLO(model_path)
        self.scan_state = "RIGHT"
        self.state = "LOCATE"
        self.scan_speed = 50

    def interact(self, target: str):
        self.target_world_xyz = double_cap_locate(target)
        self.last_ik_solution, self.last_ik_error = self.robot.solve_ik(self.target_world_xyz)
        self.arm.move_to(self.last_ik_solution)

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

    def detect_target(self,target: str) -> tuple[np.ndarray, np.ndarray]:
        ret, frame_1 = self.cap.read()
        ret, frame_0 = self.base_cap.read()
        pts_1 = self.detect_object(frame_1, target)
        pts_2 = self.detect_object(frame_0, target)
        return pts_1, pts_2

    def locate_point(self,target: str) -> None:
        self.arm.reset()
        self.state = "LOCATE"
        while self.state == "LOCATE":
            base_img = self._capture_base()
            base_uv = self.detect_object(base_img, target)
            if base_uv is None:
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
            else:
                self.state = "TRACK"
                break
            time.sleep(0.1)

    def convert_4d_to_3d(self,points_4d: np.ndarray) -> np.ndarray:
        point_3d = points_4d[:3] / points_4d[3]
        return point_3d

    #双目定位代码
    def double_cap_locate(self,target: str, extrinsics_path: str) -> np.ndarray:
        self.locate_point(target)
        with open(extrinsics_path, 'r') as f:
            ext = json.load(f)
        P1 = np.array(ext['P1_rectification'])
        P2 = np.array(ext['P2_rectification'])
        pts_1,pts_2 = self.detect_target(target)
        points_4d = cv2.triangulatePoints(P1, P2, pts_1, pts_2)
        point_3d = self.convert_4d_to_3d(points_4d)
        return point_3d
