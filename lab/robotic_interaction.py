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
from solve_ik import solve_ik
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAFE_READY_TICKS = {
    1: 2222,  # manually tuned safe ready pose
    2: 845,
    3: 3115,
    4: 898,
    5: 74,
}
SAFE_FOLD_ORDER = (3, 4)
SAFE_REST_ORDER = (2, 1, 5)
TARGET_MOVE_ORDER = (1, 2, 3, 4, 5)
SAFE_MOVE_SPEED = 800
SAFE_MOVE_ACC = 40
SAFE_MOVE_SERVO_OVERRIDES = {
    4: {"speed": 400, "acc": 20},
}
SAFE_POSITION_TOLERANCE_TICKS = 80
SAFE_FOLD_TIMEOUT_S = 8.0
SAFE_READY_SETTLE_S = 4.0


class RoboticInteraction:
    """机器人交互模块"""

    def __init__(self,
        xacro_path: str =  "interaction/so101_follower.urdf.xacro",
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

    def _move_ticks(
        self,
        ticks: dict[int, int],
        order: tuple[int, ...],
        speed: int = 500,
        acc: int = 20,
        delay_s: float = 0.25,
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
            self.arm.move_to(servo_id, tick, speed=move_speed, acc=move_acc)
            time.sleep(delay_s)

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

    def move_to_world_xyz(self, target_world_xyz: np.ndarray, return_to_ready: bool = True) -> None:
        self.target_world_xyz = target_world_xyz
        self.last_ik_solution = solve_ik(self.target_world_xyz)
        print("目标坐标:", self.target_world_xyz)
        print("目标 tick:", self.last_ik_solution)
        if return_to_ready:
            self._move_to_safe_ready()
        self._move_ticks(
            self.last_ik_solution,
            TARGET_MOVE_ORDER,
            speed=500,
            acc=20,
            delay_s=0.25,
        )

    def test_interact(self,target_world_xyz)->None:
        self.move_to_world_xyz(target_world_xyz, return_to_ready=True)


    def interact(self, target: str):
        self.target_world_xyz = self.double_cap_locate(target)
        print(self.target_world_xyz)
        self.move_to_world_xyz(self.target_world_xyz, return_to_ready=True)

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

    def camera_point_to_base(
        point_camera: np.ndarray,
        mounted_link: str,
        link_T_camera: np.ndarray,
        link_frames: dict[str, np.ndarray],
    ) -> np.ndarray:
        base_T_camera = link_frames[mounted_link] @ link_T_camera
        homogeneous = np.ones(4, dtype=float)
        homogeneous[:3] = point_camera
        return (base_T_camera @ homogeneous)[:3]

    def convert_4d_to_3d(self,points_4d: np.ndarray) -> np.ndarray:
        camera1_point_3d = points_4d[:3] / points_4d[3]
        gripper_point_3d = self.camera_point_to_base(camera1_point_3d, "gripper_link", self.link_T_camera, self.link_frames)
        return gripper_point_3d

    #双目定位代码
    def double_cap_locate(self,target: str, extrinsics_path: str) -> np.ndarray:
        self.locate_point(target)
        with open(extrinsics_path, 'r') as f:
            ext = json.load(f)
        P1 = np.array(ext['P1_rectification'])
        P2 = np.array(ext['P2_rectification'])
        #手部相机p2 底座相机p1 
        pts_1,pts_2 = self.detect_target(target)
        points_4d = cv2.triangulatePoints(P2, P1, pts_2, pts_1)
        #返回相对手部相机的坐标
        point_3d = self.convert_4d_to_3d(points_4d)
        return point_3d
if __name__ == "__main__":
    robot = RoboticInteraction()
    point = np.array([0.30, 0.00, 0.24])
    robot.test_interact(point)
