from __future__ import annotations

import threading

import cv2
import numpy as np

from camera import CameraManager
from object_interaction import ObjectInteraction
from object_tracking import ObjectTracking


class RobotRuntime:
    def __init__(self) -> None:
        self.camera = CameraManager()
        self.object_tracking = ObjectTracking(camera=self.camera)
        self.object_interaction = ObjectInteraction(camera=self.camera)
        self.tracking_thread: threading.Thread | None = None

    def capture_scene_image(self) -> np.ndarray:
        frame = self.camera.read_hand_raw()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def start_tracking(self, target: str) -> None:
        self.stop_tracking_if_running()
        self.tracking_thread = threading.Thread(
            target=self.object_tracking.track,
            args=(target,),
            daemon=True,
        )
        self.tracking_thread.start()

    def stop_tracking_if_running(self) -> bool:
        if self.object_tracking is not None and self._tracking_is_running():
            print("检测到正在追踪，先停止追踪")
            self.object_tracking.stop()
            self.tracking_thread.join(timeout=5.0)
            if self.tracking_thread.is_alive():
                print("追踪线程仍未退出，将继续在后台尝试停止")
                return False
            self.tracking_thread = None
            return True
        self.tracking_thread = None
        return False

    def reset_arm(self) -> None:
        self.object_interaction.arm.reset()

    def release_for_pick_place(self) -> None:
        self.stop_tracking_if_running()
        self._close_serial_owner(self.object_tracking)
        self._close_serial_owner(self.object_interaction)
        self.camera.release()

    def restore_after_pick_place(self) -> None:
        self.camera = CameraManager()
        self.object_tracking = ObjectTracking(camera=self.camera)
        self.object_interaction = ObjectInteraction(camera=self.camera)
        self.tracking_thread = None

    def _tracking_is_running(self) -> bool:
        return self.tracking_thread is not None and self.tracking_thread.is_alive()

    def _close_serial_owner(self, owner) -> None:
        arm = getattr(owner, "arm", None)
        serial_obj = getattr(arm, "_ser", None)
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass


def format_nlu_error(parsed: dict) -> str:
    if parsed.get("error") == "missing_target":
        return "我还不知道要操作哪个目标，请说清楚目标物体。"
    return "指令解析不完整，请重新说一遍。"


def target_bowl_from_nlu(parsed: dict) -> str:
    destination = parsed.get("destination")
    destination_qualifier = parsed.get("destination_qualifier")
    target_qualifier = parsed.get("target_qualifier")

    if destination == "bowl" and destination_qualifier in {"left", "right"}:
        return destination_qualifier
    if target_qualifier in {"left", "right"}:
        return target_qualifier
    return "right"
