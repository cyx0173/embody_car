from __future__ import annotations

import cv2
import numpy as np

CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_WORKING_SIZE = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT) // 2  # 540

HAND_YOLO_ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE
BASE_YOLO_ROTATE_CODE = None
HAND_CAMERA_ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE


def center_crop_resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    cropped = frame[y0 : y0 + crop, x0 : x0 + crop]
    if cropped.shape[:2][::-1] == (CAMERA_WORKING_SIZE, CAMERA_WORKING_SIZE):
        return cropped
    return cv2.resize(cropped, (CAMERA_WORKING_SIZE, CAMERA_WORKING_SIZE), interpolation=cv2.INTER_AREA)


class CameraManager:
    def __init__(
        self,
        hand_id: int = 1,
        base_id: int = 0,
        *,
        rotate_hand: bool = True,
    ):
        self.hand_id = hand_id
        self.base_id = base_id
        self.rotate_hand = rotate_hand

        self.hand: cv2.VideoCapture | None = None
        self.base: cv2.VideoCapture | None = None

        self._open_hand()
        self._open_base()

    def _open_hand(self) -> None:
        self.hand = cv2.VideoCapture(self.hand_id)
        if not self.hand.isOpened():
            raise RuntimeError(f"手部相机 (id={self.hand_id}) 打开失败")
        self.hand.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_NATIVE_WIDTH))
        self.hand.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_NATIVE_HEIGHT))

    def _open_base(self) -> None:
        self.base = cv2.VideoCapture(self.base_id)
        if not self.base.isOpened():
            raise RuntimeError(f"底座相机 (id={self.base_id}) 打开失败")
        self.base.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_NATIVE_WIDTH))
        self.base.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_NATIVE_HEIGHT))

    def read_hand_raw(self) -> np.ndarray:
        if self.hand is None:
            raise RuntimeError("手部相机未初始化")
        for attempt in range(3):
            ret, frame = self.hand.read()
            if ret and frame is not None:
                if self.rotate_hand and HAND_CAMERA_ROTATE_CODE is not None:
                    frame = cv2.rotate(frame, HAND_CAMERA_ROTATE_CODE)
                return frame
        raise RuntimeError("手部相机连续读取失败")

    def read_base_raw(self) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("底座相机未初始化")
        for attempt in range(3):
            ret, frame = self.base.read()
            if ret and frame is not None:
                return frame
        raise RuntimeError("底座相机连续读取失败")

    def read_hand_calibrated(self) -> np.ndarray:
        return center_crop_resize(self.read_hand_raw())

    def read_base_calibrated(self) -> np.ndarray:
        return center_crop_resize(self.read_base_raw())

    def read_both_calibrated(self) -> tuple[np.ndarray, np.ndarray]:
        return self.read_hand_calibrated(), self.read_base_calibrated()

    def release(self) -> None:
        if self.hand is not None:
            self.hand.release()
            self.hand = None
        if self.base is not None:
            self.base.release()
            self.base = None


if __name__ == "__main__":
    cam = CameraManager()
    h, b = cam.read_both_calibrated()
    cam.release()
