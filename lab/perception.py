from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", "/private/tmp/Ultralytics")

from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class Camera:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._cap: cv2.VideoCapture | None = None

    def capture(self) -> np.ndarray:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"无法打开摄像头: {self.camera_id}。如果是macOS，请到 "
                    "System Settings -> Privacy & Security -> Camera "
                    "给 Terminal/iTerm/当前终端授权；授权后重启终端再运行。"
                )
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("摄像头读取失败，请检查摄像头是否被其他程序占用。")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class YoloPerception:
    def __init__(self, model_path: str | Path = "yolo11n.pt", confidence: float = 0.3):
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, image: np.ndarray, target: str | None = None) -> list[Detection]:
        results = self.model(image, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue
                label = self.model.names[int(box.cls[0])]
                if target and label != target:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append(Detection(label=label, confidence=conf, bbox_xyxy=(x1, y1, x2, y2)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        label = f"{det.label} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def show_image(title: str, image: np.ndarray, wait_ms: int = 1) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(wait_ms)


def close_windows() -> None:
    cv2.destroyAllWindows()


def summarize_detections(detections: list[Detection], limit: int = 8) -> str:
    if not detections:
        return "YOLO没有检测到明确物体。"
    parts = []
    for det in detections[:limit]:
        cx, cy = det.center
        parts.append(f"{det.label}(conf={det.confidence:.2f}, center=({cx:.0f},{cy:.0f}))")
    return "YOLO检测结果: " + "; ".join(parts)
