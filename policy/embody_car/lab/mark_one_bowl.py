#!/usr/bin/env python3
"""YOLO segmentation marker wrapper.

Core usage:

    import cv2
    from yolo_seg_marker import YOLOSegMarker

    yoloseg = YOLOSegMarker(model="yolo11n-seg.pt", device="cuda")
    img = cv2.imread("orange.jpg")
    mark_img = yoloseg.infer(img, "orange")
    cv2.imwrite("orange_mark.jpg", mark_img)

The input/output image format is OpenCV BGR numpy array.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing ultralytics. Install it with:\n"
        "  pip install ultralytics opencv-python\n"
    ) from exc


class YOLOSegMarker:
    """Detect one target object with YOLO segmentation and draw a colored marker.

    Parameters
    ----------
    model:
        YOLO segmentation model path, e.g. "yolo11n-seg.pt" or custom "best.pt".
    device:
        "cuda", "cuda:0", or "cpu".
    conf:
        Detection confidence threshold.
    iou:
        NMS IoU threshold.
    alpha:
        Transparent blue fill alpha for masks. Keep it low to preserve texture.
    thickness:
        Blue bbox/contour thickness.
        mode:
        "bbox", "mask", or "both".
    color_bgr:
        OpenCV BGR marker color. Defaults to blue to preserve the original
        marked-bowl dataset style.
    choose:
        How to choose if multiple target objects are found:
        "largest", "highest_conf", "nearest_center", "leftmost", or "rightmost".
    """

    def __init__(
        self,
        model: str | Path = "yolo11n-seg.pt",
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.7,
        alpha: float = 0.20,
        thickness: int = 6,
        mode: str = "both",
        choose: str = "largest",
        color_bgr: tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        if mode not in {"bbox", "mask", "both"}:
            raise ValueError(f"mode must be bbox/mask/both, got {mode}")
        if choose not in {"largest", "highest_conf", "nearest_center", "leftmost", "rightmost"}:
            raise ValueError(
                "choose must be largest/highest_conf/nearest_center/leftmost/rightmost, "
                f"got {choose}"
            )

        self.model_path = str(model)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.alpha = alpha
        self.thickness = thickness
        self.mode = mode
        self.choose = choose
        self.color_bgr = tuple(int(v) for v in color_bgr)
        self.model = YOLO(self.model_path)

    def infer(
        self,
        img: np.ndarray,
        object_name: str = "orange",
        return_meta: bool = False,
        conf: float | None = None,
        iou: float | None = None,
        mode: str | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Detect and mark target object in one image.

        Parameters
        ----------
        img:
            OpenCV BGR image, shape [H, W, 3].
        object_name:
            Target class name, e.g. "orange". Use "any" to mark the selected
            best detection regardless of class.
        return_meta:
            If True, return (marked_img, meta). Otherwise only return marked_img.
        conf, iou, mode:
            Optional per-call override.
        """
        if img is None:
            raise ValueError("img is None")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape [H,W,3], got {img.shape}")

        use_conf = self.conf if conf is None else conf
        use_iou = self.iou if iou is None else iou
        use_mode = self.mode if mode is None else mode

        result = self.model.predict(
            img,
            conf=use_conf,
            iou=use_iou,
            device=self.device,
            verbose=False,
        )[0]

        _idx, mask, bbox, score, class_name = self._choose_detection(
            result=result,
            target=object_name,
            choose=self.choose,
        )
        marked = self._draw_blue_marker(img, mask, bbox, mode=use_mode)

        meta = {
            "found": bbox is not None,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "confidence": score,
            "class_name": class_name,
        }
        if return_meta:
            return marked, meta
        return marked

    def draw_bbox(
        self,
        img: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int] | list[int] | None,
    ) -> np.ndarray:
        """Draw the same marker style from an already known bbox."""
        bbox = tuple(int(x) for x in bbox_xyxy) if bbox_xyxy is not None else None
        return self._draw_blue_marker(img, mask=None, bbox=bbox, mode="bbox")

    def _choose_detection(
        self,
        result: Any,
        target: str,
        choose: str,
    ) -> tuple[int | None, np.ndarray | None, tuple[int, int, int, int] | None, float | None, str | None]:
        """Return selected detection index, mask, bbox, conf, class_name."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, None, None, None, None

        names = result.names
        img_h, img_w = result.orig_shape[:2]
        candidates = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if target != "any" and class_name != target:
                continue

            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist_center = ((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2) ** 0.5
            candidates.append((i, area, score, dist_center, cx, (x1, y1, x2, y2), class_name))

        if not candidates:
            return None, None, None, None, None

        if choose == "largest":
            selected = max(candidates, key=lambda x: x[1])
        elif choose == "highest_conf":
            selected = max(candidates, key=lambda x: x[2])
        elif choose == "nearest_center":
            selected = min(candidates, key=lambda x: x[3])
        elif choose == "leftmost":
            selected = min(candidates, key=lambda x: x[4])
        elif choose == "rightmost":
            selected = max(candidates, key=lambda x: x[4])
        else:
            raise ValueError(f"Unsupported choose mode: {choose}")

        det_i, _area, score, _dist, _cx, bbox, class_name = selected

        mask = None
        if result.masks is not None and result.masks.data is not None:
            mask_t = result.masks.data[det_i].detach().cpu().numpy()
            mask = cv2.resize(mask_t.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5

        return det_i, mask, bbox, score, class_name

    def _draw_blue_marker(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        bbox: tuple[int, int, int, int] | None,
        mode: str,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        color = self.color_bgr

        if mask is not None and mode in ("mask", "both"):
            overlay = out.copy()
            overlay[mask] = color
            out = cv2.addWeighted(overlay, self.alpha, out, 1.0 - self.alpha, 0)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, max(1, self.thickness // 2))

        if bbox is not None and mode in ("bbox", "both"):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.thickness)

        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test YOLOSegMarker on one image.")
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="orange")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--thickness", type=int, default=6)
    p.add_argument("--mode", choices=["mask", "bbox", "both"], default="both")
    p.add_argument(
        "--choose",
        choices=["largest", "highest_conf", "nearest_center", "leftmost", "rightmost"],
        default="largest",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.source}")

    yoloseg = YOLOSegMarker(
        model=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        alpha=args.alpha,
        thickness=args.thickness,
        mode=args.mode,
        choose=args.choose,
    )

    mark_img, meta = yoloseg.infer(img, args.target, return_meta=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mark_img)
    print(f"saved: {out_path}")
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
