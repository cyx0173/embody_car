from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "yolo11s.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview YOLO detections from a Mac/OpenCV camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO .pt model path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview like FaceTime. Detection runs on the mirrored image.",
    )
    return parser.parse_args()


def open_camera(camera_id: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {camera_id}。如果这是第一次运行，请检查 macOS 的相机权限。"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def draw_status(frame, fps: float, detection_count: int) -> None:
    text = f"FPS {fps:4.1f} | detections {detection_count} | q/ESC quit"
    cv2.rectangle(frame, (12, 12), (430, 48), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (22, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    window_name = f"YOLO camera {args.camera}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("YOLO 摄像头预览已启动。按 q 或 ESC 退出。")
    print(f"camera={args.camera}, model={args.model}, conf={args.conf}, imgsz={args.imgsz}")

    last_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("摄像头读取失败，正在重试...")
                time.sleep(0.1)
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            annotated = result.plot()

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + instant_fps * 0.1

            detection_count = 0 if result.boxes is None else len(result.boxes)
            draw_status(annotated, smoothed_fps, detection_count)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
