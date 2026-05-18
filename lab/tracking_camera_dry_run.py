import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

TARGET_ALIASES = {
    "computer": ("laptop", "tv"),
    "screen": ("laptop", "tv"),
    "monitor": ("tv", "laptop"),
    "电脑": ("laptop", "tv"),
    "屏幕": ("laptop", "tv"),
    "手机": ("cell phone",),
    "瓶子": ("bottle",),
    "杯子": ("cup",),
    "人": ("person",),
}


def target_candidates(target: str) -> set[str]:
    return set(TARGET_ALIASES.get(target, (target,)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run wrist tracking from one camera without hardware."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", default="bottle")
    parser.add_argument("--model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--deadzone", type=int, default=60)
    parser.add_argument("--kp", type=float, default=0.45)
    parser.add_argument("--max-speed", type=int, default=650)
    parser.add_argument("--flex-sign", type=int, default=int(os.getenv("TRACK_WRIST_FLEX_SIGN", "1")))
    parser.add_argument("--roll-sign", type=int, default=int(os.getenv("TRACK_WRIST_ROLL_SIGN", "1")))
    parser.add_argument("--rotate-ccw90", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    model = YOLO(args.model)
    candidates = target_candidates(args.target)
    print(
        f"Dry-run tracking: camera={args.camera}, target={args.target}, "
        f"candidates={sorted(candidates)}"
    )
    print(
        "说明: err_x 控制 5 号轴, err_y 控制 4 号轴；这里只打印硬件应执行的 spin。"
        f" flex_sign={args.flex_sign}, roll_sign={args.roll_sign}"
    )

    deadline = time.monotonic() + args.seconds
    last_log = 0.0
    last_cmd: tuple[int, int] | None = None
    try:
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("摄像头读取失败")
                time.sleep(0.1)
                continue
            if args.rotate_ccw90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h, w = frame.shape[:2]
            center_x = w / 2
            center_y = h / 2

            results = model(frame, verbose=False)
            best: tuple[float, np.ndarray, str] | None = None
            visible: list[str] = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    cls_name = model.names[int(box.cls[0])]
                    visible.append(f"{cls_name}:{conf:.2f}")
                    if cls_name in candidates:
                        xyxy = box.xyxy[0].cpu().numpy()
                        if best is None or conf > best[0]:
                            best = (conf, xyxy, cls_name)

            now = time.monotonic()
            if best is None:
                if now - last_log >= 1.0:
                    last_log = now
                    text = ", ".join(visible[:8]) if visible else "none"
                    print(f"未检测到目标; visible={text}; 硬件动作: brake 4, brake 5")
                if last_cmd != (0, 0):
                    last_cmd = (0, 0)
                    print("[DRY RUN] spin servo 4: speed=0; spin servo 5: speed=0")
            else:
                conf, xyxy, cls_name = best
                cx = float((xyxy[0] + xyxy[2]) / 2)
                cy = float((xyxy[1] + xyxy[3]) / 2)
                err_x = cx - center_x
                err_y = cy - center_y
                speed5 = 0 if abs(err_x) <= args.deadzone else int(args.roll_sign * err_x * args.kp)
                speed4 = 0 if abs(err_y) <= args.deadzone else int(args.flex_sign * err_y * args.kp)
                speed5 = int(np.clip(speed5, -args.max_speed, args.max_speed))
                speed4 = int(np.clip(speed4, -args.max_speed, args.max_speed))
                cmd = (speed4, speed5)

                if now - last_log >= 0.3 or cmd != last_cmd:
                    last_log = now
                    last_cmd = cmd
                    print(
                        f"detected={cls_name}:{conf:.2f}, center=({cx:.1f},{cy:.1f}), "
                        f"err=({err_x:.1f},{err_y:.1f}) -> "
                        f"spin servo 4 speed={speed4}, servo 5 speed={speed5}"
                    )

                if args.show:
                    cv2.rectangle(
                        frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255),
                        2,
                    )
                    cv2.circle(frame, (int(center_x), int(center_y)), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)

            if args.show:
                cv2.imshow("tracking dry run", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
