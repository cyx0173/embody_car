import argparse
import time
from pathlib import Path

import cv2
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


def target_candidates(target: str | None) -> set[str]:
    if not target:
        return set()
    target = target.strip()
    return set(TARGET_ALIASES.get(target, (target,)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO detections from one camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", default="bottle")
    parser.add_argument("--model", default=str(BASE_DIR / "yolo11s.pt"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    candidates = target_candidates(args.target)
    print(
        f"YOLO camera check: camera={args.camera}, target={args.target}, "
        f"candidates={sorted(candidates)}, seconds={args.seconds}"
    )

    deadline = time.monotonic() + args.seconds
    last_log = 0.0
    found_count = 0
    try:
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("摄像头读取失败")
                time.sleep(0.1)
                continue

            results = model(frame, verbose=False)
            visible: list[str] = []
            found: list[str] = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    cls_name = model.names[int(box.cls[0])]
                    visible.append(f"{cls_name}:{conf:.2f}")
                    if cls_name in candidates:
                        found.append(f"{cls_name}:{conf:.2f}")

            if found:
                found_count += 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                last_log = now
                visible_text = ", ".join(visible[:8]) if visible else "none"
                found_text = ", ".join(found) if found else "none"
                print(f"visible={visible_text}; target_found={found_text}")

            if args.show:
                annotated = results[0].plot()
                cv2.imshow("YOLO camera check", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"target_found_frames={found_count}")


if __name__ == "__main__":
    main()
