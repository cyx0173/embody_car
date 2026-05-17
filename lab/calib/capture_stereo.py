"""
Stereo Image Capture — 双目同步采集

同时打开 cap(0) 和 cap(1)，实时展示拼接画面。
当检测到棋盘格且按下 s 键时，同步保存左右帧用于后续外参计算。

用法:
    python capture_stereo.py --cols 9 --rows 6 --output-dir ./stereo_captures
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
from arm_control import ServoController
import cv2
import numpy as np

from calibrate_intrinsics import center_crop_and_resize_frame

CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_SQUARE_CROP_SIZE = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT)
CAMERA_WORKING_WIDTH = CAMERA_SQUARE_CROP_SIZE // 2
CAMERA_WORKING_HEIGHT = CAMERA_SQUARE_CROP_SIZE // 2


def main() -> None:
    arm = ServoController()
    print("Resetting arm to home position...")
    arm.reset()

    parser = argparse.ArgumentParser(description="Stereo image capture for extrinsic calibration.")
    parser.add_argument("--cap0", type=int, default=1, help="Camera index for left (base) camera.")
    parser.add_argument("--cap1", type=int, default=0, help="Camera index for right (aux) camera.")
    parser.add_argument("--cols", type=int, default=9, help="Chessboard inner corners along width.")
    parser.add_argument("--rows", type=int, default=6, help="Chessboard inner corners along height.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="stereo_captures",
        help="Directory to save captured stereo image pairs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap0 = cv2.VideoCapture(args.cap0)
    cap1 = cv2.VideoCapture(args.cap1)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cap0}.")
    if not cap1.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cap1}.")

    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_NATIVE_WIDTH))
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_NATIVE_HEIGHT))
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_NATIVE_WIDTH))
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_NATIVE_HEIGHT))

    pair_count = 0
    pattern_size = (args.cols, args.rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    print("Controls:")
    print("  s  capture stereo pair if chessboard detected in BOTH cameras")
    print("  q  quit")
    print("NOTE: Saved images are raw captures (no overlays).")

    while True:
        cap0.grab()
        cap1.grab()
        ok0, frame0_raw = cap0.retrieve()
        ok1, frame1_raw = cap1.retrieve()
        if not (ok0 and ok1):
            print("Warning: failed to retrieve from one or both cameras.")
            continue

        frame0 = center_crop_and_resize_frame(
            frame0_raw, (CAMERA_WORKING_WIDTH, CAMERA_WORKING_HEIGHT)
        )
        frame1 = center_crop_and_resize_frame(
            frame1_raw, (CAMERA_WORKING_WIDTH, CAMERA_WORKING_HEIGHT)
        )

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        found0, corners0 = cv2.findChessboardCorners(
            gray0, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        found1, corners1 = cv2.findChessboardCorners(
            gray1, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        refined0 = None
        if found0:
            refined0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame0, pattern_size, refined0, found0)

        refined1 = None
        if found1:
            refined1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame1, pattern_size, refined1, found1)

        both = found0 and found1
        status_color = (0, 200, 0) if both else (0, 0, 255)
        status = (
            f"pairs={pair_count}  left={'OK' if found0 else 'NO board'}  "
            f"right={'OK' if found1 else 'NO board'}"
        )

        both_label = "BOTH DETECTED — press s to save" if both else "show board in both cams"
        vis0 = frame0.copy()
        vis1 = frame1.copy()
        cv2.putText(vis0, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(vis0, both_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        cv2.putText(vis1, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(
            vis1,
            "L" if found0 else " ",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 0) if found0 else (100, 100, 100),
            2,
            cv2.LINE_AA,
        )

        stereo = np.hstack([vis0, vis1])
        cv2.imshow("Stereo Capture  [Left | Right]", stereo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if not both:
                print("Chessboard not detected in both cameras, capture skipped.")
                continue
            pair_count += 1
            left_path = output_dir / f"left_{pair_count:03d}.png"
            right_path = output_dir / f"right_{pair_count:03d}.png"
            cv2.imwrite(str(left_path), frame0_raw)
            cv2.imwrite(str(right_path), frame1_raw)
            print(f"Saved pair {pair_count}: {left_path} / {right_path}")

    arm.close()
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
