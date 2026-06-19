#python calibrate_intrinsics.py --square-size 0.0254
#一定要修改下真实尺寸（棋盘格方块边长，单位米）

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_SQUARE_CROP_SIZE = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT)
CAMERA_WORKING_WIDTH = CAMERA_SQUARE_CROP_SIZE // 2
CAMERA_WORKING_HEIGHT = CAMERA_SQUARE_CROP_SIZE // 2


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def center_crop_and_resize_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    cropped = frame[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if (cropped.shape[1], cropped.shape[0]) == (target_w, target_h):
        return cropped
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def save_calibration(
    output_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    cols: int,
    rows: int,
    square_size: float,
    capture_count: int,
) -> None:
    payload = {
        "model": "opencv_pinhole",
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms_reprojection_error": float(rms),
        "board": {
            "type": "chessboard",
            "inner_corners_cols": int(cols),
            "inner_corners_rows": int(rows),
            "square_size_m": float(square_size),
        },
        "capture_count": int(capture_count),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive camera intrinsics calibration with a chessboard.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--raw-width", type=int, default=CAMERA_NATIVE_WIDTH, help="Requested camera capture width before downsampling.")
    parser.add_argument("--raw-height", type=int, default=CAMERA_NATIVE_HEIGHT, help="Requested camera capture height before downsampling.")
    parser.add_argument("--width", type=int, default=CAMERA_WORKING_WIDTH, help="Working image width after downsampling.")
    parser.add_argument("--height", type=int, default=CAMERA_WORKING_HEIGHT, help="Working image height after downsampling.")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners along the board width.")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners along the board height.")
    parser.add_argument("--square-size", type=float, required=True, help="Chessboard square size in meters.")
    parser.add_argument(
        "--output",
        default="camera_intrinsics.json",
        help="Output JSON file for the calibration result.",
    )
    parser.add_argument(
        "--min-captures",
        type=int,
        default=15,
        help="Minimum accepted views recommended before calibration.",
    )
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)
    object_points_template = build_object_points(args.cols, args.rows, args.square_size)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.raw_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.raw_height))
    reported_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) or int(args.raw_width)
    reported_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) or int(args.raw_height)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    print("Controls:")
    print("  space  capture current frame if chessboard is detected")
    print("  enter  run calibration and save file")
    print("  q      quit without saving")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            continue
        frame = center_crop_and_resize_frame(frame, (int(args.width), int(args.height)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        display = frame.copy()
        status = f"captures={len(image_points)}"
        color = (0, 0, 255)

        refined_corners = None
        if found:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, refined_corners, found)
            status = f"captures={len(image_points)}  board=detected"
            color = (0, 200, 0)
        else:
            status = f"captures={len(image_points)}  board=not detected"

        cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(
            display,
            f"board={args.cols}x{args.rows}  square={args.square_size:.4f} m",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"cam={args.raw_width}x{args.raw_height}  reported={reported_width}x{reported_height}  crop={CAMERA_SQUARE_CROP_SIZE}  work={args.width}x{args.height}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Intrinsics Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            if refined_corners is None:
                print("Chessboard not detected, capture skipped.")
                continue
            object_points.append(object_points_template.copy())
            image_points.append(refined_corners)
            print(f"Captured view {len(image_points)}.")
            continue
        if key in (13, 10):
            if len(image_points) < max(3, args.min_captures):
                print(f"Need at least {max(3, args.min_captures)} captures, currently {len(image_points)}.")
                continue
            if image_size is None:
                print("No image size available.")
                continue

            rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None,
            )

            output_path = Path(args.output).resolve()
            save_calibration(
                output_path=output_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_size=image_size,
                rms=rms,
                cols=args.cols,
                rows=args.rows,
                square_size=args.square_size,
                capture_count=len(image_points),
            )

            print(f"Saved calibration to {output_path}")
            print(f"RMS reprojection error: {rms:.6f}")
            print("Camera matrix:")
            print(camera_matrix)
            print("Distortion coefficients:")
            print(dist_coeffs.reshape(-1))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# python .\calibrate_intrinsics.py --camera 1 --square-size 0.01688 