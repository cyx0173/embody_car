"""
Stereo Extrinsic Calibration — 双目相对位置标定

读取 Step 1 的两个 JSON 内参文件，读取 Step 2 拍下的照片对。
运行 cv2.stereoCalibrate (开启 CALIB_FIX_INTRINSIC)，计算并保存：
旋转矩阵 R、平移向量 t（含基线距离）、投影矩阵 P1, P2。

用法:
python calibrate_stereo.py --left-intrinsics camera_left.json --right-intrinsics camera_right.json --stereo-pairs ./stereo_captures --cols 9 --rows 6 --square-size 0.0254
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_intrinsics import center_crop_and_resize_frame

CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_WORKING_WIDTH = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT) // 2
CAMERA_WORKING_HEIGHT = CAMERA_WORKING_WIDTH


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    image_size = (int(data["image_width"]), int(data["image_height"]))
    return camera_matrix, dist_coeffs, image_size


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def discover_stereo_pairs(stereo_dir: Path) -> list[tuple[Path, Path]]:
    left_files = sorted(stereo_dir.glob("left_*.png"))
    pairs = []
    for left_path in left_files:
        stem = left_path.stem.replace("left_", "")
        right_path = left_path.parent / f"right_{stem}.png"
        if right_path.exists():
            pairs.append((left_path, right_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo extrinsic calibration using captured image pairs.")
    parser.add_argument("--left-intrinsics", type=str, required=True, help="Path to left camera intrinsics JSON.")
    parser.add_argument("--right-intrinsics", type=str, required=True, help="Path to right camera intrinsics JSON.")
    parser.add_argument(
        "--stereo-pairs",
        type=str,
        default="stereo_captures",
        help="Directory containing left_*.png / right_*.png pairs.",
    )
    parser.add_argument("--cols", type=int, default=9, help="Chessboard inner corners along width.")
    parser.add_argument("--rows", type=int, default=6, help="Chessboard inner corners along height.")
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square size in meters.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stereo_extrinsics.json",
        help="Output JSON file for stereo extrinsic calibration result.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=5,
        help="Minimum accepted stereo pairs.",
    )
    args = parser.parse_args()

    left_json = Path(args.left_intrinsics)
    right_json = Path(args.right_intrinsics)
    if not left_json.exists():
        raise FileNotFoundError(f"Left intrinsics file not found: {left_json}")
    if not right_json.exists():
        raise FileNotFoundError(f"Right intrinsics file not found: {right_json}")

    mtx_l, dist_l, size_l = load_intrinsics(left_json)
    mtx_r, dist_r, size_r = load_intrinsics(right_json)
    if size_l != size_r:
        raise ValueError(
            f"Image size mismatch: left={size_l} vs right={size_r}. "
            "Both intrinsics must use the same resolution."
        )
    image_size = size_l

    objp_template = build_object_points(args.cols, args.rows, args.square_size)
    pattern_size = (args.cols, args.rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    stereo_dir = Path(args.stereo_pairs)
    pairs = discover_stereo_pairs(stereo_dir)
    if len(pairs) < args.min_pairs:
        raise RuntimeError(
            f"Found {len(pairs)} stereo pairs, need at least {args.min_pairs}."
        )
    print(f"Discovered {len(pairs)} stereo pairs in {stereo_dir}.")

    object_points: list[np.ndarray] = []
    image_points_l: list[np.ndarray] = []
    image_points_r: list[np.ndarray] = []

    for i, (left_path, right_path) in enumerate(pairs):
        frame_l_raw = cv2.imread(str(left_path))
        frame_r_raw = cv2.imread(str(right_path))
        if frame_l_raw is None or frame_r_raw is None:
            print(f"Skipping unreadable pair: {left_path}")
            continue

        frame_l = center_crop_and_resize_frame(
            frame_l_raw, (CAMERA_WORKING_WIDTH, CAMERA_WORKING_HEIGHT)
        )
        frame_r = center_crop_and_resize_frame(
            frame_r_raw, (CAMERA_WORKING_WIDTH, CAMERA_WORKING_HEIGHT)
        )

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(
            gray_l, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        found_r, corners_r = cv2.findChessboardCorners(
            gray_r, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if not (found_l and found_r):
            print(f"Skipping pair {i+1}: chessboard not found in both images.")
            continue

        refined_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        refined_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        object_points.append(objp_template.copy())
        image_points_l.append(refined_l)
        image_points_r.append(refined_r)
        print(f"Pair {len(image_points_l)}: OK  ({left_path.name} / {right_path.name})")

    if len(object_points) < args.min_pairs:
        raise RuntimeError(
            f"Only {len(object_points)} valid pairs after filtering, need at least {args.min_pairs}."
        )

    print(f"\nRunning cv2.stereoCalibrate on {len(object_points)} pairs ...")
    result = cv2.stereoCalibrate(
        objectPoints=object_points,
        imagePoints1=image_points_l,
        imagePoints2=image_points_r,
        cameraMatrix1=mtx_l,
        distCoeffs1=dist_l,
        cameraMatrix2=mtx_r,
        distCoeffs2=dist_r,
        imageSize=image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    (
        rms,
        _,
        _,
        _,
        _,
        R,
        t,
        E,
        F,
    ) = result

    baseline_m = np.linalg.norm(t)
    print(f"RMS reprojection error: {rms:.6f}")
    if rms > 1.0:
        print("WARNING: RMS > 1.0 — possible causes:")
        print("  1. Chessboard was moving during Step 3 capture")
        print("  2. Auto-exposure was enabled (disable it before capturing)")
        print("  3. Reflections or shadows on the board")
        print("  4. Re-run Step 2 with the chessboard held still and auto-exposure off")
    print(f"Baseline (||t||): {baseline_m:.6f} m")
    print(f"R (rotation left->right):\n{R}")
    print(f"t (translation left->right):\n{t}")

    P1 = mtx_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = mtx_r @ np.hstack([R, t.reshape(3, 1)])

    payload = {
        "left_intrinsics": str(left_json),
        "right_intrinsics": str(right_json),
        "stereo_pairs_dir": str(stereo_dir),
        "valid_pair_count": len(object_points),
        "rms_reprojection_error": float(rms),
        "baseline_m": float(baseline_m),
        "R_left_to_right": R.tolist(),
        "t_left_to_right": t.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "P1_rectification": P1.tolist(),
        "P2_rectification": P2.tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "board": {
            "type": "chessboard",
            "inner_corners_cols": args.cols,
            "inner_corners_rows": args.rows,
            "square_size_m": args.square_size,
        },
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved stereo extrinsics to {output_path}")


if __name__ == "__main__":
    main()
