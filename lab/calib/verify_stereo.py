"""
Stereo Calibration Verification — 双目标定验证

用已保存的标定结果对同一张照片对做三角化，
把棋盘格角点反投影回3D空间，检查深度值和尺寸误差。

用法:
    python verify_stereo.py --stereo-extrinsics stereo_extrinsics.json --left-intrinsics camera_left.json --right-intrinsics camera_right.json --stereo-pairs ./stereo_captures --cols 9 --rows 6 --square-size 0.0254
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
import numpy as np

from calibrate_intrinsics import center_crop_and_resize_frame


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    mtx = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return mtx, dist


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify stereo extrinsic calibration via triangulation.")
    parser.add_argument("--stereo-extrinsics", type=str, default="stereo_extrinsics.json",
                        help="Path to stereo_extrinsics.json from calibrate_stereo.py")
    parser.add_argument("--left-intrinsics", type=str, default="camera_left.json")
    parser.add_argument("--right-intrinsics", type=str, default="camera_right.json")
    parser.add_argument("--stereo-pairs", type=str, default="stereo_captures",
                        help="Directory with captured stereo pairs (uses pair 1)")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=0.0254)
    args = parser.parse_args()

    ext = load_json(Path(args.stereo_extrinsics))
    left_data = load_json(Path(args.left_intrinsics))
    mtx_l, dist_l = load_intrinsics(Path(args.left_intrinsics))
    mtx_r, dist_r = load_intrinsics(Path(args.right_intrinsics))
    image_size = (int(left_data["image_width"]), int(left_data["image_height"]))
    working_size = (image_size[0] // 2, image_size[1] // 2)

    R = np.array(ext["R_left_to_right"], dtype=np.float64)
    t = np.array(ext["t_left_to_right"], dtype=np.float64)
    baseline = ext["baseline_m"]

    print("=" * 50)
    print("Stereo Extrinsics Summary")
    print("=" * 50)
    print(f"RMS reprojection error: {ext['rms_reprojection_error']:.4f}")
    print(f"Baseline: {baseline:.4f} m  ({baseline * 1000:.2f} mm)")
    print(f"R (left->right):\n{R}")
    print(f"t (left->right):\n{t.flatten()}")

    pair_dir = Path(args.stereo_pairs)
    left_img_path = sorted(pair_dir.glob("left_*.png"))[0]
    right_img_path = sorted(pair_dir.glob("right_*.png"))[0]
    print(f"\nUsing pair: {left_img_path.name} / {right_img_path.name}")

    frame_l_raw = cv2.imread(str(left_img_path))
    frame_r_raw = cv2.imread(str(right_img_path))

    frame_l = center_crop_and_resize_frame(frame_l_raw, working_size)
    frame_r = center_crop_and_resize_frame(frame_r_raw, working_size)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    grid = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp[:, :2] = grid * args.square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern_size = (args.cols, args.rows)

    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    ok_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    ok_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    if not (ok_l and ok_r):
        print("Chessboard not detected in both images.")
        return

    refined_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    refined_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    img_pts_l = refined_l.squeeze().astype(np.float64)
    img_pts_r = refined_r.squeeze().astype(np.float64)

    R_l = np.eye(3, dtype=np.float64)

    img_pts_l_ud = cv2.undistortPoints(
        img_pts_l.reshape(1, -1, 2), mtx_l, dist_l, R=R_l, P=mtx_l
    ).reshape(-1, 2).T

    img_pts_r_ud = cv2.undistortPoints(
        img_pts_r.reshape(1, -1, 2), mtx_r, dist_r, R=R, P=mtx_r
    ).reshape(-1, 2).T

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        image_size, R, t.reshape(3, 1),
        0,
    )

    points4d_h = cv2.triangulatePoints(P1, P2, img_pts_l_ud, img_pts_r_ud)
    points3d = (points4d_h[:3] / points4d_h[3]).T

    print("\n" + "=" * 50)
    print("Triangulation Results")
    print("=" * 50)
    print(f"Reconstructed {len(points3d)} 3D points (corners of the chessboard).")

    xs = points3d[:, 0]
    ys = points3d[:, 1]
    zs = points3d[:, 2]

    print(f"\nX range: {xs.min():.4f}  to  {xs.max():.4f}  m  (spread: {xs.max()-xs.min():.4f})")
    print(f"Y range: {ys.min():.4f}  to  {ys.max():.4f}  m  (spread: {ys.max()-ys.min():.4f})")
    print(f"Z range: {zs.min():.4f}  to  {zs.max():.4f}  m  (spread: {zs.max()-zs.min():.4f})")
    print(f"Mean Z (depth): {zs.mean():.4f} m")

    expected_span = (args.cols - 1) * args.square_size
    actual_span_x = xs.max() - xs.min()
    print(f"\nExpected board width (col span):  {expected_span:.4f} m")
    print(f"Actual reconstructed width (X):   {actual_span_x:.4f} m")
    print(f"X span error: {abs(actual_span_x - expected_span) / expected_span * 100:.2f}%")

    print("\n" + "=" * 50)
    print("Sanity Checks")
    print("=" * 50)
    checks_passed = 0
    checks_total = 4

    if 0.05 < baseline < 1.0:
        print(f"[OK] Baseline {baseline*1000:.1f}mm is in reasonable range (50-1000mm)")
        checks_passed += 1
    else:
        print(f"[WARN] Baseline {baseline*1000:.1f}mm is outside typical range (50-1000mm)")

    if abs(actual_span_x - expected_span) / expected_span < 0.1:
        print(f"[OK] Board width reconstruction error < 10%")
        checks_passed += 1
    else:
        print(f"[WARN] Board width reconstruction error > 10% (possible calibration issue)")

    if zs.min() > 0:
        print(f"[OK] All Z depths are positive (reasonable)")
        checks_passed += 1
    else:
        print(f"[WARN] Some Z depths are negative (check R/t sign convention)")

    if ext["rms_reprojection_error"] < 1.0:
        print(f"[OK] StereoCalibrate RMS < 1.0 px")
        checks_passed += 1
    else:
        print(f"[WARN] StereoCalibrate RMS = {ext['rms_reprojection_error']:.2f} px (> 1.0)")

    print(f"\nPassed {checks_passed}/{checks_total} checks.")


if __name__ == "__main__":
    main()
