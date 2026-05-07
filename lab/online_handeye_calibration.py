"""
Online Hand-Eye Calibration Module

Core idea:
  1. Robot executes known joint motions
  2. Record end-effector 3D world positions and corresponding image feature pixel coords
  3. Use EPnP/PnP to recover the rigid transform: base_link_T_camera
  4. Write result back to handeye_calibration.json

Usage:
  from lab.online_handeye_calibration import OnlineHandEyeCalibrator
  calib = OnlineHandEyeCalibrator(robot_controller, camera_source=0)
  result = calib.run()
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


class PosePair:
    """One observation: EE world position + image feature pixel coords."""

    def __init__(self, ee_xyz: np.ndarray, feature_uv: np.ndarray):
        self.ee_xyz = ee_xyz  # (3,)
        self.feature_uv = feature_uv  # (N, 2)


class HandEyeResult:
    """Calibration result."""

    def __init__(self, link_T_camera: np.ndarray, trans_error: float, rot_error: float, n_samples: int):
        self.link_T_camera = link_T_camera  # 4x4 homogeneous transform
        self.trans_error = trans_error
        self.rot_error = rot_error
        self.n_samples = n_samples


class OnlineHandEyeCalibrator:
    """
    Online hand-eye calibrator.

    Parameters
    ----------
    robot : RobotController
        Already-connected RobotController instance.
    camera_source : int | str | Path | cv2.VideoCapture
        Camera input (camera index / video path / open VideoCapture).
    handeye_json_path : Path | str | None
        Output path for handeye_calibration.json.
    n_poses : int
        Number of poses to collect (8-15 recommended).
    feature_detector : str
        Feature type: 'orb' | 'akaze' | 'sift'.
    """

    DEFAULT_HANDeye_PATH = (
        Path(__file__).parent.parent / "landmark_demo_v2" / "robot_runtime" / "handeye_calibration.json"
    )

    def __init__(
        self,
        robot,
        camera_source: int | str | Path | None = 0,
        handeye_json_path: Path | str | None = None,
        n_poses: int = 12,
        feature_detector: str = "orb",
    ):
        self.robot = robot
        self.handeye_json_path = Path(handeye_json_path) if handeye_json_path else self.DEFAULT_HANDeye_PATH
        self.n_poses = n_poses
        self.feature_detector = feature_detector

        # Camera intrinsics (estimated defaults; replace with calibrated values in production)
        self.K = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float
        )

        # Open camera
        if camera_source is None:
            self.cap = None
        elif isinstance(camera_source, cv2.VideoCapture):
            self.cap = camera_source
        else:
            self.cap = cv2.VideoCapture(
                int(camera_source) if isinstance(camera_source, int) else str(camera_source)
            )
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {camera_source}")

        self._pairs: list[PosePair] = []

    # ── Feature detection ───────────────────────────────────────

    def _make_detector(self):
        if self.feature_detector == "orb":
            return cv2.ORB_create(nfeatures=500)
        if self.feature_detector == "akaze":
            return cv2.AKAZE_create()
        if self.feature_detector == "sift":
            return cv2.SIFT_create(nfeatures=500)
        raise ValueError(f"Unknown detector: {self.feature_detector}")

    def _extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        detector = self._make_detector()
        kp, desc = detector.detectAndCompute(gray, None)
        pts = np.array([p.pt for p in kp], dtype=np.float32)
        return pts, kp, desc

    def _match_features(self, desc1, desc2):
        if self.feature_detector in ("orb", "akaze"):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        return sorted(matches, key=lambda m: m.distance)

    def _filter_matches(self, matches, kp1, kp2, threshold: float = 30.0):
        if len(matches) < 8:
            return matches
        src = np.float64([kp1[m.queryIdx].pt for m in matches])
        dst = np.float64([kp2[m.trainIdx].pt for m in matches])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=threshold)
        if mask is None:
            return matches
        return [m for m, valid in zip(matches, mask.ravel()) if valid]

    # ── Data collection ─────────────────────────────────────────

    def _capture_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def _get_ee_xyz(self):
        tip_pos, *_ = self.robot._geometry()
        return tip_pos[:3]

    def _move_and_capture(self, delta_joints: dict) -> tuple | None:
        """Execute a micro-joint motion and capture before/after EE 3D + image features."""
        import random

        ee_before = self._get_ee_xyz()
        frame_before = self._capture_frame()
        if frame_before is None:
            return None

        saved_joints = dict(self.robot.current_joint_values)
        for name, delta in delta_joints.items():
            if name in self.robot.current_joint_values:
                self.robot.current_joint_values[name] += delta
        self.robot.execute_pose(self.robot.current_joint_values)
        time.sleep(0.8)

        ee_after = self._get_ee_xyz()
        frame_after = self._capture_frame()
        self.robot.execute_pose(saved_joints)
        time.sleep(0.5)

        if frame_after is None:
            return None

        pts1, kp1, desc1 = self._extract_features(frame_before)
        pts2, kp2, desc2 = self._extract_features(frame_after)
        if len(pts1) < 20 or len(pts2) < 20:
            return None

        matches = self._match_features(desc1, desc2)
        matches = self._filter_matches(matches, kp1, kp2)
        if len(matches) < 8:
            return None

        uv = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        return ee_before, ee_after, uv

    def collect_data(self):
        """Execute n_poses random micro-motions and collect data pairs."""
        import random

        self._pairs.clear()
        movable = [j.name for j in self.robot.movable_joints]
        if not movable:
            raise RuntimeError("No movable joints found")

        n_success = 0
        for i in range(self.n_poses):
            joint_name = random.choice(movable)
            delta = random.uniform(-0.3, 0.3)
            result = self._move_and_capture({joint_name: delta})

            if result is not None:
                ee_before, ee_after, uv = result
                self._pairs.append(PosePair(ee_xyz=ee_after, feature_uv=uv))
                n_success += 1
                dx = ee_after[0] - ee_before[0]
                dy = ee_after[1] - ee_before[1]
                dz = ee_after[2] - ee_before[2]
                print(
                    f"  [{i+1}/{self.n_poses}] OK, {len(uv)} features, "
                    f"d_xyz=({dx:.2f},{dy:.2f},{dz:.2f}) mm"
                )
            else:
                print(f"  [{i+1}/{self.n_poses}] failed, retrying...")

        print(f"\n[collect] {n_success}/{self.n_poses} pairs collected")
        return n_success

    # ── Calibration solver ──────────────────────────────────────

    def calibrate(self):
        """Solve hand-eye transform using EPnP."""
        if len(self._pairs) < 6:
            print(f"[calibrate] Need >= 6 pairs, got {len(self._pairs)}")
            return None

        all_xyz = np.vstack([p.ee_xyz for p in self._pairs])
        all_uv = np.vstack([p.feature_uv for p in self._pairs])

        centroid_xyz = all_xyz.mean(axis=0)
        xyz_centered = all_xyz - centroid_xyz

        dist_coeffs = np.zeros(5, dtype=float)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=xyz_centered.astype(np.float64),
            imagePoints=all_uv.astype(np.float64),
            cameraMatrix=self.K.astype(np.float64),
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            ransacReprojThreshold=5.0,
            iterationsCount=1000,
        )

        if not success:
            print("[calibrate] solvePnP failed")
            return None

        n_inliers = len(inliers) if inliers is not None else 0
        print(f"[calibrate] inliers: {n_inliers}/{len(all_uv)} ({n_inliers/len(all_uv):.1%})")

        R, _ = cv2.Rodrigues(rvec)
        tvec_full = (-R.T @ tvec).ravel()

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = tvec_full

        reproj_uv, _ = cv2.projectPoints(
            xyz_centered.astype(np.float64), rvec, tvec, self.K.astype(np.float64), dist_coeffs
        )
        errors = np.linalg.norm(all_uv.astype(np.float64) - reproj_uv.squeeze(), axis=1)
        mean_reproj = float(errors.mean())
        print(f"[calibrate] mean reprojection error: {mean_reproj:.2f} px")

        rot_error = float(np.linalg.norm(rvec) / max(n_inliers, 1) * 180 / np.pi)

        return HandEyeResult(
            link_T_camera=T,
            trans_error=float(np.linalg.norm(tvec_full) * 0.001),
            rot_error=rot_error,
            n_samples=len(self._pairs),
        )

    # ── Save & load ─────────────────────────────────────────────

    def save_handeye(self, result: HandEyeResult, mounted_link: str = "camera_mount"):
        """Write calibration result to handeye_calibration.json."""
        payload = {
            "mounted_link": mounted_link,
            "link_T_camera": result.link_T_camera.tolist(),
            "calibration_info": {
                "n_samples": result.n_samples,
                "trans_error": result.trans_error,
                "rot_error": result.rot_error,
            },
        }
        self.handeye_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.handeye_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        R = result.link_T_camera[:3, :3]
        trace = np.clip((np.trace(R) - 1) / 2, -1, 1)
        rot_deg = float(np.degrees(np.arccos(trace)))
        print(f"[save_handeye] -> {self.handeye_json_path}")
        print(f"  translation = {result.link_T_camera[:3, 3].round(4)}")
        print(f"  rotation    = {rot_deg:.2f} deg")

    # ── One-shot run ────────────────────────────────────────────

    def run(self):
        """
        Full online calibration pipeline:
          1. Collect data (interactive guidance)
          2. Solve
          3. Save to handeye_calibration.json
        """
        print("=" * 60)
        print("  Online Hand-Eye Calibration")
        print("=" * 60)
        print(f"  n_poses      : {self.n_poses}")
        print(f"  detector     : {self.feature_detector}")
        print(f"  output file  : {self.handeye_json_path}")
        print("=" * 60)
        print()
        print("Robot will execute small joint motions automatically.")
        print("Make sure the camera has sufficient texture (not blank wall).")
        print()

        input("Press Enter to start data collection...")
        print()

        n = self.collect_data()
        if n < 6:
            print("[run] Insufficient valid data, aborted.")
            return None

        print("\n[run] Solving...")
        result = self.calibrate()
        if result is None:
            print("[run] Calibration failed.")
            return None

        print("\n[run] Saving...")
        self.save_handeye(result)
        print("\nDone! Re-create RobotController to load the new hand-eye relation.")
        return result


# ── Convenience entry ─────────────────────────────────────────

def calibrate_online(robot, camera_source: int | str = 0, n_poses: int = 12, feature_detector: str = "orb"):
    """
    Quick call for online hand-eye calibration.

    Example:
        from lab.online_handeye_calibration import calibrate_online
        result = calibrate_online(my_robot, camera_source=0)
    """
    calib = OnlineHandEyeCalibrator(robot=robot, camera_source=camera_source,
                                     n_poses=n_poses, feature_detector=feature_detector)
    return calib.run()
