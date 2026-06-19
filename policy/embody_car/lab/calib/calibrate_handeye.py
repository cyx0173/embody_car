import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMERA_NATIVE_WIDTH = 1920
CAMERA_NATIVE_HEIGHT = 1080
CAMERA_SQUARE_CROP_SIZE = min(CAMERA_NATIVE_WIDTH, CAMERA_NATIVE_HEIGHT)
CAMERA_WORKING_WIDTH = CAMERA_SQUARE_CROP_SIZE // 2
CAMERA_WORKING_HEIGHT = CAMERA_SQUARE_CROP_SIZE // 2

from visualize_so101 import (  # noqa: E402
    build_hardware_command,
    build_tree,
    compute_fk,
    default_joint_value,
    parse_urdf_like,
    slider_range,
)

try:
    from RoboDriver import RoboArmJoints
except Exception:
    RoboArmJoints = None


# ── lerobot calibration 文件支持 ─────────────────────────────────────────────
# Dynamixel 舵机：12-bit 编码器，全量程 4096 步对应 360°（2π rad）
TICKS_PER_REV = 4096


def load_calibration(path: Path) -> dict:
    """读取 lerobot 格式的 calibration JSON，返回关节名 -> 参数字典。"""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_zero_encoder(calib_joint: dict) -> int:
    lower = int(calib_joint["range_min"])
    upper = int(calib_joint["range_max"])
    offset = int(calib_joint["homing_offset"])

    candidates = (
        offset,
        offset % TICKS_PER_REV,
        2048 + offset,
        2048 - offset,
    )
    for candidate in candidates:
        candidate_int = int(candidate)
        if lower <= candidate_int <= upper:
            return candidate_int

    return int((lower + upper) / 2)


def encoder_to_rad(encoder_value: float, calib_joint: dict) -> float:
    """
    Dynamixel 编码器值 → 关节弧度。
    零位定义：encoder_value == calibration 解析出的零位时输出 0 rad。
    drive_mode=1 表示该轴方向取反（lerobot 约定）。
    """
    zero_encoder = resolve_zero_encoder(calib_joint)
    ticks = encoder_value - zero_encoder
    if calib_joint["drive_mode"]:
        ticks = -ticks
    return ticks / TICKS_PER_REV * 2.0 * np.pi


def rad_to_encoder(rad: float, calib_joint: dict) -> int:
    """
    关节弧度 → Dynamixel 编码器值。
    用于将滑块弧度值转回舵机可接受的编码器指令。
    """
    if calib_joint["drive_mode"]:
        rad = -rad
    ticks = rad / (2.0 * np.pi) * TICKS_PER_REV
    zero_encoder = resolve_zero_encoder(calib_joint)
    return int(round(ticks + zero_encoder))


def calib_range_rad(calib_joint: dict) -> tuple[float, float]:
    """
    从 calibration 条目的 range_min / range_max 换算出关节弧度范围。
    drive_mode=1 时 min/max 会互换，取 min/max 保证顺序正确。
    """
    lo = encoder_to_rad(calib_joint["range_min"], calib_joint)
    hi = encoder_to_rad(calib_joint["range_max"], calib_joint)
    return (min(lo, hi), max(lo, hi))
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DetectionResult:
    found: bool
    corners: np.ndarray | None
    refined_corners: np.ndarray | None
    frame: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    target_T_camera: np.ndarray | None
    reprojection_error_px: float | None
    message: str = ""


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * square_size
    return object_points


def orient_chessboard_corners(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    oriented = np.array(corners, dtype=np.float32).reshape(rows, cols, 2).copy()

    if cols > 1:
        u = oriented[0, 1] - oriented[0, 0]
        if u[0] < 0:
            oriented = oriented[:, ::-1, :]

    if rows > 1:
        v = oriented[1, 0] - oriented[0, 0]
        if v[1] < 0:
            oriented = oriented[::-1, :, :]

    if cols > 1 and rows > 1:
        u = oriented[0, 1] - oriented[0, 0]
        v = oriented[1, 0] - oriented[0, 0]
        signed_area = float(u[0] * v[1] - u[1] * v[0])
        if signed_area < 0:
            oriented = oriented[::-1, :, :]

    return oriented.reshape(-1, 1, 2)


def bilinear_sample(gray: np.ndarray, point: np.ndarray) -> float:
    height, width = gray.shape[:2]
    x = float(np.clip(point[0], 0.0, width - 1.001))
    y = float(np.clip(point[1], 0.0, height - 1.001))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    top = (1.0 - dx) * gray[y0, x0] + dx * gray[y0, x1]
    bottom = (1.0 - dx) * gray[y1, x0] + dx * gray[y1, x1]
    return float((1.0 - dy) * top + dy * bottom)


def orient_chessboard_with_color(gray: np.ndarray, corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    oriented = orient_chessboard_corners(corners, cols, rows).reshape(rows, cols, 2)
    if cols < 2 or rows < 2:
        return oriented.reshape(-1, 1, 2)

    origin = oriented[0, 0]
    u = oriented[0, 1] - oriented[0, 0]
    v = oriented[1, 0] - oriented[0, 0]
    u_norm = float(np.linalg.norm(u))
    v_norm = float(np.linalg.norm(v))
    if u_norm < 1e-6 or v_norm < 1e-6:
        return oriented.reshape(-1, 1, 2)

    step = 0.35 * min(u_norm, v_norm)
    u_hat = u / u_norm
    v_hat = v / v_norm

    nw = bilinear_sample(gray, origin - step * u_hat - step * v_hat)
    ne = bilinear_sample(gray, origin + step * u_hat - step * v_hat)
    sw = bilinear_sample(gray, origin - step * u_hat + step * v_hat)
    se = bilinear_sample(gray, origin + step * u_hat + step * v_hat)

    # Our generated board starts with a dark square at the outer top-left corner,
    # so the first inner corner should see dark/bright/bright/dark around it.
    correct_score = (255.0 - nw) + ne + sw + (255.0 - se)
    flipped_score = nw + (255.0 - ne) + (255.0 - sw) + se
    if flipped_score > correct_score:
        oriented = oriented[::-1, ::-1, :]
    return oriented.reshape(-1, 1, 2)


def ensure_board_settings(
    intrinsics_payload: dict,
    cols: int | None,
    rows: int | None,
    square_size: float | None,
) -> tuple[int, int, float]:
    board = intrinsics_payload.get("board", {})
    resolved_cols = cols if cols is not None else board.get("inner_corners_cols")
    resolved_rows = rows if rows is not None else board.get("inner_corners_rows")
    resolved_square_size = square_size if square_size is not None else board.get("square_size_m")
    if resolved_cols is None or resolved_rows is None or resolved_square_size is None:
        raise ValueError("Board settings are incomplete. Provide --cols, --rows, and --square-size.")
    return int(resolved_cols), int(resolved_rows), float(resolved_square_size)


def load_intrinsics(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=float)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=float).reshape(-1, 1)
    return payload, camera_matrix, dist_coeffs


def scaled_camera_matrix_for_size(
    camera_matrix: np.ndarray,
    intrinsics_payload: dict,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    source_width = int(intrinsics_payload.get("image_width", target_width))
    source_height = int(intrinsics_payload.get("image_height", target_height))
    if source_width <= 0 or source_height <= 0:
        return np.array(camera_matrix, dtype=float)
    if source_width == target_width and source_height == target_height:
        return np.array(camera_matrix, dtype=float)
    scaled = np.array(camera_matrix, dtype=float)
    scaled[0, 0] *= float(target_width) / float(source_width)
    scaled[1, 1] *= float(target_height) / float(source_height)
    scaled[0, 2] *= float(target_width) / float(source_width)
    scaled[1, 2] *= float(target_height) / float(source_height)
    return scaled


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


def pose_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array(tvec, dtype=float).reshape(3)
    return transform


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-8
    if not singular:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def format_transform_debug(name: str, transform: np.ndarray | None) -> list[str]:
    if transform is None:
        return [f"{name}: unavailable"]

    translation = transform[:3, 3]
    rpy_rad = matrix_to_rpy(transform[:3, :3])
    rpy_deg = np.degrees(rpy_rad)
    return [
        f"{name} t[m]: [{translation[0]:+.4f}, {translation[1]:+.4f}, {translation[2]:+.4f}]",
        f"{name} rpy[deg]: [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]",
    ]


def save_handeye(path: Path, mounted_link: str, link_T_camera: np.ndarray, sample_count: int) -> None:
    rotation = link_T_camera[:3, :3]
    translation = link_T_camera[:3, 3]
    rpy_rad = matrix_to_rpy(rotation)
    payload = {
        "mounted_link": mounted_link,
        "sample_count": int(sample_count),
        "link_T_camera": link_T_camera.tolist(),
        "translation_xyz_m": translation.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rpy_rad": rpy_rad.tolist(),
        "rpy_deg": np.degrees(rpy_rad).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_sample_manifest(
    path: Path,
    mounted_link: str,
    cols: int,
    rows: int,
    square_size: float,
    samples: list[dict],
) -> None:
    payload = {
        "mounted_link": mounted_link,
        "board": {
            "type": "chessboard",
            "inner_corners_cols": int(cols),
            "inner_corners_rows": int(rows),
            "square_size_m": float(square_size),
        },
        "sample_count": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_existing_handeye_guess(path: Path, mounted_link: str) -> tuple[np.ndarray | None, str | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_link = payload.get("mounted_link")
        if saved_link is not None and saved_link != mounted_link:
            return None, f"handeye guess ignored: mounted_link={saved_link}"
        link_T_camera = np.array(payload["link_T_camera"], dtype=float)
        if link_T_camera.shape != (4, 4):
            return None, "handeye guess ignored: invalid shape"
        return link_T_camera, "using existing handeye guess for board world preview"
    except Exception as exc:
        return None, f"handeye guess ignored: {type(exc).__name__}: {exc}"


def detect_chessboard(
    frame: np.ndarray,
    pattern_size: tuple[int, int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> DetectionResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return DetectionResult(
            found=False,
            corners=corners,
            refined_corners=None,
            frame=frame,
            rvec=None,
            tvec=None,
            target_T_camera=None,
            reprojection_error_px=None,
            message="chessboard not detected",
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    refined = orient_chessboard_with_color(gray, refined, pattern_size[0], pattern_size[1])
    ok, rvec, tvec = cv2.solvePnP(object_points, refined, camera_matrix, dist_coeffs)
    if not ok:
        return DetectionResult(
            found=False,
            corners=corners,
            refined_corners=refined,
            frame=frame,
            rvec=None,
            tvec=None,
            target_T_camera=None,
            reprojection_error_px=None,
            message="solvePnP failed",
        )

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    observed = refined.reshape(-1, 2)
    reprojection_error_px = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))
    return DetectionResult(
        found=True,
        corners=corners,
        refined_corners=refined,
        frame=frame,
        rvec=rvec,
        tvec=tvec,
        target_T_camera=pose_to_transform(rvec, tvec),
        reprojection_error_px=reprojection_error_px,
        message="ready",
    )


def draw_detection_overlay(
    frame: np.ndarray,
    detection: DetectionResult,
    pattern_size: tuple[int, int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    display = frame.copy()
    principal_point = np.round(camera_matrix[:2, 2]).astype(int)
    cv2.drawMarker(
        display,
        tuple(principal_point),
        (255, 165, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "principal point",
        (principal_point[0] + 12, principal_point[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 165, 0),
        2,
        cv2.LINE_AA,
    )
    if detection.found and detection.refined_corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, detection.refined_corners, True)
        if detection.rvec is not None and detection.tvec is not None:
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, detection.rvec, detection.tvec, 0.05, 2)
            origin_object_point = object_points[:1]
            origin_image_point, _ = cv2.projectPoints(
                origin_object_point,
                detection.rvec,
                detection.tvec,
                camera_matrix,
                dist_coeffs,
            )
            origin_xy = np.round(origin_image_point.reshape(2)).astype(int)
            cv2.circle(display, tuple(origin_xy), 7, (0, 255, 255), -1)
            cv2.circle(display, tuple(origin_xy), 11, (0, 0, 0), 2)
            cv2.putText(
                display,
                "target origin",
                (origin_xy[0] + 12, origin_xy[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    color = (0, 190, 0) if detection.found else (0, 0, 255)
    cv2.putText(display, detection.message, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if detection.reprojection_error_px is not None:
        cv2.putText(
            display,
            f"reproj={detection.reprojection_error_px:.3f}px",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def current_link_pose(
    root_link: str,
    children_map: dict[str, list[object]],
    joint_values: dict[str, float],
    mounted_link: str,
) -> np.ndarray:
    link_frames, _ = compute_fk(root_link, children_map, joint_values)
    if mounted_link not in link_frames:
        raise KeyError(f"Mounted link '{mounted_link}' not found in robot model.")
    return link_frames[mounted_link]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive eye-in-hand calibration workflow.")
    parser.add_argument("--intrinsics", default="camera_intrinsics.json", help="Path to intrinsics JSON.")
    parser.add_argument("--xacro", default="so101_follower.urdf.xacro", help="Simplified xacro path.")
    parser.add_argument("--mounted-link", default="gripper_link", help="Robot link where the camera is mounted.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--raw-width", type=int, default=CAMERA_NATIVE_WIDTH, help="Requested camera capture width before downsampling.")
    parser.add_argument("--raw-height", type=int, default=CAMERA_NATIVE_HEIGHT, help="Requested camera capture height before downsampling.")
    parser.add_argument("--width", type=int, default=CAMERA_WORKING_WIDTH, help="Working image width after downsampling.")
    parser.add_argument("--height", type=int, default=CAMERA_WORKING_HEIGHT, help="Working image height after downsampling.")
    parser.add_argument("--serial-port", default="/dev/cu.usbmodem5AE60562991", help="Optional serial port for the robot arm, e.g. /dev/tty.usbserial-0001 or COM3.")
    parser.add_argument("--cols", type=int, default=None, help="Chessboard inner corner columns.")
    parser.add_argument("--rows", type=int, default=None, help="Chessboard inner corner rows.")
    parser.add_argument("--square-size", type=float, default=None, help="Chessboard square size in meters.")
    parser.add_argument("--output", default="handeye_calibration.json", help="Output calibration file.")
    parser.add_argument("--samples-json", default="handeye_samples.json", help="Saved sample manifest.")
    parser.add_argument("--images-dir", default="handeye_images", help="Directory to store captured images.")
    parser.add_argument("--min-samples", type=int, default=12, help="Minimum recommended sample count.")
    # ── 新增：lerobot calibration 文件路径参数 ──
    parser.add_argument(
        "--calibration",
        default="calibration.json",
        help="Path to lerobot-format calibration JSON (homing_offset / range_min / range_max).",
    )
    args = parser.parse_args()

    intrinsics_path = (PROJECT_ROOT / args.intrinsics).resolve() if not Path(args.intrinsics).is_absolute() else Path(args.intrinsics)
    xacro_path = (PROJECT_ROOT / args.xacro).resolve() if not Path(args.xacro).is_absolute() else Path(args.xacro)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    samples_path = (PROJECT_ROOT / args.samples_json).resolve() if not Path(args.samples_json).is_absolute() else Path(args.samples_json)
    images_dir = (PROJECT_ROOT / args.images_dir).resolve() if not Path(args.images_dir).is_absolute() else Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 lerobot calibration 文件 ──
    calib_path = (PROJECT_ROOT / args.calibration).resolve() if not Path(args.calibration).is_absolute() else Path(args.calibration)
    if calib_path.exists():
        calib = load_calibration(calib_path)
        print(f"[calibration] loaded {calib_path} ({len(calib)} joints)")
    else:
        calib = {}
        print(f"[calibration] WARNING: {calib_path} not found, falling back to URDF defaults")

    intrinsics_payload, camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path)
    target_width = int(args.width)
    target_height = int(args.height)
    target_camera_matrix = scaled_camera_matrix_for_size(camera_matrix, intrinsics_payload, target_width, target_height)
    cols, rows, square_size = ensure_board_settings(intrinsics_payload, args.cols, args.rows, args.square_size)
    object_points = build_object_points(cols, rows, square_size)
    pattern_size = (cols, rows)

    _, joints = parse_urdf_like(str(xacro_path))
    children_map, parent_links, child_links, _ = build_tree(joints)
    roots = sorted(parent_links - child_links)
    root_link = "world" if "world" in parent_links else roots[0]

    movable_joints = [joint for joint in joints if joint.joint_type in {"revolute", "continuous"}]

    # ── 关节零位：calibration 中 homing_offset 对应的物理零位即 0 rad ──
    joint_values: dict[str, float] = {}
    for joint in movable_joints:
        if joint.name in calib:
            joint_values[joint.name] = 0.0  # homing_offset 定义的就是零位
        else:
            joint_values[joint.name] = default_joint_value(joint)  # fallback

    samples: list[dict] = []
    latest_detection = DetectionResult(False, None, None, None, None, None, None, None, "starting camera...")
    preview_link_T_camera, preview_handeye_message = load_existing_handeye_guess(output_path, args.mounted_link)
    debug_state = {
        "base_T_link": None,
        "target_T_camera": None,
        "board_origin_world": None,
    }

    hardware_state = {
        "arm": None,
        "connecting": False,
        "connected": False,
        "message": "driver: idle",
    }
    motion_state = {
        "pending_command": None,
        "sending": False,
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.raw_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.raw_height))
    reported_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) or int(args.raw_width)
    reported_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) or int(args.raw_height)

    fig = plt.figure(figsize=(13.5, 9.2))
    ax = fig.add_axes([0.06, 0.52, 0.6, 0.42])
    ax.axis("off")
    status_ax = fig.add_axes([0.70, 0.58, 0.27, 0.32])
    status_ax.axis("off")
    status_text = status_ax.text(0.0, 1.0, "", va="top", fontsize=10)

    slider_by_name: dict[str, Slider] = {}
    for index, joint in enumerate(movable_joints):
        slider_ax = fig.add_axes([0.06, 0.44 - index * 0.05, 0.6, 0.025])

        # ── 关节范围：优先用 calibration 的 range_min/max，否则用 URDF limit ──
        if joint.name in calib:
            lower, upper = calib_range_rad(calib[joint.name])
        else:
            lower, upper = slider_range(joint)

        slider = Slider(
            ax=slider_ax,
            label=f"{joint.name} (rad)",
            valmin=lower,
            valmax=upper,
            valinit=joint_values[joint.name],
        )
        slider_by_name[joint.name] = slider

    cols_box = TextBox(fig.add_axes([0.70, 0.49, 0.08, 0.05]), "cols", initial=str(cols))
    rows_box = TextBox(fig.add_axes([0.80, 0.49, 0.08, 0.05]), "rows", initial=str(rows))
    square_size_box = TextBox(fig.add_axes([0.70, 0.41, 0.18, 0.05]), "square(m)", initial=f"{square_size:.5f}")
    apply_board_button = Button(fig.add_axes([0.90, 0.41, 0.07, 0.05]), "Apply")

    connect_button = Button(fig.add_axes([0.70, 0.33, 0.27, 0.06]), "Connect")
    capture_button = Button(fig.add_axes([0.70, 0.25, 0.27, 0.06]), "Capture Sample")
    solve_button = Button(fig.add_axes([0.70, 0.17, 0.27, 0.06]), "Solve HandEye")
    reset_button = Button(fig.add_axes([0.70, 0.09, 0.27, 0.06]), "Reset Pose")

    help_ax = fig.add_axes([0.70, 0.00, 0.27, 0.08])
    help_ax.axis("off")
    help_ax.text(
        0.0,
        1.0,
        "Workflow:\n"
        "1. Drag sliders to move the real arm.\n"
        "2. Keep the chessboard static in the world.\n"
        "3. Capture 12+ samples with varied viewpoints.\n"
        "4. Click Solve HandEye to save gripper_link -> camera.",
        va="top",
        fontsize=9,
    )

    image_artist = ax.imshow(np.zeros((target_height, target_width, 3), dtype=np.uint8))
    ax.set_title("Camera Preview")

    def refresh_status(extra: str = "") -> None:
        lines = [
            f"samples: {len(samples)}",
            f"mounted_link: {args.mounted_link}",
            f"camera: {args.camera}",
            f"requested camera mode: {args.raw_width}x{args.raw_height}",
            f"camera reported mode: {reported_width}x{reported_height}",
            f"square crop: {CAMERA_SQUARE_CROP_SIZE}x{CAMERA_SQUARE_CROP_SIZE}",
            f"working image size: {target_width}x{target_height}",
            f"serial_port: {args.serial_port or 'auto'}",
            f"board: {cols}x{rows}, square={square_size:.5f} m",
            f"calibration: {'loaded' if calib else 'URDF fallback'}",
            latest_detection.message,
            hardware_state["message"],
        ]
        if preview_handeye_message:
            lines.append(preview_handeye_message)
        if latest_detection.reprojection_error_px is not None:
            lines.append(f"preview reproj: {latest_detection.reprojection_error_px:.3f}px")
        lines.extend(format_transform_debug("base_T_link", debug_state["base_T_link"]))
        lines.extend(format_transform_debug("target_T_camera", debug_state["target_T_camera"]))
        board_origin_world = debug_state["board_origin_world"]
        if board_origin_world is not None:
            lines.append(
                f"board origin world[m]: [{board_origin_world[0]:+.4f}, {board_origin_world[1]:+.4f}, {board_origin_world[2]:+.4f}]"
            )
        if extra:
            lines.append(extra)
        status_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def update_preview() -> None:
        nonlocal latest_detection
        ok, frame = cap.read()
        if not ok:
            latest_detection = DetectionResult(False, None, None, None, None, None, None, None, "camera read failed")
            refresh_status()
            return
        frame = center_crop_and_resize_frame(frame, (target_width, target_height))

        detection = detect_chessboard(frame, pattern_size, object_points, target_camera_matrix, dist_coeffs)
        latest_detection = detection
        try:
            debug_state["base_T_link"] = current_link_pose(root_link, children_map, joint_values, args.mounted_link)
        except Exception:
            debug_state["base_T_link"] = None
        debug_state["target_T_camera"] = detection.target_T_camera if detection.found else None
        if (
            debug_state["base_T_link"] is not None
            and detection.target_T_camera is not None
            and preview_link_T_camera is not None
        ):
            base_T_camera = debug_state["base_T_link"] @ preview_link_T_camera
            base_T_target = base_T_camera @ detection.target_T_camera
            debug_state["board_origin_world"] = base_T_target[:3, 3].copy()
        else:
            debug_state["board_origin_world"] = None
        display = draw_detection_overlay(frame, detection, pattern_size, object_points, target_camera_matrix, dist_coeffs)
        image_artist.set_data(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        refresh_status()

    timer = fig.canvas.new_timer(interval=80)
    timer.add_callback(update_preview)
    timer.start()

    def queue_pose_send() -> None:
        if not hardware_state["connected"] or hardware_state["arm"] is None:
            return

        # RoboArmJoints.joints_move_radian expects radians, so send the UI joint
        # values directly instead of converting them to encoder ticks here.
        motion_state["pending_command"] = build_hardware_command(joint_values)
        if motion_state["sending"]:
            return

        def worker() -> None:
            while motion_state["pending_command"] is not None:
                command = motion_state["pending_command"]
                motion_state["pending_command"] = None
                motion_state["sending"] = True
                try:
                    hardware_state["arm"].joints_move_radian(command, speed=500, acc=1000)
                    hardware_state["message"] = "driver: live pose sent"
                except Exception as exc:
                    hardware_state["message"] = f"driver: {type(exc).__name__}: {exc}"
                    motion_state["pending_command"] = None
                    break
            motion_state["sending"] = False
            fig.canvas.draw_idle()

        threading.Thread(target=worker, daemon=True).start()

    def on_slider_change(_: float) -> None:
        for name, slider in slider_by_name.items():
            joint_values[name] = float(slider.val)
        queue_pose_send()
        refresh_status()

    for slider in slider_by_name.values():
        slider.on_changed(on_slider_change)

    def connect_hardware_async(_: object) -> None:
        if RoboArmJoints is None:
            hardware_state["message"] = "driver: import failed"
            refresh_status()
            return
        if hardware_state["connecting"] or hardware_state["connected"]:
            refresh_status()
            return

        hardware_state["connecting"] = True
        hardware_state["message"] = "driver: connecting..."
        refresh_status()

        def worker() -> None:
            try:
                hardware_state["arm"] = RoboArmJoints(serial_port=args.serial_port)
                hardware_state["connected"] = True
                hardware_state["message"] = "driver: connected"
            except Exception as exc:
                hardware_state["connected"] = False
                hardware_state["arm"] = None
                hardware_state["message"] = f"driver: {type(exc).__name__}: {exc}"
            finally:
                hardware_state["connecting"] = False
                fig.canvas.draw_idle()

        threading.Thread(target=worker, daemon=True).start()

    def apply_board_settings(_: object) -> None:
        nonlocal cols, rows, square_size, object_points, pattern_size, samples
        try:
            new_cols = int(cols_box.text)
            new_rows = int(rows_box.text)
            new_square_size = float(square_size_box.text)
        except ValueError:
            refresh_status("board update failed: invalid values")
            return

        if new_cols <= 0 or new_rows <= 0 or new_square_size <= 0:
            refresh_status("board update failed: values must be positive")
            return

        if samples:
            samples = []
            save_sample_manifest(samples_path, args.mounted_link, new_cols, new_rows, new_square_size, samples)

        cols = new_cols
        rows = new_rows
        square_size = new_square_size
        object_points = build_object_points(cols, rows, square_size)
        pattern_size = (cols, rows)
        refresh_status("board settings applied; samples cleared")

    def capture_sample(_: object) -> None:
        if not latest_detection.found or latest_detection.target_T_camera is None or latest_detection.frame is None:
            refresh_status("capture failed: chessboard not ready")
            return

        try:
            base_T_link = current_link_pose(root_link, children_map, joint_values, args.mounted_link)
        except Exception as exc:
            refresh_status(f"capture failed: {type(exc).__name__}: {exc}")
            return

        image_index = len(samples) + 1
        image_path = images_dir / f"sample_{image_index:03d}.png"
        cv2.imwrite(str(image_path), latest_detection.frame)

        sample = {
            "index": image_index,
            "image_path": str(image_path),
            "joint_values": {name: float(value) for name, value in joint_values.items()},
            "base_T_link": base_T_link.tolist(),
            "target_T_camera": latest_detection.target_T_camera.tolist(),
            "reprojection_error_px": latest_detection.reprojection_error_px,
        }
        samples.append(sample)
        save_sample_manifest(samples_path, args.mounted_link, cols, rows, square_size, samples)
        refresh_status(f"captured sample {image_index}")

    def solve_handeye(_: object) -> None:
        nonlocal preview_link_T_camera, preview_handeye_message
        if len(samples) < max(3, args.min_samples):
            refresh_status(f"need at least {max(3, args.min_samples)} samples")
            return

        base_R_gripper = []
        base_t_gripper = []
        target_R_cam = []
        target_t_cam = []
        for sample in samples:
            base_T_link = np.array(sample["base_T_link"], dtype=float)
            target_T_cam = np.array(sample["target_T_camera"], dtype=float)
            base_R_gripper.append(base_T_link[:3, :3])
            base_t_gripper.append(base_T_link[:3, 3].reshape(3, 1))
            target_R_cam.append(target_T_cam[:3, :3])
            target_t_cam.append(target_T_cam[:3, 3].reshape(3, 1))

        try:
            link_R_camera, link_t_camera = cv2.calibrateHandEye(
                R_gripper2base=base_R_gripper,
                t_gripper2base=base_t_gripper,
                R_target2cam=target_R_cam,
                t_target2cam=target_t_cam,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )
        except Exception as exc:
            refresh_status(f"solve failed: {type(exc).__name__}: {exc}")
            return

        link_T_camera = np.eye(4, dtype=float)
        link_T_camera[:3, :3] = np.array(link_R_camera, dtype=float)
        link_T_camera[:3, 3] = np.array(link_t_camera, dtype=float).reshape(3)
        save_handeye(output_path, args.mounted_link, link_T_camera, len(samples))
        preview_link_T_camera = link_T_camera.copy()
        preview_handeye_message = "using solved handeye for board world preview"
        refresh_status(f"saved {output_path.name}")
        print("link_T_camera:")
        print(link_T_camera)

    def on_reset(_: object) -> None:
        # ── Reset 也对齐 calibration 零位（0 rad = homing_offset 处）──
        for joint in movable_joints:
            if joint.name in calib:
                value = 0.0
            else:
                value = default_joint_value(joint)
            joint_values[joint.name] = value
            slider_by_name[joint.name].set_val(value)
        refresh_status("pose reset")

    apply_board_button.on_clicked(apply_board_settings)
    connect_button.on_clicked(connect_hardware_async)
    capture_button.on_clicked(capture_sample)
    solve_button.on_clicked(solve_handeye)
    reset_button.on_clicked(on_reset)

    refresh_status("ready")
    try:
        plt.show()
    finally:
        timer.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
