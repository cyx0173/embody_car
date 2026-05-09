
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from landmark_demo_v2.workflow_utils import BUNDLE_ROOT, discover_latest_pose_dataset, ensure_frames_dir
from landmark_demo_v2.text_landmark_pipeline import load_detail


def _sanitize(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "case"


# ─── Robot 运行时加载 ────────────────────────────────────────

class RobotController:
    """封装 landmark_demo_v2 的 RobotController，复用 IK + 串口执行。"""

    def __init__(
        self,
        serial_port: str | None = None,
        landmark_npz_path: Path | str | None = None,
        window_idx: int = 0,
        threshold_px: float = 10.0,
        top_n: int = 500,
    ):
        self.serial_port = serial_port
        self.window_idx = window_idx
        self.threshold_px = threshold_px
        self.top_n = top_n

        # 加载 landmark 数据库
        self.landmark_npz_path = self._resolve_npz(landmark_npz_path)
        self._load_landmark_db()

        # 加载 RobotController（依赖 robot runtime）
        self._robot = self._load_robot_runtime()

    def _resolve_npz(self, landmark_npz_path: Path | str | None) -> Path:
        if landmark_npz_path:
            return Path(landmark_npz_path)

        try:
            dataset_json, _ = discover_latest_pose_dataset(
                BUNDLE_ROOT / "data",
                BUNDLE_ROOT / "data" / "test_data",
                min_frames=5,
            )
        except Exception:
            raise FileNotFoundError("No pose dataset found. Run capture_camera_pose_dataset.py first.")

        # 搜索 notebook_outputs 下对应的 output 目录中的 npz
        digest = hashlib.sha1(str(dataset_json.resolve()).encode("utf-8")).hexdigest()[:8]
        safe = _sanitize(f"{dataset_json.parent.name}_{dataset_json.stem}_{digest}")
        output_root = BUNDLE_ROOT / "notebook_outputs" / safe

        candidates = sorted(output_root.glob("details/*_details.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No landmark detail npz found in {output_root / 'details'}. "
                "Run auto_landmark_pipeline first to generate landmarks."
            )

        # 默认取 window_00
        default = next((p for p in candidates if "window_00" in p.name), candidates[-1])
        print(f"[RobotController] using landmark npz: {default}")
        return default

    def _load_landmark_db(self):
        self._detail_data, df = load_detail(self.landmark_npz_path, threshold_px=self.threshold_px)

        # 构建 active 候选集
        ranked = df[df["all_later_valid"] & np.isfinite(df["mean_later_err"])].copy()
        active = ranked[ranked["alive_under_threshold"]].copy()
        if active.empty:
            active = ranked.nsmallest(self.top_n, "mean_later_err").copy()
        if len(active) > self.top_n:
            active = active.nsmallest(self.top_n, "mean_later_err").copy()
        self._landmarks_df = active.reset_index(drop=True)

        self._image_size = 128  # landmark 图像坐标基于 128×128

    def _load_robot_runtime(self):
        from landmark_demo_v2.robot_runtime_bridge import RobotController as RC
        return RC(serial_port=self.serial_port)

    def find_nearest_landmark(self, u: float, v: float, image_size: int = 640) -> pd.Series | None:
        """
        在 landmark 数据库中找 (u,v) 最近邻。

        u, v        : 当前图像中的检测中心（像素坐标）
        image_size  : 当前图像尺寸（YOLO 输入尺寸）
        """
        if self._landmarks_df.empty:
            return None

        # 缩放到 128×128 坐标系
        scale = self._image_size / image_size
        u_scaled = u * scale
        v_scaled = v * scale

        df = self._landmarks_df
        dists = (df["source_u"] - u_scaled) ** 2 + (df["source_v"] - v_scaled) ** 2
        idx = int(dists.idxmin())
        return df.iloc[idx]

    def execute_motion(self, location: tuple[float, float], image) -> bool:
        u, v = location

        H, W = image.shape[:2]
        landmark = self.find_nearest_landmark(u, v, image_size=max(H, W))
        if landmark is None:
            return False

        xyz = np.array([landmark["x"], landmark["y"], landmark["z"]], dtype=float)
        print(f"[execute_motion] matched landmark {int(landmark['point_id'])}: "
              f"uv=({u:.1f},{v:.1f}) → xyz=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f})")

        joints, error = self._robot.solve_ik(xyz)
        print(f"[execute_motion] IK solved: error={error:.4f}m joints={joints}")

        if self.serial_port:
            ok, msg = self._robot.execute_pose(joints)
            print(f"[execute_motion] {'OK' if ok else 'FAILED'}: {msg}")
            return ok
        return True


# ─── 视觉追踪 ────────────────────────────────────────────────

def get_location(image, target: str, model, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> tuple[float, float] | None:
    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

    if not results or results[0].boxes is None:
        return None

    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    names = results[0].names

    all_detections = []
    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        cls_name = names[cls_idx]
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].item())
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        all_detections.append((cls_name, center, conf))
        print(f"[get_location] [{i}] class={cls_name} center=({center[0]:.1f}, {center[1]:.1f}) conf={conf:.2f}")

    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        cls_name = names[cls_idx]
        if cls_name == target:
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            return ((x1 + x2) / 2, (y1 + y2) / 2)

    return None


class VisualTracking:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        serial_port: str | None = None,
        landmark_npz_path: Path | str | None = None,
        window_idx: int = 0,
        threshold_px: float = 10.0,
        top_n: int = 500,
    ):
        from shutil import copy2

        model_dir = Path(__file__).parent / "model"
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / model_name
        if model_path.exists():
            print(f"[VisualTracking] Using local model: {model_path}")
        else:
            print(f"[VisualTracking] Model not found, downloading '{model_name}' ...")
            _tmp = YOLO(model_name)
            # 搜索缓存目录 + 当前工作目录（ultralytics 可能下载到这里）
            for search_root in [Path.home() / ".cache" / "ultralytics", Path.cwd()]:
                cached = next(search_root.rglob(model_name), None)
                if cached:
                    copy2(str(cached), str(model_path))
                    print(f"[VisualTracking] Model saved to {model_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"Failed to download '{model_name}', "
                    "please download manually and place in lab/model/"
                )

        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.robot = RobotController(
            serial_port=serial_port,
            landmark_npz_path=landmark_npz_path,
            window_idx=window_idx,
            threshold_px=threshold_px,
            top_n=top_n,
        )

    def track(self, target: str, image) -> bool:
        """
        追踪流程：
          1. YOLO 检测目标 → (u, v)
          2. 查 landmark 数据库最近邻
          3. 取 3D 世界坐标 → IK
          4. 串口发送执行
        """
        location = get_location(image, target, self.model, self.conf_threshold, self.iou_threshold)
        if location is None:
            return False
        return self.robot.execute_motion(location, image)


# ─── 主入口 ────────────────────────────────────────────────

def main():
    YOLO_MODEL = "yolo11n.pt"          # YOLO 模型路径（已下载）
    SERIAL_PORT = "/dev/tty.usbmodem5AE60562991"        # 串口路径（根据实际修改）
    CAMERA_ID = 0                       # 摄像头 ID
    TARGET = "bottle"                    # 目标类别名

    vt = VisualTracking(
        model_name=YOLO_MODEL,
        device="cpu",
        conf_threshold=0.25,
        iou_threshold=0.45,
        serial_port=SERIAL_PORT,
        landmark_npz_path=None,          # 自动查找
        window_idx=0,
        threshold_px=10.0,
        top_n=500,
    )

    frame = cv2.imread(str(Path.cwd() / "img.png"))
    if frame is None:
        print("[main] img.png not found in current directory")
        return

    location = get_location(frame, TARGET, vt.model, vt.conf_threshold, vt.iou_threshold)
    if not location:
        print(f"[main] '{TARGET}' not found in img.png")
        return

    vt.track(TARGET, frame)
    print("[main] Done.")


if __name__ == "__main__":
    main()
