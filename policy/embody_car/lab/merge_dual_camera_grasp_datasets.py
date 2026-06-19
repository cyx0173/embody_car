#!/usr/bin/env python3
"""Merge same-schema dual-camera LeRobot grasp datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

WORKSPACE_DIR = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_SOURCES = (
    ("embody_car/orange_dual_camera_grasp_v1", WORKSPACE_DIR / "datasets" / "orange_dual_camera_grasp_v1"),
    ("embody_car/apple_dual_camera_grasp_v1", WORKSPACE_DIR / "datasets" / "apple_dual_camera_grasp_v1"),
)
DEFAULT_OUTPUT_NAME = "fruit_dual_camera_grasp_v1"


def make_dual_camera_features(*, height: int, width: int) -> dict:
    image_feature = {
        "dtype": "video",
        "shape": (3, height, width),
        "names": ["channels", "height", "width"],
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "observation.images.wrist": dict(image_feature),
        "observation.images.external": dict(image_feature),
    }


def create_output_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
    features: dict,
    vcodec: str,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id,
        fps,
        root=root,
        robot_type="custom_so101_follower",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=vcodec,
    )


def tensor_image_to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={tuple(image.shape)}")
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image_np = image.detach().cpu().numpy()
    if image_np.dtype != np.uint8:
        if image_np.max(initial=0.0) <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def tensor_to_float32_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def tensor_scalar_to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        root = Path(raw)
        return f"embody_car/{root.name}", root
    repo_id, root = raw.split("=", 1)
    return repo_id, Path(root)


def open_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset is missing {info_path}")
    return LeRobotDataset(repo_id, root=root)


def validate_compatible(reference: LeRobotDataset, candidate: LeRobotDataset, root: Path) -> None:
    if int(reference.fps) != int(candidate.fps):
        raise ValueError(f"FPS mismatch for {root}: {candidate.fps} != {reference.fps}")
    for key in ("observation.state", "action", "observation.images.wrist", "observation.images.external"):
        if key not in candidate.features:
            raise ValueError(f"Dataset {root} is missing feature {key}")
        if candidate.features[key]["shape"] != reference.features[key]["shape"]:
            raise ValueError(
                f"Feature shape mismatch for {root} key={key}: "
                f"{candidate.features[key]['shape']} != {reference.features[key]['shape']}"
            )


def copy_dataset_episodes(source: LeRobotDataset, output: LeRobotDataset, source_name: str) -> tuple[int, int]:
    episodes = 0
    frames = 0
    current_episode: int | None = None

    for idx in range(len(source)):
        item = source[idx]
        episode_index = tensor_scalar_to_int(item["episode_index"])
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            output.save_episode()
            episodes += 1
            print(f"{source_name}: saved output episode {episodes} from source episode {current_episode}")
            current_episode = episode_index

        output.add_frame(
            {
                "observation.state": tensor_to_float32_array(item["observation.state"]),
                "action": tensor_to_float32_array(item["action"]),
                "observation.images.wrist": tensor_image_to_uint8_hwc(item["observation.images.wrist"]),
                "observation.images.external": tensor_image_to_uint8_hwc(item["observation.images.external"]),
                "task": str(item["task"]),
            }
        )
        frames += 1

    if current_episode is not None:
        output.save_episode()
        episodes += 1
        print(f"{source_name}: saved output episode {episodes} from source episode {current_episode}")

    return episodes, frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge orange/apple dual-camera grasp datasets for joint ACT training.")
    parser.add_argument(
        "--source",
        action="append",
        help="Source dataset as repo_id=/path/to/root. May be repeated. If omitted, uses orange+apple defaults.",
    )
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_DIR / "datasets" / DEFAULT_OUTPUT_NAME)
    parser.add_argument("--output-repo-id", default=f"embody_car/{DEFAULT_OUTPUT_NAME}")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sources = [parse_source(raw) for raw in args.source] if args.source else list(DEFAULT_SOURCES)

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    opened = [(repo_id, root, open_dataset(repo_id, root)) for repo_id, root in sources]
    reference = opened[0][2]
    for _repo_id, root, dataset in opened[1:]:
        validate_compatible(reference, dataset, root)

    height, width = reference.features["observation.images.wrist"]["shape"][1:]
    output = create_output_dataset(
        repo_id=args.output_repo_id,
        root=output_root,
        fps=int(reference.fps),
        features=make_dual_camera_features(height=int(height), width=int(width)),
        vcodec=args.vcodec,
    )

    total_episodes = 0
    total_frames = 0
    try:
        for repo_id, root, dataset in opened:
            print(f"Merging {repo_id}: root={root}, episodes={dataset.num_episodes}, frames={dataset.num_frames}")
            episodes, frames = copy_dataset_episodes(dataset, output, source_name=repo_id)
            total_episodes += episodes
            total_frames += frames
    finally:
        output.finalize()
        output.stop_image_writer()

    merged = LeRobotDataset(args.output_repo_id, root=output_root)
    print("\nMerged dual-camera dataset ready.")
    print(f"root={output_root}")
    print(f"repo_id={args.output_repo_id}")
    print(f"episodes={merged.num_episodes}, frames={merged.num_frames}")
    print(f"copied_episodes={total_episodes}, copied_frames={total_frames}")


if __name__ == "__main__":
    main()
