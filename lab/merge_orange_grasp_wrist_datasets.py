#!/usr/bin/env python3
"""Merge old wrist-only orange grasp data with wrist frames from dual-camera data."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("HF_DATASETS_CACHE", str(WORKSPACE_DIR / ".cache" / "hf_datasets"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from orange_grasp_config import JOINT_NAMES
from record_orange_dataset import make_features


DEFAULT_BASE_DATASET = WORKSPACE_DIR / "datasets" / "orange_wrist_grasp_formal_clean_v1"
DEFAULT_DUAL_DATASET = WORKSPACE_DIR / "datasets" / "orange_dual_camera_grasp_red_external_v1"
DEFAULT_OUTPUT_DATASET_NAME = "orange_wrist_grasp_formal_clean_plus_dual_v1"
DEFAULT_OUTPUT_DATASET = WORKSPACE_DIR / "datasets" / DEFAULT_OUTPUT_DATASET_NAME
DEFAULT_OUTPUT_REPO_ID = f"embody_car/{DEFAULT_OUTPUT_DATASET_NAME}"
DEFAULT_TASK = "grasp the orange with the gripper"


def tensor_image_to_numpy(image) -> np.ndarray:
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def tensor_vector_to_numpy(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.asarray(array, dtype=np.float32)


def tensor_scalar_to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def open_dataset(root: Path, repo_id: str) -> LeRobotDataset:
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Dataset is missing meta/info.json: {root}")
    return LeRobotDataset(repo_id, root=root)


def copy_wrist_episodes(
    *,
    source: LeRobotDataset,
    output: LeRobotDataset,
    task: str,
    source_name: str,
) -> tuple[int, int]:
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
            print(f"{source_name}: saved episode {episodes} from source episode {current_episode}")
            current_episode = episode_index

        output.add_frame(
            {
                "observation.state": tensor_vector_to_numpy(item["observation.state"]),
                "action": tensor_vector_to_numpy(item["action"]),
                "observation.images.wrist": tensor_image_to_numpy(item["observation.images.wrist"]),
                "task": task,
            }
        )
        frames += 1

    if current_episode is not None:
        output.save_episode()
        episodes += 1
        print(f"{source_name}: saved episode {episodes} from source episode {current_episode}")

    return episodes, frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a wrist-only orange grasp dataset by appending the wrist stream "
            "from a dual-camera grasp dataset to the original wrist-only dataset."
        )
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--dual-root", type=Path, default=DEFAULT_DUAL_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--base-repo-id", default="embody_car/orange_wrist_grasp_formal_clean_v1")
    parser.add_argument("--dual-repo-id", default="embody_car/orange_dual_camera_grasp_red_external_v1")
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {args.output_root}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(args.output_root)

    print(f"Base wrist dataset: {args.base_root}")
    print(f"Dual-camera source: {args.dual_root}")
    print(f"Output wrist-only dataset: {args.output_root}")
    print(f"Feature names: {JOINT_NAMES}")

    base = open_dataset(args.base_root, args.base_repo_id)
    dual = open_dataset(args.dual_root, args.dual_repo_id)

    output = LeRobotDataset.create(
        args.output_repo_id,
        args.fps,
        root=args.output_root,
        robot_type="custom_so101_follower",
        features=make_features(height=args.height, width=args.width),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=args.vcodec,
    )

    try:
        base_episodes, base_frames = copy_wrist_episodes(
            source=base,
            output=output,
            task=args.task,
            source_name="base",
        )
        dual_episodes, dual_frames = copy_wrist_episodes(
            source=dual,
            output=output,
            task=args.task,
            source_name="dual-wrist",
        )
    finally:
        output.finalize()
        output.stop_image_writer()

    print(
        "Merged wrist-only dataset complete: "
        f"episodes={base_episodes + dual_episodes}, "
        f"frames={base_frames + dual_frames}, root={args.output_root}"
    )


if __name__ == "__main__":
    main()
