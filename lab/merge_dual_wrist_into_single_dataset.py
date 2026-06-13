from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from record_orange_dataset import create_or_resume_dataset, make_features


DEFAULT_BASE_NAME = "orange_to_bowl_wrist_place_v1"
DEFAULT_DUAL_NAME = "orange_to_bowl_dual_camera_place_v1"
DEFAULT_OUTPUT_NAME = "orange_to_bowl_wrist_place_merged_v1"


def tensor_image_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(image.shape)}")
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


def episode_ranges(dataset: LeRobotDataset) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for episode in dataset.meta.episodes:
        ranges.append(
            (
                int(episode["dataset_from_index"]),
                int(episode["dataset_to_index"]),
            )
        )
    return ranges


def add_episode_from_source(
    *,
    output: LeRobotDataset,
    source: LeRobotDataset,
    start: int,
    end: int,
    image_key: str,
) -> int:
    frame_count = 0
    for index in range(start, end):
        item = source[index]
        output.add_frame(
            {
                "observation.state": tensor_to_float32_array(item["observation.state"]),
                "action": tensor_to_float32_array(item["action"]),
                "observation.images.wrist": tensor_image_to_uint8_hwc(item[image_key]),
                "task": str(item["task"]),
            }
        )
        frame_count += 1
    output.save_episode()
    return frame_count


def merge_datasets(args: argparse.Namespace) -> None:
    base_root = Path(args.base_root)
    dual_root = Path(args.dual_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}\n"
                "Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)

    base = LeRobotDataset(args.base_repo_id, root=base_root)
    dual = LeRobotDataset(args.dual_repo_id, root=dual_root)

    if base.fps != dual.fps:
        raise ValueError(f"FPS mismatch: base={base.fps}, dual={dual.fps}")

    height, width = base.features["observation.images.wrist"]["shape"][1:]
    features = make_features(height=int(height), width=int(width))
    output = create_or_resume_dataset(
        repo_id=args.output_repo_id,
        root=output_root,
        fps=int(base.fps),
        features=features,
        resume=False,
        vcodec=args.vcodec,
    )

    total_frames = 0
    try:
        print(f"Base single-wrist dataset: {base_root} episodes={base.num_episodes} frames={base.num_frames}")
        for episode_idx, (start, end) in enumerate(episode_ranges(base)):
            frames = add_episode_from_source(
                output=output,
                source=base,
                start=start,
                end=end,
                image_key="observation.images.wrist",
            )
            total_frames += frames
            print(f"copied base episode {episode_idx}: {frames} frames")

        print(f"Dual-camera dataset: {dual_root} episodes={dual.num_episodes} frames={dual.num_frames}")
        for episode_idx, (start, end) in enumerate(episode_ranges(dual)):
            frames = add_episode_from_source(
                output=output,
                source=dual,
                start=start,
                end=end,
                image_key="observation.images.wrist",
            )
            total_frames += frames
            print(f"added dual wrist episode {episode_idx}: {frames} frames")
    finally:
        output.finalize()
        output.stop_image_writer()

    merged = LeRobotDataset(args.output_repo_id, root=output_root)
    print("\nMerged single-wrist dataset ready.")
    print(f"root={output_root}")
    print(f"episodes={merged.num_episodes}, frames={merged.num_frames}, expected_frames={total_frames}")
    print(f"features={sorted(merged.features.keys())}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a single-wrist LeRobot dataset with the wrist stream from a dual-camera dataset."
    )
    parser.add_argument("--base-root", default=str(WORKSPACE_DIR / "datasets" / DEFAULT_BASE_NAME))
    parser.add_argument("--dual-root", default=str(WORKSPACE_DIR / "datasets" / DEFAULT_DUAL_NAME))
    parser.add_argument("--output-root", default=str(WORKSPACE_DIR / "datasets" / DEFAULT_OUTPUT_NAME))
    parser.add_argument("--base-repo-id", default=f"embody_car/{DEFAULT_BASE_NAME}")
    parser.add_argument("--dual-repo-id", default=f"embody_car/{DEFAULT_DUAL_NAME}")
    parser.add_argument("--output-repo-id", default=f"embody_car/{DEFAULT_OUTPUT_NAME}")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    merge_datasets(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
