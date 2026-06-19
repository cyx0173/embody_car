#!/usr/bin/env python3
"""Small safe wrapper around LeRobot training.

This keeps LeRobot's normal training path, adds TensorBoard scalar logging, and
turns off DDP find_unused_parameters to avoid the repeated reducer warning when
all ACT parameters are used.
"""

from __future__ import annotations

import ast
import bisect
import copy
import logging
import os
import re
import sys
import warnings
from itertools import accumulate
from pathlib import Path

# Must be set before heavy torch imports.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*torchvision.io.video.*")

import torch
from torch.utils.data import Dataset

import lerobot.scripts.lerobot_train as lerobot_train


def _rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _parse_maybe_list(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    if "," in text:
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]
    return value


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_num_frames(dataset) -> int:
    if hasattr(dataset, "num_frames"):
        return _safe_int(getattr(dataset, "num_frames"), len(dataset))
    return len(dataset)


def _get_num_episodes(dataset) -> int:
    for name in ("num_episodes", "n_episodes"):
        if hasattr(dataset, name):
            return _safe_int(getattr(dataset, name), 0)

    meta = getattr(dataset, "meta", None)
    if meta is not None:
        for name in ("num_episodes", "n_episodes", "total_episodes"):
            if hasattr(meta, name):
                return _safe_int(getattr(meta, name), 0)

    info = getattr(dataset, "info", None)
    if isinstance(info, dict):
        for name in ("num_episodes", "n_episodes", "total_episodes"):
            if name in info:
                return _safe_int(info[name], 0)

    return 0


def _get_cli_value(name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def _parse_big_number(text: str) -> float:
    value = str(text).strip()
    if value.endswith("K"):
        return float(value[:-1]) * 1_000
    if value.endswith("M"):
        return float(value[:-1]) * 1_000_000
    if value.endswith("B"):
        return float(value[:-1]) * 1_000_000_000
    return float(value)


def _patch_ddp_find_unused_parameters() -> None:
    """Force find_unused_parameters=False unless explicitly opted out.

    PyTorch emits the warning from C++ reducer code, so Python warning filters
    do not reliably hide it. Patching DDP before LeRobot constructs the model
    removes the cause instead.
    """

    if os.environ.get("LEROBOT_KEEP_FIND_UNUSED_PARAMETERS", "0") == "1":
        return

    import torch.nn.parallel as parallel
    import torch.nn.parallel.distributed as distributed

    original_ddp = distributed.DistributedDataParallel

    class QuietDistributedDataParallel(original_ddp):
        def __init__(self, *args, **kwargs):
            kwargs["find_unused_parameters"] = False
            super().__init__(*args, **kwargs)

    parallel.DistributedDataParallel = QuietDistributedDataParallel
    distributed.DistributedDataParallel = QuietDistributedDataParallel


def _install_tensorboard_logger() -> None:
    if not _rank0():
        return

    job_name = _get_cli_value("--job_name", "live_curve") or "live_curve"
    tb_dir = Path("./outputs/tensorboard").expanduser().resolve() / job_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[TENSORBOARD] unavailable: {exc}", flush=True)
        return

    writer = SummaryWriter(log_dir=str(tb_dir))

    class TensorBoardMetricHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.INFO)
            self.last_step = None

        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "step:" not in message or "loss:" not in message:
                return

            pairs = dict(re.findall(r"([A-Za-z_]+):([^\s]+)", message))
            if "step" not in pairs or "loss" not in pairs:
                return

            try:
                step = int(_parse_big_number(pairs["step"]))
                if self.last_step == step:
                    return
                self.last_step = step

                writer.add_scalar("train/loss", float(pairs.get("loss", "nan")), step)
                writer.add_scalar("train/grad_norm", float(pairs.get("grdn", "nan")), step)
                writer.add_scalar("train/lr", float(pairs.get("lr", "nan")), step)
                writer.add_scalar("time/update_s", float(pairs.get("updt_s", "nan")), step)
                writer.add_scalar("time/data_s", float(pairs.get("data_s", "nan")), step)
                writer.add_scalar("progress/epoch", float(pairs.get("epch", "nan")), step)
                writer.add_scalar("progress/samples", _parse_big_number(pairs.get("smpl", "0")), step)
                writer.flush()
            except Exception:
                return

    logging.getLogger().addHandler(TensorBoardMetricHandler())
    print(f"[TENSORBOARD] live curves: {tb_dir}", flush=True)
    print(f"[TENSORBOARD] tensorboard --logdir {tb_dir} --host 0.0.0.0 --port 6006", flush=True)


class MultiLeRobotDatasetCompat(Dataset):
    """Expose multiple LeRobot datasets as one dataset without merging files."""

    def __init__(self, datasets, repo_ids, roots):
        if not datasets:
            raise ValueError("datasets cannot be empty")

        self.datasets = datasets
        self.repo_ids = repo_ids
        self.roots = roots
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = list(accumulate(self.lengths))
        self.num_frames = sum(_get_num_frames(dataset) for dataset in datasets)
        self.num_episodes = sum(_get_num_episodes(dataset) for dataset in datasets)

        first = datasets[0]
        self.repo_id = "+".join(repo_ids)
        self.root = roots

        for name in (
            "meta",
            "info",
            "stats",
            "features",
            "fps",
            "camera_keys",
            "video",
            "video_backend",
            "image_transforms",
            "delta_timestamps",
            "tolerance_s",
            "episodes",
            "tasks",
            "hf_dataset",
            "revision",
            "streaming",
        ):
            if hasattr(first, name):
                setattr(self, name, getattr(first, name))

        if _rank0():
            print("[DATASET] loaded:", flush=True)
            for index, dataset in enumerate(datasets):
                print(
                    f"  [{index}] repo_id={repo_ids[index]} root={roots[index]} "
                    f"frames={_get_num_frames(dataset)} episodes={_get_num_episodes(dataset)}",
                    flush=True,
                )
            print(f"  total frames={self.num_frames} episodes={self.num_episodes}", flush=True)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getattr__(self, name):
        datasets = self.__dict__.get("datasets", None)
        if datasets and hasattr(datasets[0], name):
            return getattr(datasets[0], name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def _find_dataset(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return dataset_index, index - previous_size

    def __getitem__(self, index):
        dataset_index, local_index = self._find_dataset(index)
        item = self.datasets[dataset_index][local_index]
        if isinstance(item, dict):
            item = dict(item)
            item["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        return item


_original_make_dataset = lerobot_train.make_dataset


def _make_single_dataset(cfg, repo_id, root):
    local_cfg = copy.deepcopy(cfg)
    local_cfg.dataset.repo_id = repo_id
    local_cfg.dataset.root = root
    return _original_make_dataset(local_cfg)


def _make_dataset_compat(cfg):
    repo_ids = _parse_maybe_list(cfg.dataset.repo_id)
    roots = _parse_maybe_list(cfg.dataset.root)

    if not isinstance(repo_ids, list):
        cfg.dataset.repo_id = repo_ids
        cfg.dataset.root = roots
        return _original_make_dataset(cfg)

    if roots is None:
        roots = [None] * len(repo_ids)
    elif not isinstance(roots, list):
        roots = [roots] * len(repo_ids)

    if len(repo_ids) != len(roots):
        raise ValueError(f"dataset.repo_id and dataset.root length mismatch: {len(repo_ids)} vs {len(roots)}")

    cfg.dataset.repo_id = repo_ids[0]
    cfg.dataset.root = roots[0]
    datasets = [_original_make_dataset(cfg)]
    for repo_id, root in zip(repo_ids[1:], roots[1:]):
        datasets.append(_make_single_dataset(cfg, repo_id, root))

    return MultiLeRobotDatasetCompat(datasets=datasets, repo_ids=repo_ids, roots=roots)


def main() -> None:
    _patch_ddp_find_unused_parameters()
    lerobot_train.make_dataset = _make_dataset_compat
    _install_tensorboard_logger()
    lerobot_train.main()


if __name__ == "__main__":
    main()
