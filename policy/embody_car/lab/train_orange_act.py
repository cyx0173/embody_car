"""Run ACT training for the orange wrist-grasp dataset with live logs."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from orange_grasp_config import DEFAULT_DATASET_NAME, DEFAULT_DATASET_REPO_ID, DEFAULT_JOB_NAME

DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_TRAIN_BIN = Path(shutil.which("lerobot-train") or DEFAULT_PYTHON.with_name("lerobot-train"))
INFO_PATH = Path("meta") / "info.json"


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    return (args.dataset_root or ROOT_DIR / "datasets" / args.dataset_name).expanduser().resolve()


def validate_dataset_root(dataset_root: Path) -> bool:
    info_path = dataset_root / INFO_PATH
    if info_path.exists():
        return True

    print(f"\nDataset metadata not found: {info_path}")
    print("Check --dataset-root. The dataset folder must contain meta/info.json.")
    print("Training was not started, so LeRobot will not fall back to Hugging Face.")
    return False


def cuda_available(python_bin: Path) -> tuple[bool, str]:
    code = """
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    try:
        x = torch.ones(1, device="cuda")
        print(f"cuda_tensor={x.device}")
    except Exception as exc:
        print(f"cuda_error={type(exc).__name__}: {exc}")
"""
    try:
        result = subprocess.run(
            [str(python_bin), "-c", code],
            cwd=ROOT_DIR,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        return False, f"Could not run {python_bin}: {exc}"

    output = result.stdout.strip()
    return "cuda_available=True" in output and "cuda_tensor=cuda:0" in output, output


def mps_available(python_bin: Path) -> tuple[bool, str]:
    code = """
import platform
import torch
print(f"python_arch={platform.machine()}")
print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")
try:
    x = torch.ones(1, device="mps")
    print(f"mps_tensor={x.device}")
except Exception as exc:
    print(f"mps_error={type(exc).__name__}: {exc}")
"""
    env = os.environ.copy()
    env.setdefault("SYSTEM_VERSION_COMPAT", "0")
    try:
        result = subprocess.run(
            [str(python_bin), "-c", code],
            cwd=ROOT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        return False, f"Could not run {python_bin}: {exc}"

    output = result.stdout.strip()
    return "mps_available=True" in output and "mps_tensor=mps" in output, output


def build_command(args: argparse.Namespace, device: str, output_dir: Path, dataset_root: Path) -> list[str]:
    dataset_repo_id = args.dataset_repo_id or (
        DEFAULT_DATASET_REPO_ID if args.dataset_name == DEFAULT_DATASET_NAME else f"embody_car/{args.dataset_name}"
    )

    command = [
        str(args.train_bin),
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        "--eval_freq=0",
        f"--num_workers={args.num_workers}",
        "--wandb.enable=false",
    ]
    if args.resume:
        command.append("--resume=true")
    return command


def run_live(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nLive log: {log_path}")
    print("Command:")
    print(" ".join(shlex.quote(part) for part in command))
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(shlex.quote(part) for part in command))
        log_file.write("\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
        )
        assert process.stdout is not None

        try:
            while True:
                chunk = process.stdout.read(1)
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    log_file.write(chunk)
                    log_file.flush()
                    continue
                if process.poll() is not None:
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nInterrupted. Stopping training process...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 130

        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACT on the cleaned orange wrist-grasp dataset with live progress."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT_DIR / "outputs" / "train" / f"{args.job_name}_{run_stamp}"
    output_dir = output_dir.resolve()
    dataset_root = resolve_dataset_root(args)

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT_DIR / ".cache" / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(ROOT_DIR / ".cache" / "hf_datasets"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SYSTEM_VERSION_COMPAT", "0")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    has_cuda, cuda_report = cuda_available(args.python_bin)
    has_mps, mps_report = mps_available(args.python_bin)
    if args.device == "auto":
        if has_cuda:
            device = "cuda"
        elif has_mps:
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("CUDA check:")
    print(cuda_report)
    print()
    print("MPS check:")
    print(mps_report)
    if device == "cuda" and not has_cuda:
        print("\nRequested CUDA, but this environment cannot create a CUDA tensor.")
        print("Falling back to CPU so training can still run.")
        device = "cpu"
    if device == "mps" and not has_mps:
        print("\nRequested MPS, but this environment cannot create an MPS tensor.")
        print("Falling back to CPU so training can still run.")
        device = "cpu"
    print(f"\nUsing device: {device}")

    if not validate_dataset_root(dataset_root):
        return 2

    if output_dir.exists() and not args.resume:
        print(f"\nOutput dir already exists: {output_dir}")
        print("Use --resume or choose another --output-dir.")
        return 2

    command = build_command(args, device, output_dir, dataset_root)
    log_path = output_dir.parent / f"{output_dir.name}.live.log"

    if args.dry_run:
        print("\nDry run command:")
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    return run_live(command, log_path, env)


if __name__ == "__main__":
    raise SystemExit(main())
