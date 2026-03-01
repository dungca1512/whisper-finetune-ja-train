#!/usr/bin/env python3
"""
Kaggle entrypoint for LoRA training.

This script installs dependencies in Kaggle runtime, then runs train.py.
Training defaults can be overridden by environment variables.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    value = env(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer: {value}") from exc


def env_float(name: str, default: float) -> float:
    value = env(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float: {value}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = env(name, "1" if default else "0").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean-like value: {value}")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def resolve_file_path(filename: str) -> Path | None:
    candidates = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent.parent / filename,
        Path("/kaggle/src") / filename,
        Path("/kaggle/working") / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def install_dependencies() -> None:
    requirements_path = resolve_file_path("requirements.txt")
    if requirements_path:
        print(f"✅ Using requirements file: {requirements_path}")
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        return

    print("⚠️  requirements.txt not found. Installing fallback dependencies...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch>=2.0",
        "transformers>=4.40",
        "datasets==2.21.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
        "wandb",
        "huggingface_hub",
        "peft",
        "ctranslate2",
        "faster-whisper",
        "tensorboard",
    ])


def resolve_train_entrypoint() -> list[str]:
    train_path = resolve_file_path("train.py")
    if train_path:
        print(f"✅ Using train entrypoint: {train_path}")
        return [sys.executable, str(train_path)]

    print("⚠️  train.py not found. Falling back to module entrypoint: whisper_ja.cli.train")
    return [sys.executable, "-m", "whisper_ja.cli.train"]


def main() -> int:
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    install_dependencies()

    command = [
        *resolve_train_entrypoint(),
        "--reazonspeech_size", env("REAZONSPEECH_SIZE", "small"),
        "--batch_size", str(env_int("BATCH_SIZE", 32)),
        "--num_train_epochs", str(env_int("NUM_EPOCHS", 3)),
        "--learning_rate", str(env_float("LEARNING_RATE", 1e-5)),
        "--num_proc", str(env_int("NUM_PROC", 2)),
        "--output_dir", env("LORA_OUTPUT_DIR", "./output/whisper-tiny-ja-lora"),
        "--merged_output_dir", env("MERGED_OUTPUT_DIR", "./output/whisper-tiny-ja"),
        "--hub_adapter_model_id", env("HF_ADAPTER_REPO_ID", "dungca/whisper-tiny-ja-lora"),
        "--hub_model_id", env("HF_MERGED_REPO_ID", "dungca/whisper-tiny-ja"),
        "--lora_r", str(env_int("LORA_R", 16)),
        "--lora_alpha", str(env_int("LORA_ALPHA", 32)),
        "--lora_dropout", str(env_float("LORA_DROPOUT", 0.05)),
        "--push_to_hub",
    ]

    # Enable W&B by default; disable explicitly with ENABLE_WANDB=0.
    if not env_bool("ENABLE_WANDB", True):
        command.append("--no_wandb")

    if env("ADAPTER_ONLY_HUB", "0") == "1":
        command.append("--adapter_only_hub")

    if env("NO_MERGE_LORA", "0") == "1":
        command.append("--no_merge_lora")

    if env("FULL_FINETUNE", "0") == "1":
        command.append("--full_finetune")

    target_modules = env("LORA_TARGET_MODULES", "")
    if target_modules:
        command.extend(["--lora_target_modules", target_modules])

    wandb_tags = env("WANDB_TAGS", "")
    if wandb_tags:
        command.extend(["--wandb_tags", wandb_tags])

    if env("RUN_POST_TRAIN_TEST", "0") != "1":
        command.append("--skip_final_test")

    run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
