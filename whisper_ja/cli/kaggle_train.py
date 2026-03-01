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
from urllib.parse import quote
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


def run(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    display_cmd: str | None = None,
) -> None:
    print("$", display_cmd or " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def resolve_file_path(filename: str, base_dir: Path) -> Path | None:
    candidates = [
        base_dir / filename,
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def discover_project_root() -> Path | None:
    direct_candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path("/kaggle/src"),
        Path("/kaggle/working"),
    ]
    for candidate in direct_candidates:
        if (candidate / "whisper_ja" / "cli" / "train.py").is_file():
            return candidate

    for search_root in [Path("/kaggle/src"), Path("/kaggle/working"), Path("/kaggle/input")]:
        if not search_root.exists():
            continue
        for train_file in search_root.rglob("train.py"):
            train_parent = train_file.parent
            if train_parent.name == "cli" and train_parent.parent.name == "whisper_ja":
                return train_parent.parent.parent
    return None


def authenticated_repo_url(repo_url: str, token: str) -> str:
    if not token or not repo_url.startswith("https://"):
        return repo_url
    if "@" in repo_url.split("//", 1)[1]:
        return repo_url
    prefix, suffix = repo_url.split("https://", 1)
    del prefix
    return f"https://x-access-token:{quote(token, safe='')}@{suffix}"


def bootstrap_project_from_git() -> Path:
    repo_url = env("TRAIN_REPO_URL", "https://github.com/dungca1512/whisper-finetune-ja-train.git")
    repo_ref = env("TRAIN_REPO_REF", "main")
    token = env("GITHUB_TOKEN", "")
    target_dir = Path("/kaggle/working/whisper-finetune-ja-train")

    if target_dir.exists() and (target_dir / "whisper_ja" / "cli" / "train.py").is_file():
        print(f"✅ Reusing cloned repo: {target_dir}")
        return target_dir

    auth_url = authenticated_repo_url(repo_url, token)
    run(
        ["git", "clone", "--depth", "1", "--branch", repo_ref, auth_url, str(target_dir)],
        display_cmd=f"git clone --depth 1 --branch {repo_ref} <repo_url> {target_dir}",
    )
    return target_dir


def install_dependencies(project_root: Path) -> None:
    requirements_path = resolve_file_path("requirements.txt", project_root)
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


def resolve_train_entrypoint(project_root: Path) -> list[str]:
    train_path = resolve_file_path("train.py", project_root)
    if train_path:
        print(f"✅ Using train entrypoint: {train_path}")
        return [sys.executable, str(train_path)]

    module_train = project_root / "whisper_ja" / "cli" / "train.py"
    if module_train.is_file():
        print(f"✅ Using module train file: {module_train}")
        return [sys.executable, str(module_train)]

    raise FileNotFoundError(
        "Cannot locate training entrypoint. Expected train.py or whisper_ja/cli/train.py."
    )


def main() -> int:
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    project_root = discover_project_root()
    if project_root is None:
        print("⚠️  Project files not found in runtime snapshot. Cloning repo as fallback...")
        project_root = bootstrap_project_from_git()
    print(f"✅ Project root: {project_root}")

    install_dependencies(project_root)

    command = [
        *resolve_train_entrypoint(project_root),
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

    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = f"{project_root}:{runtime_env.get('PYTHONPATH', '')}".rstrip(":")
    run(command, cwd=str(project_root), env=runtime_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
