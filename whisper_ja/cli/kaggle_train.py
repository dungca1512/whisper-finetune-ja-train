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
from subprocess import CalledProcessError
from urllib.parse import quote
from pathlib import Path


_KAGGLE_SECRETS_CLIENT = None
_KAGGLE_SECRETS_CACHE: dict[str, str] = {}
_RUNTIME_SECRET_BUNDLE: dict[str, str] | None = None
# This placeholder is patched by GitHub Actions before kaggle kernels push.
EMBEDDED_RUNTIME_SECRETS: dict[str, str] = {}


def _load_runtime_secret_bundle() -> dict[str, str]:
    global _RUNTIME_SECRET_BUNDLE
    if _RUNTIME_SECRET_BUNDLE is not None:
        return _RUNTIME_SECRET_BUNDLE

    candidates = [
        Path.cwd() / "runtime_secrets.json",
        Path(__file__).resolve().parent / "runtime_secrets.json",
        Path(__file__).resolve().parent.parent / "runtime_secrets.json",
        Path(__file__).resolve().parent.parent.parent / "runtime_secrets.json",
        Path("/kaggle/src/runtime_secrets.json"),
        Path("/kaggle/working/runtime_secrets.json"),
    ]

    bundle: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                bundle = {
                    str(k): str(v).strip()
                    for k, v in raw.items()
                    if isinstance(v, (str, int, float)) and str(v).strip()
                }
                print(f"✅ Loaded runtime secret bundle from {path} (keys={sorted(bundle.keys())})")
                break
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️  Failed to read runtime secret bundle at {path}: {exc}")

    _RUNTIME_SECRET_BUNDLE = bundle
    return _RUNTIME_SECRET_BUNDLE


def _get_kaggle_secret(name: str) -> str:
    global _KAGGLE_SECRETS_CLIENT

    if name in _KAGGLE_SECRETS_CACHE:
        return _KAGGLE_SECRETS_CACHE[name]

    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        _KAGGLE_SECRETS_CACHE[name] = ""
        return ""

    try:
        if _KAGGLE_SECRETS_CLIENT is None:
            _KAGGLE_SECRETS_CLIENT = UserSecretsClient()
        value = _KAGGLE_SECRETS_CLIENT.get_secret(name) or ""
        value = value.strip()
        if value:
            print(f"✅ Loaded Kaggle Secret: {name}")
        _KAGGLE_SECRETS_CACHE[name] = value
        return value
    except Exception:
        _KAGGLE_SECRETS_CACHE[name] = ""
        return ""


def env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value

    embedded_value = EMBEDDED_RUNTIME_SECRETS.get(name, "").strip()
    if embedded_value:
        os.environ[name] = embedded_value
        return embedded_value

    bundle = _load_runtime_secret_bundle()
    bundle_value = bundle.get(name, "").strip()
    if bundle_value:
        os.environ[name] = bundle_value
        return bundle_value

    secret_value = _get_kaggle_secret(name)
    if secret_value:
        os.environ[name] = secret_value
        return secret_value

    return default.strip()


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

    if target_dir.exists():
        run(["rm", "-rf", str(target_dir)])

    auth_url = authenticated_repo_url(repo_url, token)
    try:
        run(
            ["git", "clone", "--depth", "1", "--branch", repo_ref, auth_url, str(target_dir)],
            display_cmd=f"git clone --depth 1 --branch {repo_ref} <repo_url> {target_dir}",
        )
    except CalledProcessError as exc:
        if "github.com" in repo_url and not token:
            raise RuntimeError(
                "Cannot clone TRAIN_REPO_URL from GitHub without credentials. "
                "If the repo is private, add Kaggle Secret GITHUB_TOKEN "
                "(classic PAT with repo read scope or fine-grained token with contents read)."
            ) from exc
        raise

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
        return [sys.executable, "-u", str(train_path)]

    module_train = project_root / "whisper_ja" / "cli" / "train.py"
    if module_train.is_file():
        print(f"✅ Using module train file: {module_train}")
        return [sys.executable, "-u", str(module_train)]

    raise FileNotFoundError(
        "Cannot locate training entrypoint. Expected train.py or whisper_ja/cli/train.py."
    )


def build_runtime_env(project_root: Path) -> dict[str, str]:
    runtime_env = os.environ.copy()

    # Force-load secrets from embedded bundle / runtime bundle / Kaggle Secrets
    # so child processes (train.py) always receive them.
    for key in ["HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN", "TRAIN_REPO_URL", "TRAIN_REPO_REF"]:
        value = env(key, "")
        if value:
            runtime_env[key] = value

    runtime_env["PYTHONPATH"] = f"{project_root}:{runtime_env.get('PYTHONPATH', '')}".rstrip(":")

    available_keys = [k for k in ["HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN"] if runtime_env.get(k)]
    print(f"✅ Runtime env keys available for train: {available_keys}")
    return runtime_env


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

    model_size = env("MODEL_SIZE", "tiny")
    model_tag = f"whisper-{model_size}-ja"
    repo_owner = env("HF_REPO_OWNER", "dungca")
    model_name = env("MODEL_NAME", "")

    command = [
        *resolve_train_entrypoint(project_root),
        "--model_size", model_size,
        "--reazonspeech_size", env("REAZONSPEECH_SIZE", "small"),
        "--batch_size", str(env_int("BATCH_SIZE", 32)),
        "--num_train_epochs", str(env_int("NUM_EPOCHS", 3)),
        "--learning_rate", str(env_float("LEARNING_RATE", 1e-5)),
        "--num_proc", str(env_int("NUM_PROC", 1)),
        "--output_dir", env("LORA_OUTPUT_DIR", f"./output/{model_tag}-lora"),
        "--merged_output_dir", env("MERGED_OUTPUT_DIR", f"./output/{model_tag}"),
        "--ct2_output_dir", env("CT2_OUTPUT_DIR", f"./output/{model_tag}-ct2"),
        "--wandb_project", env("WANDB_PROJECT", model_tag),
        "--hub_adapter_model_id", env("HF_ADAPTER_REPO_ID", f"{repo_owner}/{model_tag}-lora"),
        "--hub_model_id", env("HF_MERGED_REPO_ID", f"{repo_owner}/{model_tag}"),
        "--lora_r", str(env_int("LORA_R", 16)),
        "--lora_alpha", str(env_int("LORA_ALPHA", 32)),
        "--lora_dropout", str(env_float("LORA_DROPOUT", 0.05)),
        "--push_to_hub",
    ]

    if model_name:
        command.extend(["--model_name", model_name])

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

    max_train_samples = env_int("MAX_TRAIN_SAMPLES", 0)
    if max_train_samples > 0:
        command.extend(["--max_train_samples", str(max_train_samples)])

    wandb_tags = env("WANDB_TAGS", "")
    if wandb_tags:
        command.extend(["--wandb_tags", wandb_tags])

    if env("RUN_POST_TRAIN_TEST", "0") != "1":
        command.append("--skip_final_test")

    runtime_env = build_runtime_env(project_root)
    run(command, cwd=str(project_root), env=runtime_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
