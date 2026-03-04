#!/usr/bin/env python3
"""
Kaggle entrypoint for LoRA training.

This script installs dependencies in Kaggle runtime, then runs train.py.
Training defaults are read from Config; only secrets are read from env/Kaggle Secrets.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _bootstrap_import_path() -> Path | None:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path("/kaggle/src"),
        Path("/kaggle/working"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "whisper_ja" / "config.py").is_file():
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return None


_BOOTSTRAP_ROOT = _bootstrap_import_path()

try:
    from whisper_ja.config import Config
except ModuleNotFoundError:
    print("❌ Failed to import whisper_ja.config")
    print(f"   cwd={Path.cwd()}")
    print(f"   script={Path(__file__).resolve()}")
    print(f"   bootstrap_root={_BOOTSTRAP_ROOT}")
    print(f"   sys.path[0]={sys.path[0] if sys.path else ''}")
    raise


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
    for key in ["HF_TOKEN", "WANDB_API_KEY"]:
        value = env(key, "")
        if value:
            runtime_env[key] = value

    runtime_env["PYTHONPATH"] = f"{project_root}:{runtime_env.get('PYTHONPATH', '')}".rstrip(":")

    available_keys = [k for k in ["HF_TOKEN", "WANDB_API_KEY"] if runtime_env.get(k)]
    print(f"✅ Runtime env keys available for train: {available_keys}")
    return runtime_env


def main() -> int:
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    project_root = discover_project_root()
    if project_root is None:
        raise FileNotFoundError(
            "Project files not found in runtime snapshot. "
            "Ensure kernel push includes whisper_ja package and train entrypoints."
        )
    print(f"✅ Project root: {project_root}")

    install_dependencies(project_root)

    config = Config()

    command = [
        *resolve_train_entrypoint(project_root),
        "--model_size", config.model_size,
        "--reazonspeech_size", config.reazonspeech_size,
        "--batch_size", str(config.batch_size),
        "--num_train_epochs", str(config.num_train_epochs),
        "--learning_rate", str(config.learning_rate),
        "--num_proc", str(config.num_proc),
        "--output_dir", config.output_dir,
        "--merged_output_dir", config.merged_output_dir,
        "--ct2_output_dir", config.ct2_output_dir,
        "--wandb_project", config.wandb_project,
        "--hub_adapter_model_id", config.hub_adapter_model_id,
        "--hub_model_id", config.hub_model_id,
        "--lora_r", str(config.lora_r),
        "--lora_alpha", str(config.lora_alpha),
        "--lora_dropout", str(config.lora_dropout),
    ]

    if config.model_name:
        command.extend(["--model_name", config.model_name])
    if not config.use_wandb:
        command.append("--no_wandb")
    if config.push_to_hub:
        command.append("--push_to_hub")
    if not config.push_merged_to_hub:
        command.append("--adapter_only_hub")
    if not config.save_merged_model:
        command.append("--no_merge_lora")
    if not config.use_lora:
        command.append("--full_finetune")
    if config.lora_target_modules:
        command.extend(["--lora_target_modules", ",".join(config.lora_target_modules)])
    if config.max_train_samples > 0:
        command.extend(["--max_train_samples", str(config.max_train_samples)])
    if config.max_eval_samples > 0:
        command.extend(["--max_eval_samples", str(config.max_eval_samples)])
    if config.wandb_tags:
        command.extend(["--wandb_tags", ",".join(config.wandb_tags)])
    if not config.run_post_train_test:
        command.append("--skip_final_test")

    runtime_env = build_runtime_env(project_root)
    run(command, cwd=str(project_root), env=runtime_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
