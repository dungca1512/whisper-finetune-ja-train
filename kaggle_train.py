#!/usr/bin/env python3
"""Kaggle runtime bootstrap for training entrypoint.

`kaggle kernels push` uploads the script body from `code_file`, not the full
project tree. This wrapper therefore tries to locate project files first and
falls back to downloading a source snapshot before importing `whisper_ja`.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


EMBEDDED_RUNTIME_SECRETS: dict[str, str] = {}


def _seed_runtime_env() -> None:
    for key, value in EMBEDDED_RUNTIME_SECRETS.items():
        key = str(key).strip()
        val = str(value).strip()
        if key and val and not os.environ.get(key):
            os.environ[key] = val


def _find_repo_root() -> Path | None:
    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        script_path.parent,
        script_path.parent.parent,
        Path("/kaggle/working"),
        Path("/kaggle/src"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "whisper_ja" / "config.py").is_file():
            return candidate

    for search_root in [Path("/kaggle/working"), Path("/kaggle/input"), Path("/kaggle/src")]:
        if not search_root.exists():
            continue
        for config_path in search_root.rglob("whisper_ja/config.py"):
            return config_path.parent.parent
    return None


def _download_repo_snapshot() -> Path | None:
    repo = os.environ.get("GITHUB_SOURCE_REPO", "").strip()
    ref = os.environ.get("GITHUB_SOURCE_REF", "").strip() or "main"
    token = os.environ.get("GITHUB_READ_TOKEN", "").strip()

    if not repo:
        print("Missing GITHUB_SOURCE_REPO. Skipping source snapshot download.")
        return None

    url = f"https://api.github.com/repos/{repo}/tarball/{ref}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "whisper-ja-kaggle-bootstrap",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    work_root = Path("/kaggle/working/repo_snapshot")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    archive_path = work_root / "source.tar.gz"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as out:
            out.write(response.read())
    except urllib.error.HTTPError as exc:
        print(f"Source snapshot download failed with HTTP {exc.code}.")
        if exc.code in {401, 403, 404}:
            print("Set GITHUB_READ_TOKEN with read access to this repository.")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Source snapshot download failed: {exc}")
        return None

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(work_root)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Source snapshot extract failed: {exc}")
        return None
    finally:
        archive_path.unlink(missing_ok=True)

    extracted_dirs = sorted([path for path in work_root.iterdir() if path.is_dir()])
    if not extracted_dirs:
        print("Source snapshot extract did not create a directory.")
        return None

    repo_root = extracted_dirs[0]
    if not (repo_root / "whisper_ja" / "config.py").is_file():
        print(f"Snapshot missing whisper_ja package: {repo_root}")
        return None

    print(f"Downloaded source snapshot: {repo_root}")
    return repo_root


def _bootstrap_import_path() -> Path | None:
    _seed_runtime_env()
    repo_root = _find_repo_root()
    if repo_root is None:
        repo_root = _download_repo_snapshot()

    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> int:
    repo_root = _bootstrap_import_path()
    try:
        from whisper_ja.cli.kaggle_train import main as _main
    except ModuleNotFoundError as exc:
        print("Failed to import whisper_ja package in Kaggle runtime")
        print(f"   cwd={os.getcwd()}")
        print(f"   repo_root={repo_root}")
        print(f"   sys.path[0]={sys.path[0] if sys.path else ''}")
        try:
            print(f"   repo_root files={sorted(p.name for p in repo_root.iterdir())}")
        except Exception:  # pylint: disable=broad-except
            pass
        for probe in [Path("/kaggle/src"), Path("/kaggle/working"), Path.cwd()]:
            try:
                if probe.exists():
                    print(f"   {probe} files={sorted(p.name for p in probe.iterdir())[:20]}")
            except Exception:  # pylint: disable=broad-except
                pass
        raise
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
