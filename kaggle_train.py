#!/usr/bin/env python3
"""CLI entrypoint wrapper for Kaggle runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_import_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> int:
    repo_root = _bootstrap_import_path()
    try:
        from whisper_ja.cli.kaggle_train import main as _main
    except ModuleNotFoundError as exc:
        print("❌ Failed to import whisper_ja package in Kaggle runtime")
        print(f"   cwd={os.getcwd()}")
        print(f"   repo_root={repo_root}")
        print(f"   sys.path[0]={sys.path[0] if sys.path else ''}")
        try:
            print(f"   repo_root files={sorted(p.name for p in repo_root.iterdir())}")
        except Exception:  # pylint: disable=broad-except
            pass
        raise
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
