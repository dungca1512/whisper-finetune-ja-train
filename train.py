#!/usr/bin/env python3
"""CLI entrypoint wrapper for training."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _bootstrap_import_path()
    from whisper_ja.cli.train import main as _main

    _main()


if __name__ == "__main__":
    main()
