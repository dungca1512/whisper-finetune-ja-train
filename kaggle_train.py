#!/usr/bin/env python3
"""CLI entrypoint wrapper for Kaggle runtime."""

from whisper_ja.cli.kaggle_train import main


if __name__ == "__main__":
    raise SystemExit(main())
