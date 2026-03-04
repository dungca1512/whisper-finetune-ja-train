#!/usr/bin/env python3
"""
Upload finetuned Whisper models to Hugging Face Hub.

Usage:
  export HF_TOKEN=hf_xxx
  python upload_models_to_hf.py

Optional:
  python upload_models_to_hf.py --private
  python upload_models_to_hf.py --model_size small
  python upload_models_to_hf.py --hf_repo_id dungca/whisper-small-ja
  python upload_models_to_hf.py --ct2_repo_id dungca/whisper-small-ja-ct2-int8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Whisper finetuned models (HF + CT2) to Hugging Face Hub."
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--model_size",
        default=os.environ.get("MODEL_SIZE", "tiny"),
        help="Whisper size tag used for auto defaults (tiny/base/small/medium/large-v3/turbo).",
    )
    parser.add_argument(
        "--repo_owner",
        default=os.environ.get("HF_REPO_OWNER", "dungca"),
        help="HF namespace used for auto repo-id defaults.",
    )
    parser.add_argument(
        "--hf_dir",
        default=None,
        help="Local Transformers model directory. Default: ./output/whisper-<model_size>-ja",
    )
    parser.add_argument(
        "--ct2_dir",
        default=None,
        help="Local CTranslate2 model directory. Default: ./output/whisper-<model_size>-ja-ct2",
    )
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="Target HF repo for Transformers model. Default: <repo_owner>/whisper-<model_size>-ja",
    )
    parser.add_argument(
        "--ct2_repo_id",
        default=None,
        help="Target HF repo for CTranslate2 model. Default: <repo_owner>/whisper-<model_size>-ja-ct2-int8",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repositories.",
    )
    parser.add_argument(
        "--skip_hf",
        action="store_true",
        help="Skip uploading Transformers model.",
    )
    parser.add_argument(
        "--skip_ct2",
        action="store_true",
        help="Skip uploading CTranslate2 model.",
    )
    return parser.parse_args()


def require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def upload_model_folder(
    api: HfApi,
    folder_path: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    ignore_patterns: list[str] | None = None,
) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        ignore_patterns=ignore_patterns or [],
        commit_message=commit_message,
    )
    print(f"Uploaded: {folder_path} -> https://huggingface.co/{repo_id}")


def main() -> int:
    args = parse_args()

    if not args.token:
        print("Error: missing token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        return 1

    model_size = args.model_size.strip()
    if not model_size:
        print("Error: --model_size cannot be empty.", file=sys.stderr)
        return 1
    model_tag = f"whisper-{model_size}-ja"
    hf_repo_id = args.hf_repo_id or f"{args.repo_owner}/{model_tag}"
    ct2_repo_id = args.ct2_repo_id or f"{args.repo_owner}/{model_tag}-ct2-int8"

    hf_dir = Path(args.hf_dir or f"./output/{model_tag}")
    ct2_dir = Path(args.ct2_dir or f"./output/{model_tag}-ct2")

    if args.skip_hf and args.skip_ct2:
        print("Nothing to do: both --skip_hf and --skip_ct2 were set.")
        return 0

    try:
        if not args.skip_hf:
            require_dir(hf_dir, "Transformers model")
        if not args.skip_ct2:
            require_dir(ct2_dir, "CT2 model")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)

    try:
        who = api.whoami()
        user = who.get("name", "unknown")
        print(f"Authenticated as: {user}")

        if not args.skip_hf:
            upload_model_folder(
                api=api,
                folder_path=hf_dir,
                repo_id=hf_repo_id,
                private=args.private,
                commit_message=f"Upload finetuned {model_tag} (Transformers format)",
                ignore_patterns=[
                    "checkpoint-*",
                    "logs/*",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                ],
            )

        if not args.skip_ct2:
            upload_model_folder(
                api=api,
                folder_path=ct2_dir,
                repo_id=ct2_repo_id,
                private=args.private,
                commit_message=f"Upload finetuned {model_tag} (CTranslate2 INT8 format)",
            )

    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        print(f"Hugging Face API error (status={status}): {exc}", file=sys.stderr)
        if status == 401:
            print("Hint: token is invalid or expired.", file=sys.stderr)
        elif status == 403:
            print(
                "Hint: token likely has read-only scope. Create a token with Model write permission.",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
