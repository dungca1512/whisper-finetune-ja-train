#!/usr/bin/env python3
"""
Download a Whisper model from Hugging Face Hub and run ASR inference.

Examples:
  python infer_from_hf.py --repo_id dungca/whisper-small-ja --audio ./audio.wav
  python infer_from_hf.py --repo_id dungca/whisper-small-ja-ct2-int8 --audio ./audio.mp3
  HF_TOKEN=hf_xxx python infer_from_hf.py --repo_id dungca/whisper-small-ja --audio ./audio.wav --language ja
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    import torch
except ImportError:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Whisper model from HF Hub and run inference."
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face model repo id, e.g. dungca/whisper-small-ja",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file to transcribe.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language for generation (default: ja).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task (default: transcribe).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return chunk-level timestamps.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "transformers", "faster-whisper"],
        help="Inference backend. auto = detect from downloaded model files.",
    )
    parser.add_argument(
        "--compute_type",
        default="auto",
        help="Only for faster-whisper. Example: int8, int8_float16, float16, float32.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for faster-whisper decoding (default: 5).",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if torch is not None and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return "cuda"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_pipeline_device(device: str) -> int:
    resolved = resolve_device(device)
    return 0 if resolved == "cuda" else -1


def detect_backend(model_dir: Path, backend_arg: str) -> str:
    if backend_arg != "auto":
        return backend_arg

    # CTranslate2 export contains model.bin and no HF model weights.
    if (model_dir / "model.bin").exists():
        return "faster-whisper"
    return "transformers"


def run_transformers_inference(
    model_dir: str,
    audio_path: Path,
    language: str,
    task: str,
    device: int,
    timestamps: bool,
) -> None:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Install requirements first."
        ) from exc

    print(f"Loading Transformers ASR pipeline on {'GPU' if device >= 0 else 'CPU'}...")
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_dir,
        device=device,
    )

    generate_kwargs: dict[str, str] = {
        "language": language,
        "task": task,
    }

    print(f"Running inference for: {audio_path}")
    result = asr(
        str(audio_path),
        generate_kwargs=generate_kwargs,
        return_timestamps=timestamps,
    )

    print("\nTranscription:")
    print(result["text"].strip())

    if timestamps and "chunks" in result:
        print("\nTimestamps:")
        for chunk in result["chunks"]:
            start, end = chunk.get("timestamp", (None, None))
            text = chunk.get("text", "").strip()
            print(f"[{start} -> {end}] {text}")


def run_faster_whisper_inference(
    model_dir: str,
    audio_path: Path,
    language: str,
    task: str,
    device: str,
    timestamps: bool,
    compute_type: str,
    beam_size: int,
) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Install requirements first."
        ) from exc

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading faster-whisper model on {device.upper()} (compute_type={compute_type})...")
    model = WhisperModel(model_dir, device=device, compute_type=compute_type)

    print(f"Running inference for: {audio_path}")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        beam_size=beam_size,
    )

    seg_list = list(segments)
    text = "".join(seg.text for seg in seg_list).strip()

    print("\nTranscription:")
    print(text)

    print(f"\nDetected language: {getattr(info, 'language', 'unknown')}")
    print(f"Language probability: {getattr(info, 'language_probability', 'unknown')}")

    if timestamps:
        print("\nTimestamps:")
        for seg in seg_list:
            print(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text.strip()}")


def ensure_backend_compatibility(model_dir: Path, backend: str) -> None:
    has_ct2 = (model_dir / "model.bin").exists()
    has_hf_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()

    if backend == "transformers" and has_ct2 and not has_hf_weights:
        raise RuntimeError(
            "This repo looks like a CTranslate2 model (model.bin). "
            "Use --backend faster-whisper or --backend auto."
        )
    if backend == "faster-whisper" and not has_ct2:
        raise RuntimeError(
            "This repo does not look like a CTranslate2 model (missing model.bin). "
            "Use --backend transformers or --backend auto."
        )


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    token = args.token or None
    print(f"Downloading model from HF Hub: {args.repo_id}")
    local_model_dir = snapshot_download(repo_id=args.repo_id, token=token)
    print(f"Model is ready at: {local_model_dir}")

    try:
        backend = detect_backend(Path(local_model_dir), args.backend)
        ensure_backend_compatibility(Path(local_model_dir), backend)
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Backend selected: {backend}")
    try:
        if backend == "faster-whisper":
            run_faster_whisper_inference(
                model_dir=local_model_dir,
                audio_path=audio_path,
                language=args.language,
                task=args.task,
                device=device,
                timestamps=args.timestamps,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
            )
        else:
            run_transformers_inference(
                model_dir=local_model_dir,
                audio_path=audio_path,
                language=args.language,
                task=args.task,
                device=resolve_pipeline_device(args.device),
                timestamps=args.timestamps,
            )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
