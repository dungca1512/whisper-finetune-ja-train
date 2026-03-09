"""
ONNX export for Whisper models.

Supports CPU and GPU inference via onnxruntime.
Uses HuggingFace Optimum for clean encoder/decoder export.

Usage:
    from whisper_ja.serving.onnx_export import export_to_onnx
    export_to_onnx(config, model_dir="./output/whisper-small-ja")
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def _check_optimum_installed() -> bool:
    try:
        import optimum  # noqa: F401
        return True
    except ImportError:
        return False


def _install_optimum() -> None:
    print("📦 Installing optimum[exporters]...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "optimum[exporters]", "onnxruntime",
    ])


def validate_onnx_export(onnx_output_dir: str, sample_audio_array=None) -> dict:
    """
    Validate exported ONNX model with a quick inference pass.
    Returns dict with validation result and latency.
    """
    import numpy as np

    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
    except ImportError as exc:
        return {"valid": False, "error": str(exc), "latency_ms": None}

    output_path = Path(onnx_output_dir)
    if not output_path.exists():
        return {"valid": False, "error": f"Directory not found: {onnx_output_dir}", "latency_ms": None}

    try:
        print(f"🔍 Validating ONNX export at: {onnx_output_dir}")
        processor = WhisperProcessor.from_pretrained(onnx_output_dir)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(onnx_output_dir)

        # Use provided sample or generate silence
        if sample_audio_array is None:
            sample_audio_array = np.zeros(16000, dtype=np.float32)  # 1s silence

        inputs = processor(
            sample_audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )

        start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            language="ja",
            task="transcribe",
            max_new_tokens=50,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"✅ ONNX validation passed (latency={latency_ms:.1f}ms)")
        print(f"   Sample transcription: '{transcription}'")

        return {
            "valid": True,
            "error": None,
            "latency_ms": round(latency_ms, 2),
            "sample_transcription": transcription,
        }

    except Exception as exc:  # pylint: disable=broad-except
        return {"valid": False, "error": str(exc), "latency_ms": None}


def export_to_onnx(config, model_dir: str | None = None) -> dict:
    """
    Export Whisper model to ONNX format using HuggingFace Optimum.

    Args:
        config: Config instance
        model_dir: Source model directory (merged HF model). Defaults to config.merged_output_dir.

    Returns:
        dict with export result metadata.
    """
    if not getattr(config, "export_onnx", True):
        print("⏭️  Skipping ONNX export (disabled in config)")
        return {"skipped": True}

    source_dir = model_dir or config.merged_output_dir
    output_dir = getattr(config, "onnx_output_dir", None) or f"./output/whisper-{config.model_size}-ja-onnx"

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source model directory not found: {source_dir}")
        return {"success": False, "error": f"Source not found: {source_dir}"}

    # Reject adapter-only dirs
    if (
        (source_path / "adapter_config.json").exists()
        and not (source_path / "model.safetensors").exists()
        and not (source_path / "pytorch_model.bin").exists()
    ):
        print("❌ Source directory contains only LoRA adapter. Merge first.")
        return {"success": False, "error": "adapter-only source"}

    if not _check_optimum_installed():
        _install_optimum()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"📦 Exporting to ONNX...")
    print(f"   Source: {source_dir}")
    print(f"   Output: {output_dir}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", str(source_dir),
        "--task", "automatic-speech-recognition-with-past",
        str(output_dir),
    ]

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - start
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(f"✅ ONNX export complete ({elapsed:.1f}s) -> {output_dir}")
    except subprocess.CalledProcessError as exc:
        print(f"❌ ONNX export failed:\n{exc.stderr[-2000:]}")
        return {"success": False, "error": exc.stderr[-500:]}

    # Validate
    validation = validate_onnx_export(output_dir)

    return {
        "success": True,
        "source_dir": str(source_dir),
        "output_dir": output_dir,
        "export_time_sec": round(elapsed, 2),
        "validation": validation,
    }
