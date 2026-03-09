"""
TensorRT export for Whisper models.

Requires:
  - Nvidia GPU (CUDA)
  - tensorrt >= 8.6
  - torch with CUDA support
  - polygraphy (optional, for engine validation)

This module is intentionally skipped on CPU-only environments.
Use for VPS/cloud GPU deployments only.

Usage:
    from whisper_ja.serving.tensorrt_export import export_to_tensorrt
    export_to_tensorrt(config, onnx_dir="./output/whisper-small-ja-onnx")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _trtexec_path() -> str | None:
    """Locate trtexec binary."""
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "trtexec",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
        # Check PATH
        result = subprocess.run(
            ["which", candidate], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def get_gpu_info() -> dict:
    """Get GPU metadata for engine compatibility tracking."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {"available": False, "error": str(exc)}


def build_trt_engine(
    onnx_path: str,
    engine_output_path: str,
    precision: str = "fp16",
    workspace_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
) -> dict:
    """
    Convert a single ONNX file to TensorRT engine via trtexec.

    Args:
        onnx_path: Path to input ONNX file.
        engine_output_path: Path to write .plan engine file.
        precision: fp32 / fp16 / int8
        workspace_gb: Max GPU workspace memory in GB.
        min/opt/max_batch: Dynamic batch shape profile.

    Returns:
        dict with build result metadata.
    """
    trtexec = _trtexec_path()
    if trtexec is None:
        return {
            "success": False,
            "error": "trtexec not found. Install TensorRT and ensure trtexec is in PATH.",
        }

    Path(engine_output_path).parent.mkdir(parents=True, exist_ok=True)

    workspace_mb = workspace_gb * 1024

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_output_path}",
        f"--workspace={workspace_mb}",
        "--verbose",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        # INT8 requires calibration data — warn user
        print("⚠️  INT8 precision requires calibration dataset for best accuracy.")
        print("   Without calibration, TRT will use implicit quantization.")

    print(f"🔧 Building TensorRT engine: {Path(onnx_path).name}")
    print(f"   Precision: {precision}")
    print(f"   Output:    {engine_output_path}")
    print(f"   Command:   {' '.join(cmd[:4])} ...")

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
        elapsed = time.perf_counter() - start
        print(f"✅ Engine built ({elapsed:.1f}s): {engine_output_path}")
        return {
            "success": True,
            "onnx_path": onnx_path,
            "engine_path": engine_output_path,
            "precision": precision,
            "build_time_sec": round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "trtexec timed out (>30min)"}
    except subprocess.CalledProcessError as exc:
        return {"success": False, "error": exc.stderr[-1000:]}


def export_to_tensorrt(config, onnx_dir: str | None = None) -> dict:
    """
    Convert Whisper ONNX files to TensorRT engines.

    Whisper has 2 ONNX components to convert:
      - encoder_model.onnx
      - decoder_model_merged.onnx (or decoder_with_past_model.onnx)

    Args:
        config: Config instance
        onnx_dir: Directory containing exported ONNX files.

    Returns:
        dict with overall export result.
    """
    if not getattr(config, "export_tensorrt", False):
        print("⏭️  Skipping TensorRT export (disabled in config)")
        return {"skipped": True}

    if not _is_cuda_available():
        print("⏭️  Skipping TensorRT export: no CUDA GPU detected")
        return {"skipped": True, "reason": "no_cuda"}

    if not _is_tensorrt_available():
        print("❌ TensorRT not installed. Run: pip install tensorrt")
        return {"success": False, "error": "tensorrt not installed"}

    source_onnx_dir = onnx_dir or getattr(
        config, "onnx_output_dir", f"./output/whisper-{config.model_size}-ja-onnx"
    )
    trt_output_dir = getattr(
        config, "tensorrt_output_dir", f"./output/whisper-{config.model_size}-ja-trt"
    )
    precision = getattr(config, "tensorrt_precision", "fp16")

    onnx_path = Path(source_onnx_dir)
    if not onnx_path.exists():
        print(f"❌ ONNX directory not found: {source_onnx_dir}")
        return {"success": False, "error": f"ONNX dir not found: {source_onnx_dir}"}

    gpu_info = get_gpu_info()
    print(f"🖥️  GPU: {gpu_info.get('name', 'unknown')} "
          f"({gpu_info.get('total_memory_gb', '?')}GB, "
          f"CC={gpu_info.get('compute_capability', '?')})")

    Path(trt_output_dir).mkdir(parents=True, exist_ok=True)

    # Write GPU metadata for reproducibility — TRT engines are GPU-specific
    meta_path = Path(trt_output_dir) / "gpu_info.json"
    meta_path.write_text(
        json.dumps(gpu_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Target ONNX files: encoder + decoder
    onnx_targets = [
        ("encoder_model.onnx", "encoder.plan"),
        ("decoder_model_merged.onnx", "decoder_merged.plan"),
        ("decoder_with_past_model.onnx", "decoder_with_past.plan"),
        ("decoder_model.onnx", "decoder.plan"),
    ]

    results = {}
    any_success = False

    for onnx_filename, engine_filename in onnx_targets:
        onnx_file = onnx_path / onnx_filename
        if not onnx_file.exists():
            continue

        engine_path = str(Path(trt_output_dir) / engine_filename)
        result = build_trt_engine(
            onnx_path=str(onnx_file),
            engine_output_path=engine_path,
            precision=precision,
        )
        results[onnx_filename] = result
        if result.get("success"):
            any_success = True

    if not results:
        print(f"❌ No ONNX files found in {source_onnx_dir}")
        return {"success": False, "error": "no onnx files found"}

    return {
        "success": any_success,
        "onnx_dir": source_onnx_dir,
        "trt_output_dir": trt_output_dir,
        "precision": precision,
        "gpu_info": gpu_info,
        "engines": results,
    }
