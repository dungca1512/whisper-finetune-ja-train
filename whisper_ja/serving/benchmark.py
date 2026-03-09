"""
Latency benchmark for Whisper serving backends.

Compares CT2 (faster-whisper), ONNX (onnxruntime), and TensorRT inference.
Outputs a JSON report suitable for quality gate latency regression checks.

Usage:
    from whisper_ja.serving.benchmark import run_benchmark
    report = run_benchmark(config, n_runs=20)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

import numpy as np


@dataclass
class BackendResult:
    backend: str
    available: bool
    skip_reason: str = ""
    latencies_ms: list[float] = field(default_factory=list)
    error: str = ""

    @property
    def mean_ms(self) -> float | None:
        return round(fmean(self.latencies_ms), 2) if self.latencies_ms else None

    @property
    def median_ms(self) -> float | None:
        return round(median(self.latencies_ms), 2) if self.latencies_ms else None

    @property
    def p95_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return round(sorted_latencies[min(idx, len(sorted_latencies) - 1)], 2)

    @property
    def stddev_ms(self) -> float | None:
        if len(self.latencies_ms) < 2:
            return None
        return round(stdev(self.latencies_ms), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "n_runs": len(self.latencies_ms),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "stddev_ms": self.stddev_ms,
            "latencies_ms": [round(x, 2) for x in self.latencies_ms],
        }


def _make_silence_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence audio array for benchmarking."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def _benchmark_ct2(model_dir: str, audio: np.ndarray, n_runs: int, language: str) -> BackendResult:
    result = BackendResult(backend="faster_whisper_ct2", available=False)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        result.skip_reason = "faster-whisper not installed"
        return result

    ct2_model_dir = Path(model_dir)
    if not ct2_model_dir.exists():
        result.skip_reason = f"CT2 model not found: {model_dir}"
        return result

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = WhisperModel(str(ct2_model_dir), device=device, compute_type=compute_type)

        # Warmup
        list(model.transcribe(audio, language=language)[0])

        result.available = True
        for _ in range(n_runs):
            start = time.perf_counter()
            list(model.transcribe(audio, language=language)[0])
            result.latencies_ms.append((time.perf_counter() - start) * 1000)

    except Exception as exc:  # pylint: disable=broad-except
        result.error = str(exc)

    return result


def _benchmark_onnx(model_dir: str, audio: np.ndarray, n_runs: int, language: str) -> BackendResult:
    result = BackendResult(backend="onnxruntime", available=False)
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
    except ImportError:
        result.skip_reason = "optimum[exporters] or onnxruntime not installed"
        return result

    onnx_path = Path(model_dir)
    if not onnx_path.exists():
        result.skip_reason = f"ONNX model not found: {model_dir}"
        return result

    try:
        processor = WhisperProcessor.from_pretrained(str(onnx_path))
        model = ORTModelForSpeechSeq2Seq.from_pretrained(str(onnx_path))
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        # Warmup
        model.generate(**inputs, language=language, task="transcribe", max_new_tokens=50)

        result.available = True
        for _ in range(n_runs):
            start = time.perf_counter()
            model.generate(**inputs, language=language, task="transcribe", max_new_tokens=50)
            result.latencies_ms.append((time.perf_counter() - start) * 1000)

    except Exception as exc:  # pylint: disable=broad-except
        result.error = str(exc)

    return result


def _benchmark_tensorrt(model_dir: str, n_runs: int) -> BackendResult:
    """TRT benchmark via trtexec --loadEngine timing (no full inference pipeline needed)."""
    result = BackendResult(backend="tensorrt", available=False)
    trt_path = Path(model_dir)

    if not trt_path.exists():
        result.skip_reason = f"TRT model not found: {model_dir}"
        return result

    engine_files = list(trt_path.glob("*.plan"))
    if not engine_files:
        result.skip_reason = "No .plan files found"
        return result

    try:
        import torch
        if not torch.cuda.is_available():
            result.skip_reason = "No CUDA GPU available for TRT benchmark"
            return result
    except ImportError:
        result.skip_reason = "torch not installed"
        return result

    # Use encoder engine for latency proxy
    encoder_engine = trt_path / "encoder.plan"
    if not encoder_engine.exists():
        encoder_engine = engine_files[0]

    import subprocess
    import shutil
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        result.skip_reason = "trtexec not found in PATH"
        return result

    try:
        cmd = [
            trtexec,
            f"--loadEngine={encoder_engine}",
            f"--iterations={n_runs}",
            "--percentile=95",
            "--noDataTransfers",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = proc.stdout + proc.stderr

        # Parse mean latency from trtexec output
        for line in output.splitlines():
            if "mean" in line.lower() and "ms" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if "ms" in part.lower() and i > 0:
                        try:
                            latency = float(parts[i - 1].replace("ms", "").strip())
                            result.latencies_ms = [latency]
                            result.available = True
                            break
                        except ValueError:
                            continue
                if result.available:
                    break

        if not result.available:
            result.skip_reason = "Could not parse trtexec output"

    except Exception as exc:  # pylint: disable=broad-except
        result.error = str(exc)

    return result


def run_benchmark(
    config,
    n_runs: int = 20,
    audio_duration_sec: float = 3.0,
    ct2_model_dir: str | None = None,
    onnx_model_dir: str | None = None,
    trt_model_dir: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Run latency benchmark across all available backends.

    Args:
        config: Config instance
        n_runs: Number of inference runs per backend (after 1 warmup)
        audio_duration_sec: Duration of synthetic test audio
        ct2_model_dir: Path to CT2 model directory
        onnx_model_dir: Path to ONNX model directory
        trt_model_dir: Path to TensorRT engine directory
        output_path: Where to write JSON report

    Returns:
        dict with benchmark results per backend.
    """
    resolved_ct2_dir = ct2_model_dir or getattr(
        config, "ct2_output_dir", f"./output/whisper-{config.model_size}-ja-ct2"
    )
    resolved_onnx_dir = onnx_model_dir or getattr(
        config, "onnx_output_dir", f"./output/whisper-{config.model_size}-ja-onnx"
    )
    resolved_trt_dir = trt_model_dir or getattr(
        config, "tensorrt_output_dir", f"./output/whisper-{config.model_size}-ja-trt"
    )
    resolved_output = output_path or "reports/benchmark_report.json"

    audio = _make_silence_audio(duration_sec=audio_duration_sec)
    language = getattr(config, "language", "ja")

    print(f"🏎️  Running latency benchmark ({n_runs} runs per backend)")
    print(f"   Audio: {audio_duration_sec}s silence @ 16kHz")

    results: dict[str, BackendResult] = {}

    print("\n[1/3] CT2 (faster-whisper)...")
    results["faster_whisper_ct2"] = _benchmark_ct2(resolved_ct2_dir, audio, n_runs, language)

    print("[2/3] ONNX Runtime...")
    results["onnxruntime"] = _benchmark_onnx(resolved_onnx_dir, audio, n_runs, language)

    print("[3/3] TensorRT...")
    results["tensorrt"] = _benchmark_tensorrt(resolved_trt_dir, n_runs)

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Backend':<25} {'Mean ms':>10} {'P95 ms':>10} {'Available':>10}")
    print(f"{'='*55}")
    for name, res in results.items():
        if res.available:
            print(f"{name:<25} {str(res.mean_ms):>10} {str(res.p95_ms):>10} {'✅':>10}")
        else:
            reason = res.skip_reason or res.error or "n/a"
            print(f"{name:<25} {'—':>10} {'—':>10}   ⏭ {reason[:20]}")
    print(f"{'='*55}\n")

    report = {
        "model_size": config.model_size,
        "n_runs": n_runs,
        "audio_duration_sec": audio_duration_sec,
        "results": {name: res.to_dict() for name, res in results.items()},
    }

    out_path = Path(resolved_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📄 Benchmark report written: {out_path}")

    return report


def check_latency_regression(
    report: dict[str, Any],
    baseline_report_path: str | None = None,
    backend: str = "onnxruntime",
    max_regression_pct: float = 20.0,
) -> dict[str, Any]:
    """
    Check if latency regressed beyond threshold vs baseline.

    Args:
        report: Current benchmark report dict
        baseline_report_path: Path to previous benchmark JSON report
        backend: Which backend to compare ("onnxruntime", "faster_whisper_ct2", "tensorrt")
        max_regression_pct: Max allowed latency increase percentage (default 20%)

    Returns:
        dict with decision ("pass" or "fail") and details.
    """
    current_result = report.get("results", {}).get(backend, {})
    current_mean = current_result.get("mean_ms")

    if current_mean is None:
        return {
            "decision": "skip",
            "reason": f"Backend '{backend}' not available in current report",
            "backend": backend,
        }

    if baseline_report_path is None or not Path(baseline_report_path).exists():
        return {
            "decision": "skip",
            "reason": "No baseline report provided — skipping regression check",
            "backend": backend,
            "current_mean_ms": current_mean,
        }

    baseline_data = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    baseline_mean = baseline_data.get("results", {}).get(backend, {}).get("mean_ms")

    if baseline_mean is None:
        return {
            "decision": "skip",
            "reason": f"Backend '{backend}' not in baseline report",
            "backend": backend,
        }

    regression_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
    passed = regression_pct <= max_regression_pct

    return {
        "decision": "pass" if passed else "fail",
        "backend": backend,
        "current_mean_ms": current_mean,
        "baseline_mean_ms": baseline_mean,
        "regression_pct": round(regression_pct, 2),
        "max_regression_pct": max_regression_pct,
    }
