#!/usr/bin/env python3
"""
CLI entrypoint for the full serving export + benchmark pipeline.

Actions:
  --export_onnx        Export merged HF model to ONNX
  --export_tensorrt    Convert ONNX to TensorRT engines (GPU only)
  --gen_triton         Generate Triton model repository
  --benchmark          Run latency benchmark across all backends
  --all                Run all steps sequentially

Usage:
  python serving.py --export_onnx
  python serving.py --benchmark --n_runs 30
  python serving.py --all
  python serving.py --gen_triton --triton_backend onnxruntime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_import_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper serving pipeline: ONNX / TRT / Triton / Benchmark")

    # Actions
    parser.add_argument("--export_onnx", action="store_true", help="Export merged model to ONNX")
    parser.add_argument("--export_tensorrt", action="store_true", help="Convert ONNX to TensorRT (GPU only)")
    parser.add_argument("--gen_triton", action="store_true", help="Generate Triton model repository")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--all", action="store_true", help="Run all steps")

    # Config overrides
    parser.add_argument("--model_size", type=str, help="Whisper model size (tiny/small/...)")
    parser.add_argument("--model_dir", type=str, help="Source merged model directory")
    parser.add_argument("--onnx_output_dir", type=str, help="ONNX output directory")
    parser.add_argument("--tensorrt_output_dir", type=str, help="TRT engine output directory")
    parser.add_argument("--triton_output_dir", type=str, help="Triton repo output directory")
    parser.add_argument("--ct2_output_dir", type=str, help="CT2 model directory for benchmark")

    # TRT options
    parser.add_argument(
        "--tensorrt_precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TRT engine precision (default: fp16)",
    )

    # Triton options
    parser.add_argument(
        "--triton_backend",
        default="auto",
        choices=["auto", "tensorrt", "onnxruntime"],
        help="Triton backend to configure (default: auto)",
    )

    # Benchmark options
    parser.add_argument("--n_runs", type=int, default=20, help="Benchmark runs per backend (default: 20)")
    parser.add_argument("--audio_duration", type=float, default=3.0, help="Test audio duration in seconds")
    parser.add_argument("--baseline_benchmark", type=str, default="", help="Path to baseline benchmark JSON")
    parser.add_argument("--max_latency_regression_pct", type=float, default=20.0,
                        help="Max allowed latency regression %% vs baseline (default: 20)")
    parser.add_argument(
        "--benchmark_backend",
        default="onnxruntime",
        choices=["onnxruntime", "faster_whisper_ct2", "tensorrt"],
        help="Which backend to use for regression gate (default: onnxruntime)",
    )

    # Output
    parser.add_argument(
        "--output",
        default="reports/serving_report.json",
        help="Path to output summary JSON",
    )
    parser.add_argument(
        "--fail_on_regression",
        action="store_true",
        help="Return non-zero if latency regression check fails",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from whisper_ja.config import Config
    from whisper_ja.serving.onnx_export import export_to_onnx
    from whisper_ja.serving.tensorrt_export import export_to_tensorrt
    from whisper_ja.serving.triton_config import generate_triton_repository
    from whisper_ja.serving.benchmark import run_benchmark, check_latency_regression

    # Build config
    config_kwargs: dict = {}
    if args.model_size:
        config_kwargs["model_size"] = args.model_size
    config = Config(**config_kwargs)

    # Apply overrides
    for attr in ["onnx_output_dir", "tensorrt_output_dir", "triton_output_dir",
                 "ct2_output_dir", "tensorrt_precision"]:
        val = getattr(args, attr, None)
        if val:
            setattr(config, attr, val)

    run_all = args.all
    summary: dict = {"model_size": config.model_size, "steps": {}}

    # ── ONNX export ──────────────────────────────────────────────────────────
    if run_all or args.export_onnx:
        print("\n" + "=" * 60)
        print("STEP: ONNX Export")
        print("=" * 60)
        model_dir = args.model_dir or config.merged_output_dir
        result = export_to_onnx(config, model_dir=model_dir)
        summary["steps"]["onnx_export"] = result
        if not result.get("success") and not result.get("skipped"):
            print("❌ ONNX export failed. Stopping pipeline.")
            _write_summary(summary, args.output)
            return 1

    # ── TensorRT export ───────────────────────────────────────────────────────
    if run_all or args.export_tensorrt:
        print("\n" + "=" * 60)
        print("STEP: TensorRT Export (GPU only)")
        print("=" * 60)
        onnx_dir = getattr(config, "onnx_output_dir",
                           f"./output/whisper-{config.model_size}-ja-onnx")
        result = export_to_tensorrt(config, onnx_dir=onnx_dir)
        summary["steps"]["tensorrt_export"] = result

    # ── Triton config generation ───────────────────────────────────────────────
    if run_all or args.gen_triton:
        print("\n" + "=" * 60)
        print("STEP: Triton Repository Generation")
        print("=" * 60)
        result = generate_triton_repository(config, backend=args.triton_backend)
        summary["steps"]["triton_config"] = result

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark_report = None
    if run_all or args.benchmark:
        print("\n" + "=" * 60)
        print("STEP: Latency Benchmark")
        print("=" * 60)
        benchmark_report = run_benchmark(
            config,
            n_runs=args.n_runs,
            audio_duration_sec=args.audio_duration,
        )
        summary["steps"]["benchmark"] = benchmark_report

        # Latency regression gate
        if args.baseline_benchmark:
            regression = check_latency_regression(
                report=benchmark_report,
                baseline_report_path=args.baseline_benchmark,
                backend=args.benchmark_backend,
                max_regression_pct=args.max_latency_regression_pct,
            )
            summary["steps"]["latency_regression"] = regression
            print(f"\n🚦 Latency regression: {regression['decision'].upper()}")
            if regression["decision"] == "fail":
                print(f"   ❌ {regression.get('backend')} latency regressed "
                      f"{regression.get('regression_pct')}% "
                      f"(threshold: {regression.get('max_regression_pct')}%)")
                _write_summary(summary, args.output)
                if args.fail_on_regression:
                    return 1
            elif regression["decision"] == "pass":
                print(f"   ✅ Within threshold "
                      f"({regression.get('current_mean_ms')}ms vs "
                      f"baseline {regression.get('baseline_mean_ms')}ms)")

    if not any([run_all, args.export_onnx, args.export_tensorrt, args.gen_triton, args.benchmark]):
        print("No action specified. Use --help to see available options.")
        return 1

    _write_summary(summary, args.output)
    print(f"\n✅ Serving pipeline complete. Summary: {args.output}")
    return 0


def _write_summary(summary: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
