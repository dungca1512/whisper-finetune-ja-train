#!/usr/bin/env python3
"""
Dataset validation for ASR pipelines.

Outputs a JSON report and optionally fails when quality checks do not pass.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ASR dataset quality")
    parser.add_argument("--dataset_name", required=True, help="HF dataset name")
    parser.add_argument("--dataset_config", default="", help="HF dataset config/subset")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text_column", default="transcription", help="Text column name")
    parser.add_argument("--audio_column", default="audio", help="Audio column name")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to validate")
    parser.add_argument("--min_samples", type=int, default=100, help="Minimum sample count check")
    parser.add_argument(
        "--max_empty_text_ratio",
        type=float,
        default=0.05,
        help="Fail if empty text ratio exceeds this threshold",
    )
    parser.add_argument(
        "--max_invalid_audio_ratio",
        type=float,
        default=0.05,
        help="Fail if invalid audio ratio exceeds this threshold",
    )
    parser.add_argument(
        "--min_avg_duration_sec",
        type=float,
        default=0.5,
        help="Fail if average duration (seconds) is lower than this threshold",
    )
    parser.add_argument(
        "--min_japanese_char_ratio",
        type=float,
        default=0.30,
        help="Fail if average Japanese character ratio is lower than this threshold",
    )
    parser.add_argument(
        "--skip_audio_checks",
        action="store_true",
        help="Skip audio array validation and duration checks",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True")
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to env HF_TOKEN)",
    )
    parser.add_argument(
        "--output",
        default="reports/data_validation_report.json",
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--max_issue_examples",
        type=int,
        default=20,
        help="Maximum number of issue examples to include in report",
    )
    parser.add_argument(
        "--fail_on_check",
        action="store_true",
        help="Return non-zero if any quality check fails",
    )
    return parser.parse_args()


def is_japanese_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0xFF66 <= code <= 0xFF9D  # Half-width Katakana
    )


def japanese_char_ratio(text: str) -> float:
    visible_chars = [ch for ch in text if not ch.isspace()]
    if not visible_chars:
        return 0.0
    japanese_chars = sum(1 for ch in visible_chars if is_japanese_char(ch))
    return japanese_chars / len(visible_chars)


def check_audio_sample(audio: Any) -> tuple[bool, float | None, str]:
    if not isinstance(audio, dict):
        return False, None, "audio is not a dictionary"

    if "array" not in audio:
        return False, None, "audio dict missing key 'array'"
    if "sampling_rate" not in audio:
        return False, None, "audio dict missing key 'sampling_rate'"

    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]

    if sampling_rate is None or sampling_rate <= 0:
        return False, None, "sampling_rate is invalid"
    if audio_array is None:
        return False, None, "audio array is None"
    if len(audio_array) == 0:
        return False, None, "audio array is empty"

    duration = float(len(audio_array)) / float(sampling_rate)
    return True, duration, ""


def main() -> int:
    args = parse_args()
    token = args.hf_token or None
    dataset_config = args.dataset_config or None

    load_kwargs: dict[str, Any] = {
        "split": args.split,
    }
    if token:
        load_kwargs["token"] = token
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    print(
        "Loading dataset:",
        args.dataset_name,
        f"(config={dataset_config or 'default'}, split={args.split})",
    )
    dataset = load_dataset(args.dataset_name, dataset_config, **load_kwargs)

    total_source_samples = len(dataset)
    if args.max_samples > 0 and total_source_samples > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    evaluated_samples = len(dataset)
    issues: list[dict[str, Any]] = []
    missing_columns: list[str] = []
    for required_column in [args.text_column, args.audio_column]:
        if required_column not in dataset.column_names:
            missing_columns.append(required_column)

    empty_text_count = 0
    invalid_audio_count = 0
    valid_audio_count = 0
    durations: list[float] = []
    jp_ratios: list[float] = []

    if not missing_columns:
        for idx in range(evaluated_samples):
            sample = dataset[idx]
            text_raw = sample.get(args.text_column, "")
            text = "" if text_raw is None else str(text_raw).strip()

            if not text:
                empty_text_count += 1
                if len(issues) < args.max_issue_examples:
                    issues.append(
                        {
                            "index": idx,
                            "type": "empty_text",
                            "message": f"Column '{args.text_column}' is empty",
                        }
                    )
            else:
                jp_ratios.append(japanese_char_ratio(text))

            if args.skip_audio_checks:
                continue

            audio = sample.get(args.audio_column)
            ok, duration, reason = check_audio_sample(audio)
            if ok:
                valid_audio_count += 1
                if duration is not None:
                    durations.append(duration)
            else:
                invalid_audio_count += 1
                if len(issues) < args.max_issue_examples:
                    issues.append(
                        {
                            "index": idx,
                            "type": "invalid_audio",
                            "message": reason,
                        }
                    )

    empty_text_ratio = (empty_text_count / evaluated_samples) if evaluated_samples else 1.0
    invalid_audio_ratio = (invalid_audio_count / evaluated_samples) if evaluated_samples else 1.0
    avg_duration = fmean(durations) if durations else 0.0
    avg_jp_ratio = fmean(jp_ratios) if jp_ratios else 0.0

    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "required_columns_present",
            "passed": not missing_columns,
            "actual": dataset.column_names,
            "expected": [args.text_column, args.audio_column],
            "details": "Missing columns: " + ", ".join(missing_columns) if missing_columns else "",
        }
    )
    checks.append(
        {
            "name": "minimum_samples",
            "passed": evaluated_samples >= args.min_samples,
            "actual": evaluated_samples,
            "threshold": args.min_samples,
        }
    )
    checks.append(
        {
            "name": "empty_text_ratio",
            "passed": empty_text_ratio <= args.max_empty_text_ratio,
            "actual": empty_text_ratio,
            "threshold": args.max_empty_text_ratio,
        }
    )
    if not args.skip_audio_checks:
        checks.append(
            {
                "name": "invalid_audio_ratio",
                "passed": invalid_audio_ratio <= args.max_invalid_audio_ratio,
                "actual": invalid_audio_ratio,
                "threshold": args.max_invalid_audio_ratio,
            }
        )
        checks.append(
            {
                "name": "average_audio_duration_sec",
                "passed": avg_duration >= args.min_avg_duration_sec,
                "actual": avg_duration,
                "threshold": args.min_avg_duration_sec,
            }
        )
    checks.append(
        {
            "name": "japanese_character_ratio",
            "passed": avg_jp_ratio >= args.min_japanese_char_ratio,
            "actual": avg_jp_ratio,
            "threshold": args.min_japanese_char_ratio,
        }
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "pass" if passed else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": args.dataset_name,
            "config": dataset_config,
            "split": args.split,
            "source_samples": total_source_samples,
            "evaluated_samples": evaluated_samples,
            "text_column": args.text_column,
            "audio_column": args.audio_column,
            "skip_audio_checks": args.skip_audio_checks,
        },
        "metrics": {
            "empty_text_count": empty_text_count,
            "empty_text_ratio": empty_text_ratio,
            "invalid_audio_count": invalid_audio_count,
            "invalid_audio_ratio": invalid_audio_ratio,
            "valid_audio_count": valid_audio_count,
            "avg_audio_duration_sec": avg_duration,
            "avg_japanese_char_ratio": avg_jp_ratio,
        },
        "checks": checks,
        "issue_examples": issues,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation status: {report['status']}")
    print(f"Report written: {output_path}")
    print(f"Evaluated samples: {evaluated_samples}")
    print(f"Empty text ratio: {empty_text_ratio:.4f}")
    if not args.skip_audio_checks:
        print(f"Invalid audio ratio: {invalid_audio_ratio:.4f}")
        print(f"Avg duration: {avg_duration:.2f}s")
    print(f"Avg Japanese char ratio: {avg_jp_ratio:.4f}")

    if args.fail_on_check and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
