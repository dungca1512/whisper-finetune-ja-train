#!/usr/bin/env python3
"""
Quality gate for model promotion.

Compares candidate metrics against baseline and outputs a decision report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR quality gate decision tool")
    parser.add_argument("--candidate_metrics", default="", help="Path to candidate metrics JSON")
    parser.add_argument("--baseline_metrics", default="", help="Path to baseline metrics JSON")
    parser.add_argument("--candidate_cer", type=float, default=None, help="Candidate CER override")
    parser.add_argument("--baseline_cer", type=float, default=None, help="Baseline CER override")
    parser.add_argument("--metric_key", default="eval_cer", help="Metric key to read from JSON")
    parser.add_argument(
        "--max_cer_regression",
        type=float,
        default=0.005,
        help="Maximum allowed CER regression (candidate <= baseline + this value)",
    )
    parser.add_argument(
        "--max_absolute_cer",
        type=float,
        default=0.30,
        help="Maximum allowed absolute CER for candidate",
    )
    parser.add_argument(
        "--require_baseline",
        action="store_true",
        help="Fail if baseline CER is not provided",
    )
    parser.add_argument("--model_repo_id", default="", help="Candidate merged model repo id")
    parser.add_argument("--adapter_repo_id", default="", help="Candidate adapter model repo id")
    parser.add_argument(
        "--output",
        default="reports/quality_gate_report.json",
        help="Path to output decision report",
    )
    parser.add_argument(
        "--fail_on_reject",
        action="store_true",
        help="Return non-zero when decision is reject",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_metric(data: dict[str, Any], metric_key: str) -> float | None:
    if metric_key in data and isinstance(data[metric_key], (int, float)):
        return float(data[metric_key])

    for parent_key in ["metrics", "eval_metrics", "eval", "results"]:
        parent = data.get(parent_key)
        if isinstance(parent, dict) and metric_key in parent and isinstance(parent[metric_key], (int, float)):
            return float(parent[metric_key])

    return None


def main() -> int:
    args = parse_args()

    candidate_cer = args.candidate_cer
    baseline_cer = args.baseline_cer

    if candidate_cer is None and args.candidate_metrics:
        candidate_data = load_json(args.candidate_metrics)
        candidate_cer = extract_metric(candidate_data, args.metric_key)

    if baseline_cer is None and args.baseline_metrics:
        baseline_data = load_json(args.baseline_metrics)
        baseline_cer = extract_metric(baseline_data, args.metric_key)

    checks: list[dict[str, Any]] = []

    candidate_available = candidate_cer is not None
    checks.append(
        {
            "name": "candidate_metric_available",
            "passed": candidate_available,
            "actual": candidate_cer,
            "details": f"metric_key={args.metric_key}",
        }
    )

    if candidate_available and args.max_absolute_cer > 0:
        checks.append(
            {
                "name": "candidate_absolute_cer",
                "passed": candidate_cer <= args.max_absolute_cer,
                "actual": candidate_cer,
                "threshold": args.max_absolute_cer,
            }
        )

    baseline_available = baseline_cer is not None
    checks.append(
        {
            "name": "baseline_metric_available",
            "passed": baseline_available if args.require_baseline else True,
            "actual": baseline_cer,
            "required": args.require_baseline,
        }
    )

    if candidate_available and baseline_available:
        allowed_cer = baseline_cer + args.max_cer_regression
        checks.append(
            {
                "name": "cer_regression_limit",
                "passed": candidate_cer <= allowed_cer,
                "actual": candidate_cer,
                "threshold": allowed_cer,
                "baseline": baseline_cer,
                "max_regression": args.max_cer_regression,
            }
        )

    passed = all(check["passed"] for check in checks)
    decision = "promote" if passed else "reject"

    report = {
        "decision": decision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_repo_id": args.model_repo_id,
            "adapter_repo_id": args.adapter_repo_id,
        },
        "metric_key": args.metric_key,
        "candidate_cer": candidate_cer,
        "baseline_cer": baseline_cer,
        "checks": checks,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Decision: {decision}")
    print(f"Report written: {output_path}")
    print(f"Candidate CER: {candidate_cer}")
    print(f"Baseline CER: {baseline_cer}")

    if args.fail_on_reject and decision != "promote":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
