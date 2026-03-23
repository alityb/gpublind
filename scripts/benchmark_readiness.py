from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from eval.analyze_results import automatic_grounded_reasoning, latest_per_combo, load_results, wilson_interval
from registry import KernelRegistry

LABELS = [
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "occupancy-limited",
    "register-spill",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GPUBlind benchmark readiness")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    return parser.parse_args(argv)


def build_registry(args: argparse.Namespace) -> KernelRegistry:
    registry = KernelRegistry(profile_dir=args.profiles, mock=True)
    registry.load_mined(args.mined)
    registry.load_mined(args.kernelbench_compute)
    registry.load_mined(args.latency)
    registry.load_mined(args.register_spill)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    return registry


def status(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = build_registry(args)
    entries = registry.filter(confidence=args.min_confidence)
    rows = latest_per_combo(load_results(args.results))
    label_rows = [row for row in rows if str(row.get("question_format", "label")) == "label"]
    label_rows = [row for row in label_rows if str(row.get("kernel_id")) in {entry.id for entry in entries}]
    label_df = pd.DataFrame(label_rows)

    verified = [entry for entry in entries if entry.ncu_profile.raw.get("ground_truth_verified", True)]
    bottleneck_counts = Counter(entry.true_bottleneck for entry in verified)
    missing_labels = [label for label in LABELS if bottleneck_counts.get(label, 0) == 0]
    enough_verified = len(verified) >= 50

    parse_error_rate = 0.0
    if not label_df.empty:
        parse_error_rate = float((label_df["predicted_label"] == "parse_error").mean())

    widest_ci = 0.0
    ci_records: list[str] = []
    if not label_df.empty:
        for (model, level), subset in label_df.groupby(["model", "level"], dropna=False):
            total = len(subset)
            successes = int(subset["correct"].astype(bool).sum())
            lower, upper = wilson_interval(successes, total)
            width = (upper - lower) * 100.0
            widest_ci = max(widest_ci, width)
            ci_records.append(f"{model} L{level}: {successes}/{total}, CI width={width:.1f}pp")

    grounded_count = 0
    groundable_total = 0
    if not label_df.empty:
        grounded_values = [automatic_grounded_reasoning(row, registry) for row in label_df.to_dict("records")]
        scored = [value for value in grounded_values if value is not None]
        grounded_count = sum(value is True for value in scored)
        groundable_total = len(scored)
    grounded_rate = (grounded_count / groundable_total) if groundable_total else 0.0

    print("GPUBlind Benchmark Readiness")
    print()
    print(f"Corpus size: {len(entries)} filtered kernels")
    print(f"Verified kernels: {len(verified)} [{status(enough_verified)}: target >= 50]")
    print(f"Class coverage: {dict(sorted(bottleneck_counts.items()))} [{status(not missing_labels)}]")
    if missing_labels:
        print(f"Missing classes: {', '.join(missing_labels)}")
    print(f"Label result count: {len(label_df)}")
    print(f"Parse error rate: {parse_error_rate * 100.0:.2f}% [{status(parse_error_rate < 0.05)}: target < 5%]")
    print(f"Widest Wilson CI: {widest_ci:.1f}pp [{status(widest_ci <= 30.0)}: target <= 30pp]")
    print(f"Evidence-grounded correct rate: {grounded_rate * 100.0:.2f}% ({grounded_count}/{groundable_total} groundable answers)")
    print()
    print("Per-slice CI widths:")
    if ci_records:
        for record in ci_records:
            print(f"- {record}")
    else:
        print("- No label results available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
