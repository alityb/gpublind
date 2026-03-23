from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from eval.analyze_results import load_results
from registry import KernelRegistry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPUBlind result integrity")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = KernelRegistry(profile_dir=args.profiles, mock=True)
    registry.load_mined(args.mined)
    registry.load_mined(args.kernelbench_compute)
    registry.load_mined(args.latency)
    registry.load_mined(args.register_spill)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    entries = registry.filter(confidence=args.min_confidence)
    entry_lookup = {entry.id: entry for entry in entries}
    rows = load_results(args.results)
    issues: list[str] = []
    warnings: list[str] = []
    if not rows:
        issues.append("No result files found.")
    api_errors = [row for row in rows if row.get("predicted_label") == "api_error"]
    if api_errors:
        issues.append(f"Found {len(api_errors)} api_error entries.")
    combos = Counter(
        (
            str(row.get("model")),
            int(row.get("trial", 1)),
            int(row.get("level", 1)),
            str(row.get("question_format", "label")),
            str(row.get("kernel_id")),
        )
        for row in rows
    )
    duplicates = [combo for combo, count in combos.items() if count > 1]
    if duplicates:
        issues.append(f"Found {len(duplicates)} duplicate model/trial/level/format/kernel combinations.")
    grouped: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[
            (
                str(row.get("model")),
                int(row.get("trial", 1)),
                int(row.get("level", 1)),
                str(row.get("question_format", "label")),
            )
        ].add(str(row.get("kernel_id")))
    runs: dict[tuple[str, int, int], dict[str, set[str]]] = defaultdict(dict)
    for (model, trial, level, question_format), kernel_ids in grouped.items():
        runs[(model, trial, level)][question_format] = kernel_ids
    for run_key, formats in sorted(runs.items()):
        baseline_ids = formats.get("label")
        if baseline_ids is None:
            baseline_ids = set().union(*formats.values())
        for question_format, seen_ids in sorted(formats.items()):
            combo = (*run_key, question_format)
            missing = sorted(baseline_ids - seen_ids)
            if missing:
                message = f"Missing {len(missing)} kernels for combo {combo}: first missing={missing[0]}"
                if question_format == "label":
                    issues.append(message)
                else:
                    warnings.append(message)
    mismatches = []
    for row in rows:
        kernel_id = str(row.get("kernel_id"))
        if kernel_id not in entry_lookup:
            continue
        if str(row.get("true_bottleneck")) != entry_lookup[kernel_id].true_bottleneck:
            mismatches.append(kernel_id)
    if mismatches:
        issues.append(f"Found {len(mismatches)} true_bottleneck mismatches against registry.")
    parsed_rows = [row for row in rows if row.get("predicted_label") is not None]
    parse_errors = [row for row in parsed_rows if row.get("predicted_label") == "parse_error"]
    if parsed_rows and (len(parse_errors) / len(parsed_rows)) >= 0.05:
        issues.append(f"Parse error rate is {len(parse_errors) / len(parsed_rows):.2%}, which exceeds 5%.")
    if issues:
        print("FAIL")
        for issue in issues:
            print(f"- {issue}")
        for warning in warnings:
            print(f"- WARNING: {warning}")
        return 1
    print("PASS")
    print(f"Validated {len(rows)} result entries across {len(grouped)} model/level/format combinations.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
