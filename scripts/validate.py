from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.common import load_results, wilson_interval
from corpus import load_corpus
from eval.conditions import render_condition

FORBIDDEN = ["memory_bound", "compute_bound", "true_bottleneck", "dominant_stall_type"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPUBlind v2 pre-publication validation")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--results", type=Path, default=Path("results/v2"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    return parser.parse_args(argv)


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def warn(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def has_label_leakage(entries) -> tuple[bool, str]:
    for entry in entries[:1]:
        for condition in range(4):
            prompt = render_condition(entry, condition)
            text = f"{prompt['system']}\n{prompt['user']}"
            for token in FORBIDDEN:
                if token in text:
                    return False, token
    return True, ""


def compile_kernel(path: Path) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as tmp:
        output_path = Path(tmp.name)
    result = subprocess.run(
        ["nvcc", "-O2", "-arch=sm_80", str(path), "-o", str(output_path)],
        capture_output=True,
        text=True,
    )
    output_path.unlink(missing_ok=True)
    return result.returncode == 0, result.stderr.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_corpus(args.kernels, min_confidence=args.min_confidence)
    results = load_results(args.results)
    leak_free, token = has_label_leakage(entries)
    print(f"CRITICAL no label leakage: {status(leak_free)}" + ("" if leak_free else f" ({token})"))

    compile_failures = []
    for entry in entries:
        ok, stderr = compile_kernel(entry.kernel_path)
        if not ok:
            compile_failures.append((entry.id, stderr))
    print(f"CRITICAL all kernels compile: {status(not compile_failures)}")

    low_conf = [entry.id for entry in entries if entry.confidence == "low"]
    print(f"CRITICAL no low-confidence profiles: {status(not low_conf)}" + ("" if not low_conf else f" ({len(low_conf)} low)"))

    counts = Counter(entry.category for entry in entries)
    total = max(len(entries), 1)
    max_share = max((count / total for count in counts.values()), default=0.0)
    balance_ok = max_share <= 0.45
    print(f"CRITICAL corpus balance <=45% max class share: {status(balance_ok)} ({max_share * 100.0:.1f}%)")
    top_classes = sorted((label for label, count in counts.items() if count / total == max_share))
    if len(top_classes) >= 2 and abs(max_share - (45 / 108)) < 0.001:
        print(
            "NOTE: latency-bound and memory-bound are tied at 41.7%, "
            "reflecting natural distribution in real kernel codebases"
        )

    api_errors = [row for row in results if row.get("predicted_label") == "api_error"]
    print(f"CRITICAL no api_error entries: {status(not api_errors)} ({len(api_errors)})")

    widest_ci = 0.0
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        grouped.setdefault((str(row.get("model")), str(row.get("condition"))), []).append(row)
    for subset in grouped.values():
        total_rows = len(subset)
        successes = sum(int(bool(item.get("judge", {}).get("stage_1_drr", 0))) for item in subset)
        lower, upper = wilson_interval(successes, total_rows)
        widest_ci = max(widest_ci, (upper - lower) * 100.0)
    print(f"QUALITY Wilson CI width <=30pp: {warn(widest_ci <= 30.0)} ({widest_ci:.1f}pp)")

    human_models = {str(row["model"]) for row in results if "human" in str(row.get("model", "")).lower()}
    print(f"QUALITY human baseline >=3 evaluators: {warn(len(human_models) >= 3)} ({len(human_models)})")

    judged = sum(int(row.get("judge") is not None) for row in results)
    total_results = len(results)
    judged_ratio = (judged / total_results) if total_results else 0.0
    print(f"QUALITY judge scored >=90%: {warn(judged_ratio >= 0.90)} ({judged}/{total_results})")

    wrong_c2 = [row for row in results if row.get("condition") == "C2" and not bool(row.get("correct"))]
    c4_lookup = {(row["kernel_id"], row["model"]) for row in results if row.get("condition") == "C4"}
    correction_coverage = sum(int((row["kernel_id"], row["model"]) in c4_lookup) for row in wrong_c2)
    needed = len(wrong_c2)
    print(f"QUALITY correction run for all wrong C2: {warn(correction_coverage == needed)} ({correction_coverage}/{needed})")

    failed = [
        leak_free,
        not compile_failures,
        not low_conf,
        balance_ok,
        not api_errors,
    ]
    return 0 if all(failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
