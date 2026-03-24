from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from analysis.by_category import build_category_table
from analysis.by_condition import build_information_sensitivity
from analysis.common import load_results, markdown_table
from analysis.correction import build_correction_table
from analysis.funnel import build_funnel_table
from corpus import load_corpus
from eval.baselines import BASELINES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPUBlind v2 markdown report")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--results", type=Path, default=Path("results/v2"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/report.md"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    return parser.parse_args(argv)


def baseline_rows(entries, condition: str = "C2") -> list[list[str]]:
    rows: list[list[str]] = []
    total = len(entries)
    for baseline in BASELINES:
        correct = sum(int(baseline.predict(entry) == entry.true_bottleneck) for entry in entries)
        drr = f"{(100.0 * correct / total):.1f}%" if total else "—"
        rows.append([baseline.name, condition, drr, "—", "—"])
    return rows


def confidence_table(rows):
    grouped: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["model"])][str(row.get("confidence", "LOW"))].append(bool(row.get("correct")))
    output: list[list[str]] = []
    for model, by_conf in sorted(grouped.items()):
        vals = []
        for label in ["HIGH", "MEDIUM", "LOW"]:
            subset = by_conf.get(label, [])
            vals.append(f"{(100.0 * sum(subset) / len(subset)):.1f}%" if subset else "—")
        output.append([model, *vals])
    return markdown_table(["Model", "When HIGH confidence", "When MEDIUM", "When LOW"], output)


def hw_b_case_study(rows) -> str:
    selected = [row for row in rows if row.get("kernel_id") == "hw_B" and row.get("model") == "claude-opus-4-6" and row.get("condition") in {"C0", "C1", "C2"}]
    selected.sort(key=lambda item: item["condition"])
    blocks = []
    for row in selected:
        blocks.append(
            f"#### {row['condition']}\n"
            f"BOTTLENECK: {row.get('predicted_label')}\n"
            f"CONFIDENCE: {row.get('confidence')}\n"
            f"REASONING: {row.get('reasoning')}\n"
        )
    if not blocks:
        return "No hw_B case study results available."
    return "\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_corpus(args.kernels, min_confidence=args.min_confidence)
    rows = load_results(args.results)
    report = [
        "# GPUBlind v2 Report",
        "",
        "## 1. The Diagnostic Funnel",
        "",
        build_funnel_table(rows, baseline_rows(entries)),
        "",
        "## 2. Per-category breakdown (C2)",
        "",
        build_category_table(rows, entries, condition="C2"),
        "",
        "## 3. Information sensitivity",
        "",
        build_information_sensitivity(rows),
        "",
        "## 4. One-shot correction rate",
        "",
        build_correction_table(rows),
        "",
        "## 5. Confidence calibration",
        "",
        confidence_table(rows),
        "",
        "## 6. The hw_B case study",
        "",
        hw_b_case_study(rows),
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
