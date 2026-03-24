from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.common import markdown_table


def build_information_sensitivity(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for row in rows:
        judge = row.get("judge", {})
        model = str(row["model"])
        condition = str(row["condition"])
        correct, total = grouped[model].get(condition, (0, 0))
        grouped[model][condition] = (correct + int(bool(judge.get("stage_1_drr", 0))), total + 1)

    def rate(pair: tuple[int, int]) -> float:
        correct, total = pair
        return (correct / total) if total else 0.0

    rows_out: list[list[str]] = []
    for model in sorted(grouped):
        c0 = rate(grouped[model].get("C0", (0, 0)))
        c1 = rate(grouped[model].get("C1", (0, 0)))
        c2 = rate(grouped[model].get("C2", (0, 0)))
        c3 = rate(grouped[model].get("C3", (0, 0)))
        rows_out.append([
            model,
            f"{(c1 - c0) * 100.0:+.1f} pp",
            f"{(c2 - c0) * 100.0:+.1f} pp",
            f"{(c2 - c1) * 100.0:+.1f} pp",
            f"{(c3 - c0) * 100.0:+.1f} pp",
        ])
    return markdown_table(["Metric", "C0→C1", "C0→C2", "C1→C2", "C0→C3"], rows_out)
