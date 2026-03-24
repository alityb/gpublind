from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.common import markdown_table
from corpus import CorpusEntry


def build_category_table(rows: list[dict[str, Any]], entries: list[CorpusEntry], condition: str = "C2") -> str:
    lookup = {entry.id: entry.category for entry in entries}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("condition") != condition:
            continue
        grouped[str(row["model"])][lookup.get(str(row["kernel_id"]), "unknown")].append(row)
    categories = ["memory-bound", "compute-bound", "latency-bound", "occupancy-limited", "register-spill"]
    table_rows: list[list[str]] = []
    for model, model_rows in sorted(grouped.items()):
        row = [model]
        for category in categories:
            subset = model_rows.get(category, [])
            total = len(subset)
            correct = sum(int(bool(item.get("judge", {}).get("stage_1_drr", 0))) for item in subset)
            value = f"{(100.0 * correct / total):.1f}% (N={total})" if total else "—"
            row.append(value)
        table_rows.append(row)
    return markdown_table(["Model", "memory", "compute", "latency", "occupancy", "register"], table_rows)
