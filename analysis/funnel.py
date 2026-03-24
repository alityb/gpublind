from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.common import fmt_rate, markdown_table


def build_funnel_table(rows: list[dict[str, Any]], baseline_rows: list[list[str]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["condition"]))].append(row)
    output: list[list[str]] = []
    for (model, condition), subset in sorted(grouped.items()):
        total = len(subset)
        drr = sum(int(bool(item.get("judge", {}).get("stage_1_drr", 0))) for item in subset)
        rvr = sum(int(item.get("judge", {}).get("stage_2_rvr") == 1) for item in subset)
        mpr = sum(int(item.get("judge", {}).get("stage_3_mpr") == 1) for item in subset)
        output.append([model, condition, fmt_rate(drr, total), fmt_rate(rvr, total), fmt_rate(mpr, total)])
    output.extend(baseline_rows)
    return markdown_table(["Model", "Condition", "DRR", "RVR", "MPR"], output)
