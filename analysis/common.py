from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_results(root: Path = Path("results/v2")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.json")):
        if "judge_cache" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "predicted_label" in payload:
            rows.append(payload)
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_rate(successes: int, total: int) -> str:
    if total == 0:
        return "—"
    lower, upper = wilson_interval(successes, total)
    value = successes / total
    return f"{value * 100.0:.1f}% [{lower * 100.0:.1f}%, {upper * 100.0:.1f}%]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
