from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.common import markdown_table


def build_correction_table(rows: list[dict[str, Any]]) -> str:
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model[str(row["model"])][f"{row['kernel_id']}:{row['condition']}"] = row
    table_rows: list[list[str]] = []
    for model, model_rows in sorted(by_model.items()):
        wrong_c2 = [row for key, row in model_rows.items() if key.endswith(":C2") and not bool(row.get("correct"))]
        corrected = 0
        for row in wrong_c2:
            key = f"{row['kernel_id']}:C4"
            if key in model_rows and bool(model_rows[key].get("correct")):
                corrected += 1
        total = len(wrong_c2)
        rate = f"{(100.0 * corrected / total):.1f}%" if total else "—"
        table_rows.append([model, str(total), str(corrected), rate])
    return markdown_table(["Model", "Wrong at C2", "Corrected after hint", "Correction rate"], table_rows)
