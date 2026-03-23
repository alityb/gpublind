from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

DEFAULT_INPUT = Path("data/mined_kernels.jsonl")
DEFAULT_OUTPUT = Path("data/register_spill_candidates.jsonl")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract register-spill candidates from mined kernels")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def local_memory_traffic(entry: dict[str, Any]) -> float:
    raw = entry.get("ncu_profile", {}).get("raw", {})
    traffic = 0.0
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        lowered = str(key).lower()
        if "local" in lowered and ("mem" in lowered or "byte" in lowered or "spill" in lowered):
            traffic = max(traffic, float(value))
    return traffic


def is_register_spill_signature(entry: dict[str, Any]) -> bool:
    profile = entry.get("ncu_profile", {})
    return int(profile.get("register_count", 0)) >= 200 or local_memory_traffic(entry) > 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.input)
    if args.mock:
        selected = rows[:3]
    else:
        selected = [row for row in rows if is_register_spill_signature(row)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(selected)} register-spill candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
