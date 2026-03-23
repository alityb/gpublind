from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

DEFAULT_INPUT = Path("data/mined_kernels.jsonl")
DEFAULT_OUTPUT = Path("data/latency_bound_candidates.jsonl")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latency-bound candidates from mined kernels")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ridge_point(entry: dict[str, Any]) -> float:
    raw = entry.get("ncu_profile", {}).get("raw", {})
    if "ridge_point" in raw:
        return float(raw["ridge_point"])
    roof = raw.get("roof", {})
    peak_bw = float(roof.get("peak_bw_tbps", 0.0))
    peak_flops = float(roof.get("peak_flops_tflops", 0.0))
    return (peak_flops / peak_bw) if peak_bw > 0.0 else 62.0


def is_latency_signature(entry: dict[str, Any]) -> bool:
    profile = entry.get("ncu_profile", {})
    return (
        float(profile.get("stall_long_sb_pct", 0.0)) > 0.35
        and float(profile.get("dram_bw_utilization", 1.0)) < 0.10
        and float(profile.get("arithmetic_intensity", 0.0)) < ridge_point(entry)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.input)
    if args.mock:
        selected = [row for row in rows if row.get("true_bottleneck") == "latency-bound"][:5]
    else:
        selected = [row for row in rows if is_latency_signature(row)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(selected)} latency-bound candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
