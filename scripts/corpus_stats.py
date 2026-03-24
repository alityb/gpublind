from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from registry import KernelRegistry
from registry.registry import ground_truth_verified, verification_confidence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print GPUBlind corpus statistics")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--subset", type=Path, default=None)
    return parser.parse_args(argv)


def load_subset_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("kernel_ids", []) if isinstance(payload, dict) else payload
    return {str(item) for item in items}


def is_verified_entry(entry: object) -> bool:
    profile = entry.ncu_profile
    numeric_values = [
        profile.arithmetic_intensity,
        profile.global_load_efficiency,
        profile.achieved_occupancy,
        profile.stall_long_sb_pct,
        profile.stall_mem_pct,
        profile.l2_hit_rate,
        profile.dram_bw_utilization,
    ]
    return (
        ground_truth_verified(entry)
        and profile.arithmetic_intensity > 0.0
        and profile.dominant_stall_type != "unknown"
        and all(value >= 0.0 for value in numeric_values)
        and profile.register_count >= 0
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = KernelRegistry(profile_dir=args.profiles, mock=True)
    registry.load_mined(args.mined)
    registry.load_mined(args.kernelbench_compute)
    registry.load_mined(args.latency)
    registry.load_mined(args.register_spill)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    entries = list(registry)
    subset_ids = load_subset_ids(args.subset)
    if subset_ids:
        entries = [entry for entry in entries if entry.id in subset_ids]
    verified = [entry for entry in entries if is_verified_entry(entry)]
    by_bottleneck = Counter(entry.true_bottleneck for entry in verified)
    by_source = Counter(entry.source for entry in entries)
    by_difficulty = Counter(entry.difficulty for entry in entries)
    by_confidence = Counter(verification_confidence(entry) for entry in verified)
    print(f"Total kernels: {len(entries)}")
    print(f"Verified (ncu profiled): {len(verified)}")
    print(f"By bottleneck: {dict(sorted(by_bottleneck.items()))}")
    print(f"By source: {dict(sorted(by_source.items()))}")
    print(f"By difficulty: {dict(sorted(by_difficulty.items()))}")
    print(f"By confidence: {dict(sorted(by_confidence.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
