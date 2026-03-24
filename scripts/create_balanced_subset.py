from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

from registry import KernelRegistry
from registry.registry import ground_truth_verified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a less imbalanced verified evaluation subset")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    parser.add_argument("--output", type=Path, default=Path("data/balanced_subset.json"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


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

    verified = [entry for entry in registry.filter(confidence=args.min_confidence) if is_verified_entry(entry)]
    non_memory = sorted([entry for entry in verified if entry.true_bottleneck != "memory-bound"], key=lambda entry: entry.id)
    memory = sorted([entry for entry in verified if entry.true_bottleneck == "memory-bound"], key=lambda entry: entry.id)

    rng = random.Random(args.seed)
    sampled_memory = list(memory)
    # Use a stricter cap than 2:1 so the exported subset actually reduces memory-bound dominance.
    max_memory = max(1, len(non_memory) - 1) if non_memory else len(memory)
    if len(memory) > max_memory:
        sampled_memory = sorted(rng.sample(memory, max_memory), key=lambda entry: entry.id)

    selected = sorted(non_memory + sampled_memory, key=lambda entry: entry.id)
    counts = Counter(entry.true_bottleneck for entry in selected)
    payload = {
        "kernel_ids": [entry.id for entry in selected],
        "counts": dict(sorted(counts.items())),
        "memory_share": round((counts.get("memory-bound", 0) / max(len(selected), 1)) * 100.0, 2),
        "min_confidence": args.min_confidence,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} kernel ids to {args.output}")
    print(f"Counts: {dict(sorted(counts.items()))}")
    print(f"Memory-bound share: {payload['memory_share']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
