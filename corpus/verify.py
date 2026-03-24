from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus.schema import load_corpus, load_entry


def roofline_test(profile: dict[str, Any]) -> str:
    ai = float(profile["arithmetic_intensity_flop_per_byte"])
    ridge = float(profile["hardware"]["ridge_point_flop_per_byte"])
    return "compute-bound" if ai > ridge else "memory-bound"


def bandwidth_test(profile: dict[str, Any]) -> str:
    regs = int(profile["register_count_per_thread"])
    occ = float(profile["achieved_occupancy_pct"])
    stall_long = float(profile["stall_long_scoreboard_pct"])
    dram = float(profile["dram_bw_utilization_pct"])
    local_bytes = float(profile.get("local_memory_bytes", 0.0))
    ai = float(profile["arithmetic_intensity_flop_per_byte"])
    ridge = float(profile["hardware"]["ridge_point_flop_per_byte"])
    if regs >= 200 and local_bytes > 0.0:
        return "register-spill"
    if occ < 35.0:
        return "occupancy-limited"
    if stall_long > 30.0 and dram < 10.0:
        return "latency-bound"
    if ai < ridge and dram > 40.0:
        return "memory-bound"
    if ai > ridge:
        return "compute-bound"
    return "latency-bound"


def stall_test(profile: dict[str, Any]) -> str:
    regs = int(profile["register_count_per_thread"])
    occ = float(profile["achieved_occupancy_pct"])
    stall_long = float(profile["stall_long_scoreboard_pct"])
    stall_mem = float(profile["stall_memory_pct"])
    dram = float(profile["dram_bw_utilization_pct"])
    local_bytes = float(profile.get("local_memory_bytes", 0.0))
    ai = float(profile["arithmetic_intensity_flop_per_byte"])
    ridge = float(profile["hardware"]["ridge_point_flop_per_byte"])
    if regs >= 200 and local_bytes > 0.0:
        return "register-spill"
    if occ < 35.0:
        return "occupancy-limited"
    if stall_long > max(30.0, stall_mem) and dram < 10.0:
        return "latency-bound"
    if stall_mem >= stall_long and dram > 30.0:
        return "memory-bound"
    if ai > ridge:
        return "compute-bound"
    return "memory-bound"


def verify_profile(profile: dict[str, Any]) -> dict[str, str]:
    votes = {
        "roofline_test": roofline_test(profile),
        "bandwidth_test": bandwidth_test(profile),
        "stall_test": stall_test(profile),
    }
    counts = Counter(votes.values())
    label, count = counts.most_common(1)[0]
    confidence = "high" if count == 3 else "medium" if count == 2 else "low"
    return {
        **votes,
        "consensus": label if count >= 2 else "ambiguous",
        "confidence": confidence,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GPUBlind corpus profiles")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_corpus(args.kernels, min_confidence="low")
    mismatches = 0
    confidence_counts: Counter[str] = Counter()
    for entry in entries:
        verification = verify_profile(entry.profile)
        confidence_counts[verification["confidence"]] += 1
        if verification["consensus"] != entry.true_bottleneck:
            mismatches += 1
        if args.write:
            payload = dict(entry.profile)
            payload["verification"] = verification
            entry.profile_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Verified kernels: {len(entries)}")
    print(f"Confidence counts: {dict(sorted(confidence_counts.items()))}")
    print(f"Consensus mismatches vs meta: {mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
