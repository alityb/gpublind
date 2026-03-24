from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus import CorpusEntry, load_corpus

VALID_LABELS = [
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "occupancy-limited",
    "register-spill",
]


@dataclass
class RandomBaseline:
    name: str = "Random"
    seed: int = 7

    def predict(self, entry: CorpusEntry) -> str:
        return random.Random(f"{self.seed}:{entry.id}").choice(VALID_LABELS)


@dataclass
class FrequencyBaseline:
    name: str = "Frequency (always memory-bound)"

    def predict(self, entry: CorpusEntry) -> str:
        return "memory-bound"


@dataclass
class RooflineBaseline:
    name: str = "Roofline (AI vs ridge)"

    def predict(self, entry: CorpusEntry) -> str:
        ai = float(entry.profile["arithmetic_intensity_flop_per_byte"])
        ridge = float(entry.profile["hardware"]["ridge_point_flop_per_byte"])
        return "compute-bound" if ai > ridge else "memory-bound"


@dataclass
class RuleBasedExpertBaseline:
    name: str = "Rule-Based Expert"

    def predict(self, entry: CorpusEntry) -> str:
        p = entry.profile
        regs = int(p["register_count_per_thread"])
        occ = float(p["achieved_occupancy_pct"])
        stall_long = float(p["stall_long_scoreboard_pct"])
        dram = float(p["dram_bw_utilization_pct"])
        ai = float(p["arithmetic_intensity_flop_per_byte"])
        ridge = float(p["hardware"]["ridge_point_flop_per_byte"])

        if regs >= 200 and dram < 20.0:
            return "register-spill"
        if occ < 35.0:
            return "occupancy-limited"
        if stall_long > 30.0 and dram < 10.0:
            return "latency-bound"
        if ai < ridge:
            if dram > 40.0:
                return "memory-bound"
            return "latency-bound"
        return "compute-bound"


BASELINES = [
    RandomBaseline(),
    FrequencyBaseline(),
    RooflineBaseline(),
    RuleBasedExpertBaseline(),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print v2 baseline predictions")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_corpus(args.kernels, min_confidence="low")
    for baseline in BASELINES:
        print(f"=== {baseline.name} ===")
        for entry in entries:
            print(f"{entry.id}: {baseline.predict(entry)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
