from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from registry.kernel_entry import kernel_entry_to_dict
from registry.registry import KernelRegistry


def load_optional_sources(
    registry: KernelRegistry,
    mined: Path,
    kernelbot: Path | None,
    kernelbench_compute: Path | None,
    latency: Path | None,
    register_spill: Path | None,
) -> None:
    registry.load_mined(mined)
    for path in [kernelbench_compute, latency, register_spill]:
        if path is not None:
            registry.load_mined(path)
    if kernelbot is not None:
        registry.load_kernelbot(kernelbot)


def build_registry(
    mined: Path,
    kernels: Path,
    profiles: Path,
    mock: bool,
    kernelbot: Path | None = None,
    kernelbench_compute: Path | None = None,
    latency: Path | None = None,
    register_spill: Path | None = None,
) -> list[dict]:
    registry = KernelRegistry(profile_dir=profiles, mock=mock)
    load_optional_sources(registry, mined, kernelbot, kernelbench_compute, latency, register_spill)
    registry.load_handwritten(kernels)
    return [kernel_entry_to_dict(entry) for entry in registry]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the merged GPUBlind registry")
    parser.add_argument("--mined", type=Path, required=True)
    parser.add_argument("--kernelbot", type=Path, default=None)
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = build_registry(
        args.mined,
        args.kernels,
        args.profiles,
        args.mock,
        args.kernelbot,
        args.kernelbench_compute,
        args.latency,
        args.register_spill,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2), encoding="utf-8")
    by_bottleneck = Counter(item["true_bottleneck"] for item in items)
    by_source = Counter(item["source"] for item in items)
    print(f"Registry size: {len(items)}")
    print(f"By bottleneck: {dict(sorted(by_bottleneck.items()))}")
    print(f"By source: {dict(sorted(by_source.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
