from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from registry.kernel_entry import kernel_entry_to_dict
from registry.registry import KernelRegistry


def build_registry(mined: Path, kernelbot: Path | None, kernels: Path, profiles: Path, mock: bool) -> list[dict]:
    registry = KernelRegistry(profile_dir=profiles, mock=mock)
    registry.load_mined(mined)
    if kernelbot is not None:
        registry.load_kernelbot(kernelbot)
    registry.load_handwritten(kernels)
    return [kernel_entry_to_dict(entry) for entry in registry]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the merged GPUBlind registry")
    parser.add_argument("--mined", type=Path, required=True)
    parser.add_argument("--kernelbot", type=Path, default=None)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = build_registry(args.mined, args.kernelbot, args.kernels, args.profiles, args.mock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
