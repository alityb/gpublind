from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus.schema import load_corpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corpus stats for GPUBlind v2")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    all_entries = load_corpus(args.kernels, min_confidence="low")
    entries = load_corpus(args.kernels, min_confidence=args.min_confidence)
    by_category = Counter(entry.category for entry in entries)
    by_source = Counter(entry.source for entry in all_entries)
    by_difficulty = Counter(entry.difficulty for entry in all_entries)
    by_confidence = Counter(entry.confidence for entry in all_entries)
    print(f"Total kernels: {len(all_entries)}")
    print(f"Verified (confidence>={args.min_confidence}): {len(entries)}")
    print(f"By category: {dict(sorted(by_category.items()))}")
    print(f"By source: {dict(sorted(by_source.items()))}")
    print(f"By difficulty: {dict(sorted(by_difficulty.items()))}")
    print(f"By confidence: {dict(sorted(by_confidence.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
