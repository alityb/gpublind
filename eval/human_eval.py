from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus import load_corpus
from eval.conditions import render_condition
from eval.run_eval import parse_response, result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPUBlind v2 human evaluation")
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--condition", choices=["C0", "C2"], default="C0")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/human"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--filter", action="append", default=[])
    return parser.parse_args(argv)


def parse_filters(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_filters:
        if "=" in raw:
            key, value = raw.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    condition = int(args.condition[1:])
    entries = load_corpus(args.kernels, min_confidence=args.min_confidence)
    filters = parse_filters(args.filter)
    if "source" in filters:
        entries = [entry for entry in entries if entry.source == filters["source"]]
    for entry in entries:
        prompt = render_condition(entry, condition)
        print(f"=== {entry.id} ===")
        print(prompt["user"])
        start = time.time()
        label = input("BOTTLENECK: ").strip()
        confidence = input("CONFIDENCE: ").strip().upper()
        reasoning = input("REASONING: ").strip()
        elapsed = time.time() - start
        raw_response = f"BOTTLENECK: {label}\nCONFIDENCE: {confidence}\nREASONING: {reasoning}"
        predicted_label, parsed_confidence, parsed_reasoning = parse_response(raw_response)
        payload = {
            "kernel_id": entry.id,
            "model": args.evaluator,
            "condition": args.condition,
            "condition_name": condition,
            "true_bottleneck": entry.true_bottleneck,
            "predicted_label": predicted_label,
            "confidence": parsed_confidence,
            "reasoning": parsed_reasoning,
            "correct": predicted_label == entry.true_bottleneck,
            "time_taken_seconds": round(elapsed, 3),
            "raw_response": raw_response,
        }
        path = result_path(args.output, args.evaluator.replace(" ", "_"), condition, entry.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
