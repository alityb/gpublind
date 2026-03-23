from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from eval.prompts import render_prompt
from eval.run_eval import parse_filter_args, parse_response, prompt_kwargs
from registry import KernelRegistry, SEVERITY


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a human GPUBlind evaluation session")
    parser.add_argument("--evaluator-name", default="human")
    parser.add_argument("--output", type=Path, default=Path("results/human"))
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    return parser.parse_args(argv)


def result_path_for(output_dir: Path, evaluator_name: str, kernel_id: str) -> Path:
    if evaluator_name == "human":
        return output_dir / "level_1" / "label" / f"{kernel_id}.json"
    return output_dir / evaluator_name / "level_1" / "label" / f"{kernel_id}.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = KernelRegistry(profile_dir=args.profiles, mock=True)
    registry.load_mined(args.mined)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    filters = parse_filter_args(args.filter)
    entries = registry.filter(
        source=filters.get("source"),
        category=filters.get("category"),
        difficulty=filters.get("difficulty"),
        true_bottleneck=filters.get("true_bottleneck"),
        confidence=args.min_confidence,
    )
    for entry in entries:
        path = result_path_for(args.output, args.evaluator_name, entry.id)
        if path.exists():
            continue
        rendered = render_prompt(1, "label", **prompt_kwargs(entry))
        print(f"\n=== {entry.id} ===\n")
        print(rendered["user"])
        start = time.monotonic()
        label_line = input("\nBOTTLENECK: ").strip()
        reasoning = input("REASONING: ").strip()
        elapsed = time.monotonic() - start
        raw_response = f"BOTTLENECK: {label_line}\nREASONING: {reasoning}"
        predicted_label, parsed_reasoning = parse_response(raw_response)
        payload = {
            "kernel_id": entry.id,
            "model": "human",
            "trial": 1,
            "level": 1,
            "question_format": "label",
            "true_bottleneck": entry.true_bottleneck,
            "predicted_label": predicted_label,
            "correct": predicted_label == entry.true_bottleneck,
            "severity": SEVERITY.get((entry.true_bottleneck, predicted_label), 0),
            "fell_for_adversarial": None,
            "raw_response": raw_response,
            "parsed_reasoning": parsed_reasoning,
            "prompt_rendered": rendered,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluator_name": args.evaluator_name,
            "response_time_sec": round(elapsed, 3),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
