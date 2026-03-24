from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from corpus import CorpusEntry, load_corpus
from eval.conditions import CONDITIONS, render_condition
from eval.judge import judge_response

MODEL_CONFIG = {
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gpt-5.4": "openai/gpt-5.4",
    "mock": "mock",
}

LABEL_RE = re.compile(r"BOTTLENECK:\s*([A-Za-z\-]+)", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)
REASON_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPUBlind v2 evaluation")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--conditions", default="0,1,2,3")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--output", type=Path, default=Path("results/v2"))
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4-6")
    return parser.parse_args(argv)


def parse_conditions(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def result_path(base: Path, model: str, condition: int, kernel_id: str) -> Path:
    return base / model / f"C{condition}" / f"{kernel_id}.json"


def parse_response(text: str) -> tuple[str, str, str]:
    label = LABEL_RE.search(text)
    confidence = CONF_RE.search(text)
    reasoning = REASON_RE.search(text)
    return (
        label.group(1).strip().lower() if label else "parse_error",
        confidence.group(1).strip().upper() if confidence else "LOW",
        reasoning.group(1).strip() if reasoning else "",
    )


def mock_response(entry: CorpusEntry, condition: int, prior_result: dict[str, Any] | None = None) -> str:
    if condition == 0:
        label = "memory-bound" if entry.true_bottleneck != "compute-bound" else "compute-bound"
    elif condition == 1:
        label = "latency-bound" if entry.profile["dram_bw_utilization_pct"] < 10.0 and entry.profile["stall_long_scoreboard_pct"] > 30.0 else entry.true_bottleneck
    elif condition in {2, 3}:
        label = entry.true_bottleneck
    elif condition == 4:
        label = entry.true_bottleneck
    else:
        label = "parse_error"
    confidence = "HIGH" if label == entry.true_bottleneck else "LOW"
    reasoning = entry.correct_explanation or f"The kernel is {label}."
    if condition == 4 and prior_result is not None:
        reasoning = f"I am revising the earlier diagnosis because {reasoning.lower()}"
    return f"BOTTLENECK: {label}\nCONFIDENCE: {confidence}\nREASONING: {reasoning}"


def call_model(model_name: str, prompt: dict[str, str], *, mock: bool, entry: CorpusEntry, condition: int, prior_result: dict[str, Any] | None) -> str:
    if mock or model_name == "mock":
        return mock_response(entry, condition, prior_result)
    from litellm import completion
    from litellm.exceptions import InternalServerError, RateLimitError

    attempt = 0
    while True:
        try:
            response = completion(
                model=MODEL_CONFIG[model_name],
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            )
            time.sleep(0.5)
            return str(response.choices[0].message.content)
        except RateLimitError:
            time.sleep(min(2**attempt, 60))
            attempt += 1
        except InternalServerError:
            time.sleep(min(2**attempt, 60))
            attempt += 1


def needs_judge(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("judge") is None and payload.get("predicted_label") != "api_error"


def judge_existing(path: Path, entry: CorpusEntry, args: argparse.Namespace) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["judge"] = judge_response(entry, payload.get("raw_response", ""), judge_model=args.judge_model, mock=args.mock)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_api_keys(model: str, judge: bool, judge_model: str, mock: bool) -> int:
    if mock or model == "mock":
        return 0
    if "claude" in model and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    if "gpt" in model and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    if judge and "claude" in judge_model and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set for judge", file=sys.stderr)
        return 1
    if judge and ("gpt" in judge_model or "openai" in judge_model) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set for judge", file=sys.stderr)
        return 1
    return 0


def evaluate(entry: CorpusEntry, model: str, condition: int, args: argparse.Namespace) -> dict[str, Any] | None:
    path = result_path(args.output, model, condition, entry.id)
    if condition == 4:
        prior_path = result_path(args.output, model, 2, entry.id)
        if not prior_path.exists():
            return None
        prior_result = json.loads(prior_path.read_text(encoding="utf-8"))
        if bool(prior_result.get("correct")):
            return None
    else:
        prior_result = None
    if args.resume and path.exists() and not (args.judge and needs_judge(path)):
        return None
    if args.resume and args.judge and needs_judge(path):
        judge_existing(path, entry, args)
        return None
    prompt = render_condition(entry, condition, prior_result=prior_result)
    raw_response = call_model(model, prompt, mock=args.mock, entry=entry, condition=condition, prior_result=prior_result)
    predicted_label, confidence, reasoning = parse_response(raw_response)
    payload = {
        "kernel_id": entry.id,
        "model": model,
        "condition": f"C{condition}",
        "condition_name": CONDITIONS[condition].name,
        "true_bottleneck": entry.true_bottleneck,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "reasoning": reasoning,
        "correct": predicted_label == entry.true_bottleneck,
        "raw_response": raw_response,
        "prompt": prompt,
    }
    if args.judge:
        payload["judge"] = judge_response(entry, raw_response, judge_model=args.judge_model, mock=args.mock)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[{model}] C{condition} {entry.id} correct={payload['correct']}", file=sys.stderr)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failure = ensure_api_keys(args.model, args.judge, args.judge_model, args.mock)
    if failure:
        return failure
    entries = load_corpus(args.kernels, min_confidence=args.min_confidence)
    if args.verified_only:
        entries = [
            entry
            for entry in entries
            if entry.profile.get("verification", {}).get("consensus") == entry.true_bottleneck
        ]
    conditions = parse_conditions(args.conditions)
    for entry in entries:
        for condition in conditions:
            evaluate(entry, args.model, condition, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
