from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from eval.prompts import PROMPTS, VALID_LABELS
from registry import KernelRegistry, SEVERITY

MODEL_CONFIG = {
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "deepseek-v3": "deepseek/deepseek-chat",
    "qwen2.5-coder": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "llama-3.1-8b": "openrouter/meta-llama/llama-3.1-8b-instruct",
}
OPPOSITE_BOTTLENECK: dict[str, str] = {
    "memory-bound": "compute-bound",
    "compute-bound": "memory-bound",
    "latency-bound": "memory-bound",
    "occupancy-limited": "compute-bound",
    "register-spill": "memory-bound",
}
COST_PER_1K_TOKENS = {
    "gpt-4o": 0.005,
    "claude-sonnet-4-6": 0.003,
    "deepseek-v3": 0.001,
    "qwen2.5-coder": 0.001,
    "llama-3.1-8b": 0.001,
}

BOTTLENECK_RE = re.compile(r"BOTTLENECK:\s*([A-Za-z\-]+)", re.IGNORECASE)
REASONING_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPUBlind evaluation")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG.keys()), required=True)
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    return parser.parse_args(argv)


def parse_filter_args(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_filters:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def build_registry(args: argparse.Namespace) -> KernelRegistry:
    registry = KernelRegistry(profile_dir=args.profiles, mock=args.mock)
    registry.load_mined(args.mined)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    return registry


def framing_label(label: str) -> str:
    return label


def wrong_label_for(true_label: str) -> str:
    return OPPOSITE_BOTTLENECK.get(true_label, "compute-bound")


def prompt_kwargs(entry: Any) -> dict[str, Any]:
    return {
        "kernel_code": entry.code,
        "latency_ms": round(float(entry.ncu_profile.raw.get("latency_ms", 0.0)), 3),
        "occupancy_pct": round(entry.ncu_profile.achieved_occupancy * 100.0, 2),
        "load_eff_pct": round(entry.ncu_profile.global_load_efficiency * 100.0, 2),
        "dram_bw_pct": round(entry.ncu_profile.dram_bw_utilization * 100.0, 2),
        "ncu_json": json.dumps(entry.ncu_profile.raw, indent=2, sort_keys=True),
        "wrong_bottleneck": wrong_label_for(entry.true_bottleneck),
        "correct_bottleneck": framing_label(entry.true_bottleneck),
    }


def parse_response(text: str) -> tuple[str, str]:
    label_match = BOTTLENECK_RE.search(text)
    reasoning_match = REASONING_RE.search(text)
    label = label_match.group(1).strip().lower() if label_match else "parse_error"
    if label not in VALID_LABELS:
        label = "parse_error"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return label, reasoning


def score_prediction(true_label: str, predicted_label: str, wrong_bottleneck: str | None = None) -> dict[str, Any]:
    return {
        "correct": predicted_label == true_label,
        "severity": SEVERITY.get((true_label, predicted_label), 0),
        "fell_for_adversarial": predicted_label == wrong_bottleneck if wrong_bottleneck is not None else None,
    }


def mock_completion(entry: Any, level: int) -> str:
    predicted = entry.true_bottleneck if level in {2, 3, 5} else next(
        label for label in VALID_LABELS if label != entry.true_bottleneck
    )
    return f"BOTTLENECK: {predicted}\nREASONING: Mock response for {entry.id} at level {level}."


def is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "ratelimit" in name or "rate_limit" in name


def is_retryable_api_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "apierror" in name or "timeout" in name or isinstance(exc, TimeoutError)


def resolve_completion_fn() -> Callable[..., Any]:
    from litellm import completion

    return completion


def call_model(
    model_name: str,
    rendered_prompt: dict[str, str],
    mock: bool,
    entry: Any,
    level: int,
    completion_fn: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, bool]:
    if mock:
        return mock_completion(entry, level), False
    try:
        completion = completion_fn or resolve_completion_fn()
    except Exception as exc:
        print(
            f"[{model_name}] level={level} kernel_id={entry.id} unexpected API error: {exc}",
            file=sys.stderr,
        )
        return str(exc), True
    messages = [
        {"role": "system", "content": rendered_prompt["system"]},
        {"role": "user", "content": rendered_prompt["user"]},
    ]
    rate_limit_retries = 0
    api_retries = 0
    while True:
        try:
            response = completion(model=MODEL_CONFIG[model_name], messages=messages)
            sleep_fn(0.5)
            return str(response.choices[0].message.content), False
        except Exception as exc:  # pragma: no cover - branch selection is tested by exception class name
            if is_rate_limit_error(exc):
                if rate_limit_retries >= 4:
                    print(
                        f"[{model_name}] level={level} kernel_id={entry.id} rate limit exhausted: {exc}",
                        file=sys.stderr,
                    )
                    return str(exc), True
                wait_seconds = min(2 ** (rate_limit_retries + 1), 32)
                rate_limit_retries += 1
                print(
                    f"[{model_name}] level={level} kernel_id={entry.id} rate limited, retrying in {wait_seconds}s",
                    file=sys.stderr,
                )
                sleep_fn(float(wait_seconds))
                continue
            if is_retryable_api_error(exc):
                if api_retries >= 3:
                    print(
                        f"[{model_name}] level={level} kernel_id={entry.id} API retry exhausted: {exc}",
                        file=sys.stderr,
                    )
                    return str(exc), True
                api_retries += 1
                print(
                    f"[{model_name}] level={level} kernel_id={entry.id} API error, retrying in 5s: {exc}",
                    file=sys.stderr,
                )
                sleep_fn(5.0)
                continue
            print(
                f"[{model_name}] level={level} kernel_id={entry.id} unexpected API error: {exc}",
                file=sys.stderr,
            )
            return str(exc), True


def existing_result_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def append_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            existing.append(payload)
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
        path.write_text(json.dumps([existing, payload], indent=2), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_entry(
    entry: Any,
    model_name: str,
    level: int,
    output_dir: Path,
    mock: bool,
    progress_index: int,
    progress_total: int,
) -> None:
    result_path = output_dir / model_name / f"level_{level}" / f"{entry.id}.json"
    rendered = PROMPTS[level].render(**prompt_kwargs(entry))
    raw_response, api_error = call_model(model_name, rendered, mock, entry, level)
    if api_error:
        predicted_label = "api_error"
        reasoning = raw_response
    else:
        predicted_label, reasoning = parse_response(raw_response)
    wrong_bottleneck = wrong_label_for(entry.true_bottleneck)
    score = score_prediction(entry.true_bottleneck, predicted_label, wrong_bottleneck if level == 4 else None)
    payload = {
        "kernel_id": entry.id,
        "model": model_name,
        "level": level,
        "true_bottleneck": entry.true_bottleneck,
        "predicted_label": predicted_label,
        "correct": score["correct"],
        "severity": score["severity"],
        "fell_for_adversarial": score["fell_for_adversarial"],
        "cited_misleading_signal": None,
        "raw_response": raw_response,
        "parsed_reasoning": reasoning,
        "prompt_rendered": rendered,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_result(result_path, payload)
    print(
        f"[{model_name}] level={level} kernel_id={entry.id} correct={score['correct']} ({progress_index}/{progress_total})",
        file=sys.stderr,
    )


def parse_levels(raw_levels: str) -> list[int]:
    return [int(level.strip()) for level in raw_levels.split(",") if level.strip()]


def build_tasks(entries: list[Any], levels: list[int], output_dir: Path, model_name: str) -> tuple[list[tuple[Any, int]], int]:
    tasks: list[tuple[Any, int]] = []
    existing_results = 0
    for entry in entries:
        for level in levels:
            result_path = output_dir / model_name / f"level_{level}" / f"{entry.id}.json"
            if existing_result_exists(result_path):
                existing_results += 1
            tasks.append((entry, level))
    return tasks, existing_results


def print_dry_run_summary(
    args: argparse.Namespace,
    entries: list[Any],
    levels: list[int],
    tasks: list[tuple[Any, int]],
    existing_results: int,
) -> None:
    estimated_tokens = len(tasks) * 500
    estimated_cost = (estimated_tokens / 1000.0) * COST_PER_1K_TOKENS[args.model]
    kernel_scope = "all kernels" if args.include_unverified else "verified only"
    print(f"Model: {args.model}")
    print(f"Levels: {','.join(str(level) for level in levels)}")
    print(f"Kernels: {len(entries)} ({kernel_scope})")
    print(f"Total calls: {len(tasks)}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Existing results: {existing_results} ({existing_results} would be skipped with --resume)")
    print("Planned combinations:")
    for entry, level in tasks:
        print(f"  level={level} kernel_id={entry.id}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = build_registry(args)
    levels = parse_levels(args.levels)
    filters = parse_filter_args(args.filter)
    entries = registry.filter(
        source=filters.get("source"),
        category=filters.get("category"),
        difficulty=filters.get("difficulty"),
        true_bottleneck=filters.get("true_bottleneck"),
        ground_truth_verified=None if args.include_unverified else True,
    )
    all_tasks, existing_results = build_tasks(entries, levels, args.output, args.model)
    tasks = [task for task in all_tasks if not (args.resume and existing_result_exists(args.output / args.model / f"level_{task[1]}" / f"{task[0].id}.json"))]
    if args.dry_run:
        print_dry_run_summary(args, entries, levels, tasks, existing_results)
        return 0
    total_tasks = len(tasks)
    for progress_index, (entry, level) in enumerate(tasks, start=1):
        evaluate_entry(entry, args.model, level, args.output, args.mock, progress_index, total_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
