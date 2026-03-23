from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from eval.prompts import PROMPTS, QUESTION_FORMATS, VALID_LABELS, render_prompt
from registry import KernelRegistry, SEVERITY

MODEL_CONFIG = {
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gpt-5.4": "openai/gpt-5.4",
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
    "gpt-5.4": 0.005,
    "claude-sonnet-4-6": 0.003,
    "claude-opus-4-6": 0.003,
    "deepseek-v3": 0.001,
    "qwen2.5-coder": 0.001,
    "llama-3.1-8b": 0.001,
}

BOTTLENECK_RE = re.compile(r"BOTTLENECK:\s*([A-Za-z\-]+)", re.IGNORECASE)
REASONING_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)
MEMORY_BOUND_RE = re.compile(r"MEMORY_BOUND:\s*(YES|NO)", re.IGNORECASE)
ASSESSMENT_RE = re.compile(r"ASSESSMENT:\s*(AGREE|DISAGREE)", re.IGNORECASE)
CHANGE_RE = re.compile(r"CHANGE:\s*(.+)", re.IGNORECASE)
EXPECTED_IMPROVEMENT_RE = re.compile(r"EXPECTED_IMPROVEMENT:\s*(.+)", re.IGNORECASE)
RANK_RE = re.compile(r"RANK_([1-5]):\s*([A-Za-z\-]+)", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPUBlind evaluation")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG.keys()), required=True)
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument("--question-formats", default="label")
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-profiles", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--balance-min-class", type=int, default=5)
    parser.add_argument("--balance-seed", type=int, default=7)
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    args = parser.parse_args(argv)
    if args.mock:
        args.mock_profiles = True
        args.mock_llm = True
    if args.dry_run:
        args.mock_profiles = True
    return args


def parse_filter_args(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_filters:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_question_formats(raw_formats: str) -> list[str]:
    formats = [fmt.strip() for fmt in raw_formats.split(",") if fmt.strip()]
    unknown = [fmt for fmt in formats if fmt not in QUESTION_FORMATS]
    if unknown:
        raise ValueError(f"Unknown question formats: {', '.join(sorted(unknown))}")
    return formats or ["label"]


def build_registry(args: argparse.Namespace) -> KernelRegistry:
    registry = KernelRegistry(profile_dir=args.profiles, mock=args.mock_profiles)
    registry.load_mined(args.mined)
    registry.load_mined(args.kernelbench_compute)
    registry.load_mined(args.latency)
    registry.load_mined(args.register_spill)
    registry.load_kernelbot(args.kernelbot)
    registry.load_handwritten(args.kernels)
    return registry


def balance_entries(entries: list[Any], min_class_size: int, seed: int) -> list[Any]:
    grouped: dict[str, list[Any]] = {}
    for entry in entries:
        grouped.setdefault(entry.true_bottleneck, []).append(entry)
    eligible = {label: items for label, items in grouped.items() if len(items) >= min_class_size}
    if not eligible:
        return []
    target = min(len(items) for items in eligible.values())
    rng = random.Random(seed)
    balanced: list[Any] = []
    for label in sorted(eligible):
        items = list(eligible[label])
        rng.shuffle(items)
        balanced.extend(sorted(items[:target], key=lambda entry: entry.id))
    return sorted(balanced, key=lambda entry: (entry.true_bottleneck, entry.id))


def framing_label(label: str) -> str:
    return label


def wrong_label_for(true_label: str) -> str:
    return OPPOSITE_BOTTLENECK.get(true_label, "compute-bound")


def prompt_kwargs(entry: Any) -> dict[str, Any]:
    raw = entry.ncu_profile.raw
    total_flops = float(raw.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)) + float(
        raw.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
    )
    duration_ns = float(raw.get("gpu__time_duration.sum", 0.0))
    peak_tflops = float(raw.get("roof", {}).get("peak_flops_tflops", 0.0))
    achieved_tflops = (total_flops / max(duration_ns, 1.0)) * 1.0e-3
    compute_pct = round((achieved_tflops / peak_tflops) * 100.0, 2) if peak_tflops > 0.0 else 0.0
    return {
        "kernel_code": entry.code,
        "latency_ms": round(float(raw.get("latency_ms", 0.0)), 3),
        "occupancy_pct": round(entry.ncu_profile.achieved_occupancy * 100.0, 2),
        "load_eff_pct": round(entry.ncu_profile.global_load_efficiency * 100.0, 2),
        "dram_bw_pct": round(entry.ncu_profile.dram_bw_utilization * 100.0, 2),
        "compute_pct": compute_pct,
        "stall_long_pct": round(entry.ncu_profile.stall_long_sb_pct * 100.0, 2),
        "ncu_json": json.dumps(raw, indent=2, sort_keys=True),
        "wrong_bottleneck": wrong_label_for(entry.true_bottleneck),
        "correct_bottleneck": framing_label(entry.true_bottleneck),
        "correct_explanation": entry.correct_explanation or f"the kernel is {entry.true_bottleneck}",
    }


def parse_response(text: str) -> tuple[str, str]:
    label_match = BOTTLENECK_RE.search(text)
    reasoning_match = REASONING_RE.search(text)
    label = label_match.group(1).strip().lower() if label_match else "parse_error"
    if label not in VALID_LABELS:
        label = "parse_error"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return label, reasoning


def parse_yesno(response: str) -> str:
    match = MEMORY_BOUND_RE.search(response)
    if not match:
        return "parse_error"
    return "memory-bound" if match.group(1).strip().upper() == "YES" else "not-memory-bound"


def parse_rank(response: str) -> list[str]:
    ranks: dict[int, str] = {}
    for rank, label in RANK_RE.findall(response):
        normalized = label.strip().lower()
        if normalized in VALID_LABELS:
            ranks[int(rank)] = normalized
    return [ranks[idx] for idx in range(1, 6) if idx in ranks]


def parse_assessment(response: str) -> str:
    match = ASSESSMENT_RE.search(response)
    if not match:
        return "parse_error"
    return match.group(1).strip().upper()


def parse_fix(response: str) -> tuple[str, str]:
    change_match = CHANGE_RE.search(response)
    improvement_match = EXPECTED_IMPROVEMENT_RE.search(response)
    return (
        change_match.group(1).strip() if change_match else "",
        improvement_match.group(1).strip() if improvement_match else "",
    )


def score_prediction(true_label: str, predicted_label: str, wrong_bottleneck: str | None = None) -> dict[str, Any]:
    return {
        "correct": predicted_label == true_label,
        "severity": SEVERITY.get((true_label, predicted_label), 0),
        "fell_for_adversarial": predicted_label == wrong_bottleneck if wrong_bottleneck is not None else None,
    }


def score_memory_binary(true_label: str, predicted_label: str) -> dict[str, Any]:
    predicted_memory_bound = predicted_label == "memory-bound"
    correct = predicted_memory_bound == (true_label == "memory-bound")
    return {
        "correct": correct,
        "severity": 0 if correct else 1,
        "fell_for_adversarial": None,
    }


def score_assessment(assessment: str, expected_agreement: bool, fell_for_adversarial: bool | None = None) -> dict[str, Any]:
    correct = (assessment == "AGREE") == expected_agreement
    return {
        "correct": correct,
        "severity": 0 if correct else 1,
        "fell_for_adversarial": fell_for_adversarial,
    }


def mock_completion(entry: Any, level: int, question_format: str) -> str:
    predicted = entry.true_bottleneck if level in {2, 3, 5} else next(
        label for label in VALID_LABELS if label != entry.true_bottleneck
    )
    if question_format in {"label", "metrics_only"}:
        return f"BOTTLENECK: {predicted}\nREASONING: Mock response for {entry.id} at level {level}."
    if question_format == "yesno_memory":
        answer = "YES" if entry.true_bottleneck == "memory-bound" else "NO"
        return f"MEMORY_BOUND: {answer}\nREASONING: Mock response for {entry.id} at level {level}."
    if question_format == "rank":
        remaining = [label for label in VALID_LABELS if label != entry.true_bottleneck]
        ranking = [entry.true_bottleneck, *remaining]
        return (
            "\n".join(f"RANK_{idx}: {label}" for idx, label in enumerate(ranking, start=1))
            + f"\nREASONING: Mock response for {entry.id} at level {level}."
        )
    if question_format == "junior_wrong":
        answer = "AGREE" if entry.true_bottleneck == "memory-bound" else "DISAGREE"
        return f"ASSESSMENT: {answer}\nREASONING: Mock response for {entry.id} at level {level}."
    if question_format == "junior_right":
        return f"ASSESSMENT: AGREE\nREASONING: Mock response for {entry.id} at level {level}."
    if question_format == "fix":
        return (
            "CHANGE: Replace the dominant bottleneck with a more coalesced or less dependent access pattern.\n"
            "EXPECTED_IMPROVEMENT: The limiting utilization metric would improve because the primary stall source is reduced."
        )
    raise ValueError(f"Unsupported question format: {question_format}")


def resolve_completion_fn() -> Callable[..., Any]:
    from litellm import completion

    return completion


def call_model(
    model_name: str,
    rendered_prompt: dict[str, str],
    mock: bool,
    entry: Any,
    level: int,
    question_format: str,
    completion_fn: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, bool]:
    if mock:
        return mock_completion(entry, level, question_format), False

    from litellm.exceptions import InternalServerError, RateLimitError

    completion = completion_fn or resolve_completion_fn()
    messages = [
        {"role": "system", "content": rendered_prompt["system"]},
        {"role": "user", "content": rendered_prompt["user"]},
    ]
    resolved_model = MODEL_CONFIG.get(model_name, model_name)
    attempt = 0
    while True:
        try:
            response = completion(model=resolved_model, messages=messages)
            sleep_fn(0.5)
            return str(response.choices[0].message.content), False
        except RateLimitError:
            wait_seconds = float(min(2**attempt, 60))
            print(f"[{model_name}] rate limited, retrying in {wait_seconds:g}s", file=sys.stderr)
            sleep_fn(wait_seconds)
            attempt += 1
        except InternalServerError:
            wait_seconds = float(min(2**attempt, 60))
            print(f"[{model_name}] server error, retrying in {wait_seconds:g}s", file=sys.stderr)
            sleep_fn(wait_seconds)
            attempt += 1
        except Exception as exc:
            message = str(exc).lower()
            if "auth" in message or "api key" in message:
                print(f"[{model_name}] auth error, aborting: {exc}", file=sys.stderr)
                return str(exc), True
            wait_seconds = float(min(2**attempt, 60))
            if "server disconnect" in message or "server disconnected" in message:
                print(f"[{model_name}] server disconnect, retrying in {wait_seconds:g}s: {exc}", file=sys.stderr)
            else:
                print(f"[{model_name}] unexpected error, retrying in {wait_seconds:g}s: {exc}", file=sys.stderr)
            sleep_fn(wait_seconds)
            attempt += 1


def existing_result_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def result_path_for(
    output_dir: Path,
    model_name: str,
    level: int,
    kernel_id: str,
    question_format: str,
    legacy_label_layout: bool,
    trial: int,
) -> Path:
    base = output_dir / model_name / f"trial_{trial}" / f"level_{level}"
    if legacy_label_layout and question_format == "label":
        return base / f"{kernel_id}.json"
    return base / question_format / f"{kernel_id}.json"


def load_existing_result(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def result_is_valid(payload: dict[str, Any] | list[dict[str, Any]] | None) -> bool:
    if payload is None:
        return False
    if isinstance(payload, list):
        return any(result_is_valid(item) for item in payload)
    if payload.get("predicted_label") == "api_error":
        return False
    question_format = payload.get("question_format", "label")
    if question_format == "fix":
        return bool(payload.get("suggested_change") or payload.get("raw_response"))
    if question_format == "junior_right":
        return payload.get("parsed_assessment") in {"AGREE", "DISAGREE"}
    return payload.get("predicted_label") is not None


def result_has_api_error(payload: dict[str, Any] | list[dict[str, Any]] | None) -> bool:
    if payload is None:
        return False
    if isinstance(payload, list):
        return any(result_has_api_error(item) for item in payload)
    return payload.get("predicted_label") == "api_error"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_entry(
    entry: Any,
    model_name: str,
    level: int,
    question_format: str,
    output_dir: Path,
    mock: bool,
    legacy_label_layout: bool,
    trial: int,
    progress_index: int,
    progress_total: int,
) -> None:
    result_path = result_path_for(output_dir, model_name, level, entry.id, question_format, legacy_label_layout, trial)
    rendered = render_prompt(level, question_format, **prompt_kwargs(entry))
    raw_response, api_error = call_model(model_name, rendered, mock, entry, level, question_format)
    predicted_label: str | None
    reasoning = ""
    parsed_ranking: list[str] | None = None
    parsed_assessment: str | None = None
    suggested_change: str | None = None
    expected_improvement: str | None = None
    if api_error:
        predicted_label = "api_error"
        reasoning = raw_response
        score = {"correct": False, "severity": 0, "fell_for_adversarial": None}
    elif question_format in {"label", "metrics_only"}:
        predicted_label, reasoning = parse_response(raw_response)
        wrong_bottleneck = wrong_label_for(entry.true_bottleneck)
        score = score_prediction(entry.true_bottleneck, predicted_label, wrong_bottleneck if level == 4 else None)
    elif question_format == "yesno_memory":
        predicted_label = parse_yesno(raw_response)
        reasoning = parse_response(raw_response)[1]
        score = score_memory_binary(entry.true_bottleneck, predicted_label)
    elif question_format == "rank":
        parsed_ranking = parse_rank(raw_response)
        predicted_label = parsed_ranking[0] if parsed_ranking else "parse_error"
        reasoning = parse_response(raw_response)[1]
        score = score_prediction(entry.true_bottleneck, predicted_label)
    elif question_format == "junior_wrong":
        parsed_assessment = parse_assessment(raw_response)
        predicted_label = "memory-bound" if parsed_assessment == "AGREE" else "not-memory-bound" if parsed_assessment == "DISAGREE" else "parse_error"
        reasoning = parse_response(raw_response)[1]
        fell_for_adversarial = parsed_assessment == "AGREE" and entry.true_bottleneck != "memory-bound"
        score = score_assessment(parsed_assessment, entry.true_bottleneck == "memory-bound", fell_for_adversarial)
    elif question_format == "junior_right":
        parsed_assessment = parse_assessment(raw_response)
        predicted_label = entry.true_bottleneck if parsed_assessment == "AGREE" else None
        reasoning = parse_response(raw_response)[1]
        score = score_assessment(parsed_assessment, True)
    elif question_format == "fix":
        predicted_label = None
        suggested_change, expected_improvement = parse_fix(raw_response)
        score = {"correct": None, "severity": None, "fell_for_adversarial": None}
    else:
        raise ValueError(f"Unsupported question format: {question_format}")
    payload = {
        "kernel_id": entry.id,
        "model": model_name,
        "trial": trial,
        "level": level,
        "question_format": question_format,
        "true_bottleneck": entry.true_bottleneck,
        "correct_explanation": entry.correct_explanation,
        "predicted_label": predicted_label,
        "correct": score["correct"],
        "severity": score["severity"],
        "fell_for_adversarial": score["fell_for_adversarial"],
        "cited_misleading_signal": None,
        "raw_response": raw_response,
        "parsed_reasoning": reasoning,
        "parsed_ranking": parsed_ranking,
        "parsed_assessment": parsed_assessment,
        "suggested_change": suggested_change,
        "expected_improvement": expected_improvement,
        "prompt_rendered": rendered,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_result(result_path, payload)
    print(
        f"[{model_name}] level={level} kernel_id={entry.id} correct={score['correct']} ({progress_index}/{progress_total})",
        file=sys.stderr,
    )


def parse_levels(raw_levels: str) -> list[int]:
    return [int(level.strip()) for level in raw_levels.split(",") if level.strip()]


def build_tasks(entries: list[Any], levels: list[int], output_dir: Path, model_name: str) -> tuple[list[tuple[Any, int]], int]:
    tasks: list[tuple[Any, int, str]] = []
    existing_results = 0
    legacy_label_layout = False
    for entry in entries:
        for level in levels:
            result_path = output_dir / model_name / f"level_{level}" / f"{entry.id}.json"
            if existing_result_exists(result_path):
                existing_results += 1
            tasks.append((entry, level, "label"))
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
    print(f"Trial: {args.trial}")
    print(f"Levels: {','.join(str(level) for level in levels)}")
    print(f"Kernels: {len(entries)} ({kernel_scope})")
    print(f"Total calls: {len(tasks)}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Existing results: {existing_results} ({existing_results} would be skipped with --resume)")
    print("Planned combinations:")
    for entry, level, question_format in tasks:
        print(f"  level={level} question_format={question_format} kernel_id={entry.id}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.mock or args.mock_llm or args.dry_run):
        import os

        required = {"claude": "ANTHROPIC_API_KEY", "gpt": "OPENAI_API_KEY"}
        model = args.model
        for prefix, env_var in required.items():
            if prefix in model and not os.environ.get(env_var):
                print(f"ERROR: {env_var} not set. Export it before running.", file=sys.stderr)
                return 1
    registry = build_registry(args)
    levels = parse_levels(args.levels)
    question_formats = parse_question_formats(args.question_formats)
    legacy_label_layout = question_formats == ["label"]
    filters = parse_filter_args(args.filter)
    entries = registry.filter(
        source=filters.get("source"),
        category=filters.get("category"),
        difficulty=filters.get("difficulty"),
        true_bottleneck=filters.get("true_bottleneck"),
        ground_truth_verified=None if args.include_unverified else True,
        confidence=args.min_confidence,
    )
    if "id" in filters:
        entries = [entry for entry in entries if entry.id == filters["id"]]
    if args.balance:
        entries = balance_entries(entries, args.balance_min_class, args.balance_seed)
        print(
            f"Balanced evaluation selected {len(entries)} kernels across eligible bottleneck classes "
            f"(min_class={args.balance_min_class}, seed={args.balance_seed}).",
            file=sys.stderr,
        )
    all_tasks: list[tuple[Any, int, str]] = []
    existing_results = 0
    for entry in entries:
        for level in levels:
            for question_format in question_formats:
                path = result_path_for(args.output, args.model, level, entry.id, question_format, legacy_label_layout, args.trial)
                if existing_result_exists(path):
                    existing_results += 1
                all_tasks.append((entry, level, question_format))
    tasks = [
        task
        for task in all_tasks
        if not (
            args.resume
            and result_is_valid(load_existing_result(result_path_for(args.output, args.model, task[1], task[0].id, task[2], legacy_label_layout, args.trial)))
        )
    ]
    if args.dry_run:
        print_dry_run_summary(args, entries, levels, tasks, existing_results)
        return 0
    total_tasks = len(tasks)
    for progress_index, (entry, level, question_format) in enumerate(tasks, start=1):
        path = result_path_for(args.output, args.model, level, entry.id, question_format, legacy_label_layout, args.trial)
        existing = load_existing_result(path)
        if result_has_api_error(existing):
            path.unlink(missing_ok=True)
        if args.resume and result_is_valid(existing):
            continue
        evaluate_entry(entry, args.model, level, question_format, args.output, args.mock_llm, legacy_label_layout, args.trial, progress_index, total_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
