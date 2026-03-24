from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from eval.run_eval import MODEL_CONFIG, build_registry, parse_filter_args

CONTAMINATION_SYSTEM = """You are evaluating whether a GPU kernel looks like something the model may have seen in training or public benchmarks.
Answer conservatively. If unsure, say NO and SOURCE: UNKNOWN."""

CONTAMINATION_PROMPT = """Here is a CUDA kernel:

{kernel_code}

Have you seen this kernel or a very similar one before?
Is this from a known benchmark, tutorial, or codebase?
Answer YES or NO, then briefly explain.

SEEN_BEFORE: <YES or NO>
SOURCE: <where you think it's from, or UNKNOWN>"""

SEEN_RE = re.compile(r"SEEN_BEFORE:\s*(YES|NO)", re.IGNORECASE)
SOURCE_RE = re.compile(r"SOURCE:\s*(.+)", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPUBlind contamination test")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG.keys()), required=True)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/contamination"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-profiles", action="store_true")
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    return parser.parse_args(argv)


def parse_contamination_response(text: str) -> dict[str, Any]:
    seen_match = SEEN_RE.search(text)
    source_match = SOURCE_RE.search(text)
    seen = seen_match.group(1).strip().upper() if seen_match else "NO"
    source = source_match.group(1).strip() if source_match else "UNKNOWN"
    return {"seen_before": seen == "YES", "source_guess": source}


def mock_contamination_result(entry: Any) -> dict[str, Any]:
    if entry.id.startswith("kernelbench_"):
        return {"seen_before": True, "source_guess": "KernelBench"}
    if entry.id.startswith("kernelbot_"):
        return {"seen_before": True, "source_guess": "GPU MODE KernelBot"}
    if entry.id.startswith("sakana_"):
        return {"seen_before": True, "source_guess": "SakanaAI AI-CUDA-Engineer-Archive"}
    return {"seen_before": False, "source_guess": "UNKNOWN"}


def call_contamination_model(
    model_name: str,
    kernel_code: str,
    mock: bool,
    entry: Any,
    completion_fn: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, dict[str, Any]]:
    if mock:
        result = mock_contamination_result(entry)
        raw = f"SEEN_BEFORE: {'YES' if result['seen_before'] else 'NO'}\nSOURCE: {result['source_guess']}"
        return raw, result

    from litellm import completion
    from litellm.exceptions import InternalServerError, RateLimitError

    resolved_completion = completion_fn or completion
    prompt = CONTAMINATION_PROMPT.format(kernel_code=kernel_code)
    attempt = 0
    while True:
        try:
            response = resolved_completion(
                model=MODEL_CONFIG[model_name],
                messages=[
                    {"role": "system", "content": CONTAMINATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = str(response.choices[0].message.content)
            sleep_fn(0.5)
            return raw, parse_contamination_response(raw)
        except RateLimitError:
            wait_seconds = float(min(2**attempt, 60))
            sleep_fn(wait_seconds)
            attempt += 1
        except InternalServerError:
            wait_seconds = float(min(2**attempt, 60))
            sleep_fn(wait_seconds)
            attempt += 1
        except Exception as exc:
            message = str(exc).lower()
            if "auth" in message or "api key" in message:
                raise
            wait_seconds = float(min(2**attempt, 60))
            sleep_fn(wait_seconds)
            attempt += 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mock:
        args.mock_profiles = True
    else:
        import os

        required = {"claude": "ANTHROPIC_API_KEY", "gpt": "OPENAI_API_KEY"}
        for prefix, env_var in required.items():
            if prefix in args.model and not os.environ.get(env_var):
                print(f"ERROR: {env_var} not set. Export it before running.", file=sys.stderr)
                return 1

    registry = build_registry(args)
    filters = parse_filter_args(args.filter)
    entries = registry.filter(
        source=filters.get("source"),
        category=filters.get("category"),
        difficulty=filters.get("difficulty"),
        true_bottleneck=filters.get("true_bottleneck"),
        ground_truth_verified=True,
        confidence=args.min_confidence,
    )
    if "id" in filters:
        entries = [entry for entry in entries if entry.id == filters["id"]]

    for entry in entries:
        raw, parsed = call_contamination_model(args.model, entry.code, args.mock, entry)
        payload = {
            "kernel_id": entry.id,
            "model": args.model,
            "seen_before": parsed["seen_before"],
            "source_guess": parsed["source_guess"],
            "raw_response": raw,
        }
        output_path = args.output / args.model / f"{entry.id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{args.model}] contamination kernel_id={entry.id} seen_before={parsed['seen_before']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
