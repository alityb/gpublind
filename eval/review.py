from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual review for reasoning quality")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    return parser.parse_args(argv)


def iter_result_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/level_*/*.json"))


def load_latest_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload[-1], payload
    return payload, None


def load_kernel_metadata(kernels_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for meta_path in kernels_dir.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["code"] = (meta_path.parent / "kernel.cu").read_text(encoding="utf-8")
        metadata[str(meta["id"])] = meta
    return metadata


def mentions_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def compute_correct_reasoning(result: dict[str, Any], meta: dict[str, Any]) -> bool:
    reasoning = str(result.get("parsed_reasoning") or result.get("raw_response") or "")
    must_mention = list(meta.get("reasoning_rubric", {}).get("must_mention", []))
    must_not = list(meta.get("reasoning_rubric", {}).get("must_not_cite_as_primary", []))
    first_sentence = reasoning.split(".", 1)[0].lower()
    return bool(
        result.get("correct") is True
        and result.get("cited_misleading_signal") is False
        and mentions_any(reasoning, must_mention)
        and not any(term.lower() in first_sentence for term in must_not)
    )


def persist_result(path: Path, single: dict[str, Any], collection: list[dict[str, Any]] | None) -> None:
    if collection is not None:
        collection[-1] = single
        path.write_text(json.dumps(collection, indent=2), encoding="utf-8")
        return
    path.write_text(json.dumps(single, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = load_kernel_metadata(args.kernels)
    for path in iter_result_paths(args.results):
        result, collection = load_latest_result(path)
        if result.get("cited_misleading_signal") is not None:
            continue
        meta = metadata.get(str(result.get("kernel_id")))
        if meta is None:
            continue
        print(f"\n=== {result['model']} L{result['level']} {result['kernel_id']} ===")
        print(f"True bottleneck: {meta['true_bottleneck']}")
        print(f"Misleading signal: {meta['misleading_signal']}")
        print("\nKernel code:\n")
        print(meta["code"])
        print("\nModel response:\n")
        print(result.get("raw_response", ""))
        answer = input("Did the model cite the misleading signal as primary cause? [y/n/skip] ").strip().lower()
        if answer not in {"y", "n"}:
            continue
        result["cited_misleading_signal"] = answer == "y"
        result["correct_reasoning"] = compute_correct_reasoning(result, meta)
        persist_result(path, result, collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
