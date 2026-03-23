from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from eval.analyze_results import infer_result_metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove api_error result payloads and flatten duplicate lists")
    parser.add_argument("--results", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def is_valid_payload(payload: dict) -> bool:
    if payload.get("predicted_label") == "api_error":
        return False
    if payload.get("question_format") == "fix":
        return bool(payload.get("suggested_change") or payload.get("raw_response"))
    return bool(payload.get("predicted_label") is not None or payload.get("raw_response"))


def payload_timestamp(payload: dict, path: Path) -> float:
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def best_payload(payload: dict | list, path: Path) -> dict | None:
    if isinstance(payload, list):
        valid = [item for item in payload if is_valid_payload(item)]
        if not valid:
            return None
        return max(valid, key=lambda item: payload_timestamp(item, path))
    if isinstance(payload, dict) and is_valid_payload(payload):
        return payload
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cleaned = 0
    deleted = 0
    candidates: list[tuple[tuple[object, ...], Path, dict]] = []
    for path in sorted(args.results.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = best_payload(payload, path)
        if chosen is None:
            path.unlink(missing_ok=True)
            deleted += 1
            print(f"Deleted {path}: no valid payload")
            continue
        if isinstance(payload, list):
            path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
            cleaned += 1
            print(f"Cleaned {path}: kept most recent valid entry")
        metadata = infer_result_metadata(path, args.results)
        combo = (
            str(chosen.get("model", metadata["model"])),
            int(chosen.get("trial", metadata.get("trial", 1))),
            int(chosen.get("level", metadata["level"])),
            str(chosen.get("question_format", metadata["question_format"])),
            str(chosen.get("kernel_id")),
        )
        candidates.append((combo, path, chosen))

    by_combo: dict[tuple[object, ...], tuple[Path, dict]] = {}
    for combo, path, payload in candidates:
        current = by_combo.get(combo)
        if current is None or payload_timestamp(payload, path) > payload_timestamp(current[1], current[0]):
            by_combo[combo] = (path, payload)

    keep_paths = {path for path, _ in by_combo.values()}
    for _, path, _ in candidates:
        if path in keep_paths:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
        print(f"Deleted {path}: duplicate combination")
    print(f"Summary: cleaned={cleaned}, deleted={deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
