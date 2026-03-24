from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus.verify import verify_profile


PROFILE_MAP = {
    "arithmetic_intensity_flop_per_byte": ("arithmetic_intensity", 1.0),
    "achieved_occupancy_pct": ("achieved_occupancy", 100.0),
    "dram_bw_utilization_pct": ("dram_bw_utilization", 100.0),
    "stall_long_scoreboard_pct": ("stall_long_sb_pct", 100.0),
    "stall_memory_pct": ("stall_mem_pct", 100.0),
    "global_load_efficiency_pct": ("global_load_efficiency", 100.0),
    "l2_hit_rate_pct": ("l2_hit_rate", 100.0),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate GPUBlind v1 handwritten kernels into corpus/ schema")
    parser.add_argument("--source", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--output", type=Path, default=Path("corpus/kernels"))
    return parser.parse_args(argv)


def convert_meta(meta: dict[str, Any]) -> dict[str, Any]:
    rubric = meta.get("reasoning_rubric", {})
    must_cite = rubric.get("must_cite_one_of") or rubric.get("must_mention") or []
    return {
        "id": meta["id"],
        "source": "handwritten",
        "true_bottleneck": meta["true_bottleneck"],
        "misleading_signal": meta.get("misleading_signal", ""),
        "correct_explanation": meta.get("correct_explanation", ""),
        "difficulty": meta.get("difficulty", "medium"),
        "category": meta.get("true_bottleneck", meta.get("category", "unknown")),
        "reasoning_rubric": {
            "must_cite_one_of": must_cite,
            "must_not_cite_as_primary": rubric.get("must_not_cite_as_primary", []),
        },
        "hardware": "A10G",
    }


def convert_profile(profile: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for output_key, (source_key, scale) in PROFILE_MAP.items():
        converted[output_key] = round(float(profile.get(source_key, 0.0)) * scale, 4 if scale == 1.0 else 2)
    converted["register_count_per_thread"] = int(profile.get("register_count", 0))
    raw = profile.get("raw", {})
    converted["gpu_time_us"] = round(float(raw.get("gpu__time_duration.sum", 0.0)) / 1000.0, 3)
    roof = raw.get("roof", {})
    converted["hardware"] = {
        "name": roof.get("gpu_name", "NVIDIA A10G"),
        "peak_bw_gbps": round(float(roof.get("peak_bw_tbps", 0.496)) * 1000.0, 3),
        "peak_flops_tflops": round(float(roof.get("peak_flops_tflops", 30.77)), 3),
        "ridge_point_flop_per_byte": round(float(raw.get("ridge_point", 62.07)), 3),
    }
    converted["verification"] = verify_profile(converted)
    return converted


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for kernel_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        meta_path = kernel_dir / "meta.json"
        kernel_path = kernel_dir / "kernel.cu"
        if not meta_path.exists() or not kernel_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        profile_path = args.profiles / f"{meta['id']}.json"
        if not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        out_dir = args.output / str(meta["id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kernel_path, out_dir / "kernel.cu")
        (out_dir / "meta.json").write_text(json.dumps(convert_meta(meta), indent=2) + "\n", encoding="utf-8")
        (out_dir / "profile.json").write_text(json.dumps(convert_profile(profile), indent=2) + "\n", encoding="utf-8")
        migrated += 1
    print(f"Migrated {migrated} kernels into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
