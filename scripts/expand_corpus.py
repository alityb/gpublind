from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


HARDWARE = {
    "name": "NVIDIA A10G",
    "peak_bw_gbps": 496.0,
    "peak_flops_tflops": 30.77,
    "ridge_point_flop_per_byte": 62.07,
}

PLACEHOLDER_PROFILE = {
    "needs_profiling": True,
    "arithmetic_intensity_flop_per_byte": -1,
    "achieved_occupancy_pct": -1,
    "dram_bw_utilization_pct": -1,
    "stall_long_scoreboard_pct": -1,
    "stall_memory_pct": -1,
    "global_load_efficiency_pct": -1,
    "l2_hit_rate_pct": -1,
    "register_count_per_thread": -1,
    "hardware": HARDWARE,
    "verification": {"confidence": "unverified"},
}

KERNEL_RE = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)")
KERNEL_SIG_RE = re.compile(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")
MAIN_RE = re.compile(r"int\s+main\s*\([^)]*\)\s*\{.*\}\s*$", re.DOTALL)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand GPUBlind v2 corpus from mined JSONL sources")
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--skip-log", type=Path, default=Path("scripts/expand_corpus_skipped.txt"))
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_ground_truth_verified(entry: dict[str, Any]) -> bool:
    return bool(dict(entry.get("ncu_profile", {})).get("raw", {}).get("ground_truth_verified", False))


def derive_rubric(true_bottleneck: str, misleading_signal: str) -> dict[str, list[str]]:
    misleading = misleading_signal.lower()
    must_not: list[str]
    if "compute" in misleading:
        must_not = ["FLOP count", "arithmetic ops"]
    elif "memory" in misleading:
        must_not = ["global memory", "bandwidth"]
    else:
        must_not = [misleading_signal] if misleading_signal else []
    must_cite_map = {
        "latency-bound": ["stall_long", "long scoreboard", "dependency"],
        "memory-bound": ["dram_bw", "global load efficiency", "cache miss"],
        "compute-bound": ["arithmetic intensity", "FLOP", "compute throughput"],
        "occupancy-limited": ["occupancy", "blocks per SM"],
        "register-spill": ["register", "local memory", "spill"],
    }
    return {
        "must_cite_one_of": must_cite_map.get(true_bottleneck, []),
        "must_not_cite_as_primary": must_not,
    }


def is_standalone_cuda(code: str) -> bool:
    lowered = code.lower()
    return "__global__" in code and "int main" in lowered and "cudamalloc" in lowered


def strip_existing_main(code: str) -> str:
    return MAIN_RE.sub("", code).strip()


def launch_arguments(signature: str) -> list[str] | None:
    params = [param.strip() for param in signature.split(",") if param.strip()]
    args: list[str] = []
    pointer_pool = ["d_a", "d_b", "d_out"]
    pointer_index = 0
    for param in params:
        lowered = param.lower()
        if "*" in param:
            if pointer_index >= len(pointer_pool):
                return None
            args.append(pointer_pool[pointer_index])
            pointer_index += 1
        elif "int" in lowered and "n" in lowered:
            args.append("n")
        else:
            return None
    return args


def build_harness(code: str) -> tuple[str, str] | None:
    cleaned = strip_existing_main(code)
    match = KERNEL_SIG_RE.search(cleaned)
    if not match:
        return None
    kernel_name = match.group(1)
    args = launch_arguments(match.group(2))
    if args is None:
        return None
    harness = f"""#include <cuda_runtime.h>
#include <cstdio>

// ===== KERNEL CODE START =====
{cleaned}
// ===== KERNEL CODE END =====

int main() {{
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 1, bytes);

    {kernel_name}<<<(n + 255) / 256, 256>>>({", ".join(args)});
    cudaDeviceSynchronize();

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\\n", h_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}}
"""
    return harness, kernel_name


def write_entry(entry: dict[str, Any], output: Path, source_name: str) -> bool:
    code = str(entry.get("code") or "").strip()
    if not code:
        return False
    standalone = is_standalone_cuda(code)
    if standalone:
        final_code = code
    else:
        harness = build_harness(code)
        if harness is None:
            return False
        final_code, _ = harness
    out_dir = output / str(entry["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.cu").write_text(final_code, encoding="utf-8")
    meta = {
        "id": entry["id"],
        "source": source_name,
        "true_bottleneck": entry["true_bottleneck"],
        "misleading_signal": entry.get("misleading_signal", ""),
        "correct_explanation": str(entry.get("misleading_signal", "")),
        "difficulty": entry.get("difficulty", "medium"),
        "category": entry.get("true_bottleneck", "unknown"),
        "reasoning_rubric": derive_rubric(str(entry.get("true_bottleneck", "")), str(entry.get("misleading_signal", ""))),
        "hardware": "A10G",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "profile.json").write_text(json.dumps(PLACEHOLDER_PROFILE, indent=2) + "\n", encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    written = 0
    datasets = [
        ("sakana", load_jsonl(args.mined)),
        ("kernelbot", load_jsonl(args.kernelbot)),
    ]
    for source_name, rows in datasets:
        for entry in rows:
            if source_name == "sakana" and not is_ground_truth_verified(entry):
                continue
            if not write_entry(entry, args.output, source_name):
                skipped.append(str(entry.get("id", "unknown")))
                continue
            written += 1
    args.skip_log.write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")
    print(f"Kernels written: {written}")
    print(f"Kernels skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
