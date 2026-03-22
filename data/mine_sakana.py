from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from registry.kernel_entry import KernelEntry, NCUProfile, kernel_entry_to_dict

DATASET_NAME = "SakanaAI/AI-CUDA-Engineer-Archive"
DEFAULT_OUTPUT = Path("data/mined_kernels.jsonl")
VALID_BOTTLENECKS = {
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "occupancy-limited",
    "register-spill",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine misleading kernels from SakanaAI archive")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=70)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def normalize_profile(raw_profile: dict[str, Any], hardware: str) -> NCUProfile:
    arithmetic_intensity = float(raw_profile.get("arithmetic_intensity", 0.0))
    ridge_point = float(raw_profile.get("ridge_point", 10.0))
    memory_bound = bool(raw_profile.get("memory_bound", arithmetic_intensity < ridge_point))
    compute_bound = bool(raw_profile.get("compute_bound", arithmetic_intensity >= ridge_point))
    stall_long = float(raw_profile.get("stall_long_sb_pct", raw_profile.get("stall_long_sb", 0.0)))
    stall_mem = float(raw_profile.get("stall_mem_pct", raw_profile.get("stall_memory_pct", 0.0)))
    stall_short = float(raw_profile.get("stall_short_sb_pct", raw_profile.get("stall_short_sb", 0.0)))
    dominant_stall_type = str(
        raw_profile.get(
            "dominant_stall_type",
            max(
                {
                    "long_scoreboard": stall_long,
                    "memory_dependency": stall_mem,
                    "short_scoreboard": stall_short,
                }.items(),
                key=lambda item: item[1],
            )[0],
        )
    )
    return NCUProfile(
        arithmetic_intensity=arithmetic_intensity,
        memory_bound=memory_bound,
        compute_bound=compute_bound,
        dominant_stall_type=dominant_stall_type,
        global_load_efficiency=float(raw_profile.get("global_load_efficiency", 1.0)),
        achieved_occupancy=float(raw_profile.get("achieved_occupancy", 1.0)),
        stall_long_sb_pct=stall_long,
        stall_mem_pct=stall_mem,
        register_count=int(raw_profile.get("register_count", 32)),
        l2_hit_rate=float(raw_profile.get("l2_hit_rate", 0.5)),
        dram_bw_utilization=float(raw_profile.get("dram_bw_utilization", 0.5)),
        raw={**raw_profile, "hardware": hardware, "ridge_point": ridge_point, "ground_truth_verified": True},
    )


def naive_prediction(arithmetic_intensity: float, ridge_point: float) -> str:
    return "compute-bound" if arithmetic_intensity > ridge_point else "memory-bound"


def classify_from_profile(profile: NCUProfile) -> str:
    dominant = profile.dominant_stall_type.lower()
    if "register" in dominant:
        return "register-spill"
    if profile.achieved_occupancy < 0.35 and not profile.memory_bound and not profile.compute_bound:
        return "occupancy-limited"
    if profile.stall_long_sb_pct > 0.30 and profile.dram_bw_utilization < 0.40:
        return "latency-bound"
    if profile.memory_bound:
        return "memory-bound"
    if profile.compute_bound:
        return "compute-bound"
    return "memory-bound"


def detect_case(code: str, profile: NCUProfile, ridge_point: float) -> tuple[str, str, str, str] | None:
    if profile.arithmetic_intensity > 10.0 and float(profile.raw.get("memory_bound_score", 0.0)) > 0.65:
        return (
            "memory-bound",
            "high arithmetic intensity in code suggests compute-bound but memory access pattern is inefficient",
            "compute_looks_memory",
            "hard",
        )
    if (
        profile.global_load_efficiency < 0.50
        and profile.stall_long_sb_pct > 0.30
        and profile.dram_bw_utilization < 0.40
    ):
        return (
            "latency-bound",
            "poor coalescing suggests memory-bound but bandwidth is not saturated - stalls are latency, not throughput",
            "memory_looks_latency",
            "hard",
        )
    if (
        profile.register_count > 128
        and profile.achieved_occupancy > 0.60
        and "register" not in profile.dominant_stall_type.lower()
    ):
        return (
            classify_from_profile(profile),
            "high register count suggests register spilling but occupancy is healthy - bottleneck is elsewhere",
            "register_pressure_decoy",
            "hard",
        )
    if "__shared__" in code and profile.achieved_occupancy < 0.35 and (profile.compute_bound or profile.memory_bound):
        return (
            classify_from_profile(profile),
            "shared memory usage looks like an optimization but has collapsed occupancy below 35%",
            "shared_memory_trap",
            "hard",
        )
    if profile.achieved_occupancy < 0.50 and profile.compute_bound and profile.stall_long_sb_pct < 0.15:
        return (
            "compute-bound",
            "low occupancy suggests occupancy-limited but compute units are actually saturated - occupancy is sufficient for latency hiding at this arithmetic intensity",
            "occupancy_looks_compute",
            "hard",
        )
    return None


def compute_stall_variance(profile: NCUProfile) -> float:
    values = [
        profile.stall_long_sb_pct,
        profile.stall_mem_pct,
        float(profile.raw.get("stall_short_sb_pct", 0.0)),
    ]
    return statistics.pvariance(values)


def compute_mislead_score(profile: NCUProfile, true_bottleneck: str, ridge_point: float) -> tuple[str, float]:
    predicted_naive_bottleneck = naive_prediction(profile.arithmetic_intensity, ridge_point)
    stall_variance = compute_stall_variance(profile)
    mislead_score = (
        float(predicted_naive_bottleneck != true_bottleneck) * 2.0
        + stall_variance * 0.5
        + (1.0 - profile.global_load_efficiency) * 0.3
    )
    return predicted_naive_bottleneck, mislead_score


def make_entry(item: dict[str, Any]) -> KernelEntry | None:
    if not bool(item.get("Correct", False)):
        return None
    code = str(item.get("code") or item.get("kernel_code") or "")
    task_id = str(item.get("task_id") or item.get("id") or "")
    hardware = str(item.get("hardware") or item.get("gpu") or "A100")
    raw_profile = item.get("NCU_Profile") or item.get("ncu_profile") or {}
    if not isinstance(raw_profile, dict):
        return None
    ridge_point = float(raw_profile.get("ridge_point", 10.0))
    profile = normalize_profile(raw_profile, hardware)
    detected = detect_case(code, profile, ridge_point)
    if detected is not None:
        true_bottleneck, misleading_signal, category, difficulty = detected
    else:
        true_bottleneck = classify_from_profile(profile)
        naive_label, mislead_score = compute_mislead_score(profile, true_bottleneck, ridge_point)
        if mislead_score <= 0.8:
            return None
        misleading_signal = (
            f"borderline case: metrics suggest {naive_label} but NCU roofline places kernel in {true_bottleneck} regime"
        )
        category = "borderline"
        difficulty = "medium"
        profile.raw = {**profile.raw, "mislead_score": mislead_score, "selection_tier": "secondary"}
    if true_bottleneck not in VALID_BOTTLENECKS:
        return None
    naive_label, mislead_score = compute_mislead_score(profile, true_bottleneck, ridge_point)
    profile.raw = {**profile.raw, "mislead_score": mislead_score, "naive_prediction": naive_label}
    return KernelEntry(
        id=f"sakana_{task_id.replace('/', '_').replace(' ', '_').lower()}",
        source="mined",
        code=code,
        pytorch_reference=item.get("pytorch_reference"),
        true_bottleneck=true_bottleneck,
        misleading_signal=misleading_signal,
        category=category,
        difficulty=difficulty,
        hardware=hardware,
        ncu_profile=profile,
        task_id=task_id,
    )


def load_dataset_entries() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    for row in dataset[split_name]:
        yield dict(row)


def build_mock_rows(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = [
        {
            "category": "compute_looks_memory",
            "profile": {
                "arithmetic_intensity": 14.0,
                "ridge_point": 10.0,
                "memory_bound_score": 0.82,
                "memory_bound": True,
                "compute_bound": False,
                "global_load_efficiency": 0.34,
                "achieved_occupancy": 0.66,
                "stall_long_sb_pct": 0.14,
                "stall_mem_pct": 0.41,
                "stall_short_sb_pct": 0.11,
                "register_count": 68,
                "l2_hit_rate": 0.28,
                "dram_bw_utilization": 0.81,
                "dominant_stall_type": "memory_dependency",
            },
            "code": "__global__ void kernel(float* x, const float* y) { int i = blockIdx.x * blockDim.x + threadIdx.x; float v = y[(i * 17) & 4095]; for (int j = 0; j < 8; ++j) { v = v * 1.1f + x[(i + j * 257) & 4095]; } x[i] = v; }",
            "difficulty": "hard",
        },
        {
            "category": "memory_looks_latency",
            "profile": {
                "arithmetic_intensity": 2.1,
                "ridge_point": 10.0,
                "memory_bound": False,
                "compute_bound": False,
                "global_load_efficiency": 0.44,
                "achieved_occupancy": 0.57,
                "stall_long_sb_pct": 0.39,
                "stall_mem_pct": 0.13,
                "stall_short_sb_pct": 0.08,
                "register_count": 72,
                "l2_hit_rate": 0.83,
                "dram_bw_utilization": 0.27,
                "dominant_stall_type": "long_scoreboard",
            },
            "code": "__global__ void kernel(float* x, const float* y) { int i = blockIdx.x * blockDim.x + threadIdx.x; float v = y[i]; v = v * 1.0001f + 1.0f; v = v * 1.0002f + 2.0f; v = v * 1.0003f + 3.0f; v = v * 1.0004f + 4.0f; v = v * 1.0005f + 5.0f; v = v * 1.0006f + 6.0f; x[i] = v; }",
            "difficulty": "hard",
        },
        {
            "category": "register_pressure_decoy",
            "profile": {
                "arithmetic_intensity": 6.1,
                "ridge_point": 10.0,
                "memory_bound": True,
                "compute_bound": False,
                "global_load_efficiency": 0.61,
                "achieved_occupancy": 0.71,
                "stall_long_sb_pct": 0.10,
                "stall_mem_pct": 0.33,
                "stall_short_sb_pct": 0.07,
                "register_count": 156,
                "l2_hit_rate": 0.37,
                "dram_bw_utilization": 0.74,
                "dominant_stall_type": "math_pipe",
            },
            "code": "__global__ void kernel(float* x, const float* y) { int i = blockIdx.x * blockDim.x + threadIdx.x; float r0 = x[i]; float r1 = y[(i * 37) & 4095]; x[i] = r0 + r1; }",
            "difficulty": "hard",
        },
        {
            "category": "shared_memory_trap",
            "profile": {
                "arithmetic_intensity": 11.7,
                "ridge_point": 10.0,
                "memory_bound": False,
                "compute_bound": True,
                "global_load_efficiency": 0.88,
                "achieved_occupancy": 0.22,
                "stall_long_sb_pct": 0.32,
                "stall_mem_pct": 0.16,
                "stall_short_sb_pct": 0.10,
                "register_count": 92,
                "l2_hit_rate": 0.79,
                "dram_bw_utilization": 0.35,
                "dominant_stall_type": "long_scoreboard",
            },
            "code": "__global__ void kernel(float* x) { __shared__ float tile[128][64]; int i = blockIdx.x * blockDim.x + threadIdx.x; tile[threadIdx.x][0] = x[i]; __syncthreads(); x[i] = tile[threadIdx.x][0]; }",
            "difficulty": "hard",
        },
        {
            "category": "occupancy_looks_compute",
            "profile": {
                "arithmetic_intensity": 15.2,
                "ridge_point": 10.0,
                "memory_bound": False,
                "compute_bound": True,
                "global_load_efficiency": 0.91,
                "achieved_occupancy": 0.41,
                "stall_long_sb_pct": 0.09,
                "stall_mem_pct": 0.10,
                "stall_short_sb_pct": 0.08,
                "register_count": 84,
                "l2_hit_rate": 0.77,
                "dram_bw_utilization": 0.33,
                "dominant_stall_type": "math_pipe",
            },
            "code": "__global__ void kernel(float* x) { __shared__ float tile[64][16]; int i = blockIdx.x * blockDim.x + threadIdx.x; float v = x[i]; for (int j = 0; j < 16; ++j) { v = v * 1.01f + tile[j][threadIdx.x & 15]; } x[i] = v; }",
            "difficulty": "hard",
        },
        {
            "category": "borderline",
            "profile": {
                "arithmetic_intensity": 9.7,
                "ridge_point": 10.0,
                "memory_bound": True,
                "compute_bound": False,
                "global_load_efficiency": 0.18,
                "achieved_occupancy": 0.64,
                "stall_long_sb_pct": 0.06,
                "stall_mem_pct": 0.07,
                "stall_short_sb_pct": 0.05,
                "register_count": 48,
                "l2_hit_rate": 0.22,
                "dram_bw_utilization": 0.58,
                "dominant_stall_type": "memory_dependency",
            },
            "code": "__global__ void kernel(float* x, const float* y) { int i = blockIdx.x * blockDim.x + threadIdx.x; x[i] = y[(i * 97) & 4095] + 1.0f; }",
            "difficulty": "medium",
        },
    ]
    for index in range(limit * 3):
        template = templates[index % len(templates)]
        rows.append(
            {
                "Correct": True,
                "task_id": f"mock_{index:03d}",
                "hardware": "A100",
                "NCU_Profile": dict(template["profile"]),
                "code": template["code"],
                "category": template["category"],
                "difficulty": template["difficulty"],
            }
        )
    return rows


def mine_candidates(rows: Iterable[dict[str, Any]], limit: int) -> list[KernelEntry]:
    candidates = [entry for entry in (make_entry(row) for row in rows) if entry is not None]
    ranked = sorted(
        candidates,
        key=lambda entry: (float(entry.ncu_profile.raw.get("mislead_score", 0.0)), entry.difficulty == "hard"),
        reverse=True,
    )
    return ranked[:limit]


def write_jsonl(entries: Iterable[KernelEntry], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(kernel_entry_to_dict(entry)) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_mock_rows(args.limit) if args.mock else load_dataset_entries()
    entries = mine_candidates(rows, args.limit)
    write_jsonl(entries, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
