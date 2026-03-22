from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from registry.kernel_entry import KernelEntry, NCUProfile, kernel_entry_to_dict

DATASET_NAME = "GPUMODE/kernelbot-data"
DEFAULT_OUTPUT = Path("data/kernelbot_kernels.jsonl")
PATTERN_ORDER = [
    "shared_memory_small_block",
    "strided_access",
    "dependency_chain",
    "register_pressure",
]
PATTERN_META = {
    "shared_memory_small_block": (
        "occupancy-limited",
        "shared memory usage looks optimized but small threadblock may collapse occupancy",
    ),
    "strided_access": (
        "memory-bound",
        "non-power-of-2 stride causes cache line waste - may look compute-bound if FLOP count is high",
    ),
    "dependency_chain": (
        "latency-bound",
        "sequential dependencies create long scoreboard stalls - looks memory-bound due to load instructions",
    ),
    "register_pressure": (
        "register-spill",
        "large local arrays likely spill to local memory, appearing as global memory traffic",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine suspicious kernels from KernelBot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def sequential_dependency_count(code: str) -> int:
    streak = 0
    best = 0
    assignment_re = re.compile(r"^\s*(?:float|double|int)?\s*([A-Za-z_]\w*)\s*=\s*(.+)\s*$")
    statements = [statement.strip() for statement in code.replace("\n", ";").split(";")]
    for line in statements:
        if not line or line.startswith("//"):
            continue
        match = assignment_re.match(line)
        if match is None:
            streak = 0
            continue
        lhs, rhs = match.groups()
        if re.search(rf"\b{lhs}\b", rhs):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def detect_patterns(code: str) -> list[str]:
    matches: list[str] = []
    launch_match = re.search(r"<<<[^,]+,\s*(\d+)", code)
    if "__shared__" in code and launch_match is not None and int(launch_match.group(1)) <= 64:
        matches.append("shared_memory_small_block")

    for stride_match in re.finditer(r"\w+\[\s*\w+\s*\*\s*([0-9]+)", code):
        stride = int(stride_match.group(1))
        if stride > 32 and not is_power_of_two(stride):
            matches.append("strided_access")
            break

    if sequential_dependency_count(code) >= 6:
        matches.append("dependency_chain")

    local_array_match = re.search(r"(?:float|int)\s+\w+\s*\[\s*(\d+)\s*\]", code)
    decl_names: set[str] = set()
    for decl_match in re.finditer(r"(?:float|int)\s+([^;]+);", code):
        for piece in decl_match.group(1).split(","):
            candidate = piece.strip().split("=")[0].strip()
            candidate = candidate.split("[")[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", candidate):
                decl_names.add(candidate)
    decl_count = len(decl_names)
    if (local_array_match is not None and int(local_array_match.group(1)) >= 16) or decl_count > 20:
        matches.append("register_pressure")

    seen: list[str] = []
    for pattern in matches:
        if pattern not in seen:
            seen.append(pattern)
    return seen


def detect_pattern(code: str) -> tuple[str, str, str] | None:
    """
    Returns (suspected_bottleneck, misleading_signal, category)
    or None if no interesting pattern found.
    Only returns kernels that warrant profiling - not ground truth.
    """
    matches = detect_patterns(code)
    if not matches:
        return None
    category = min(matches, key=lambda item: PATTERN_ORDER.index(item))
    suspected_bottleneck, misleading_signal = PATTERN_META[category]
    return suspected_bottleneck, misleading_signal, category


def make_sentinel_profile(suspected_bottleneck: str, confidence: int, categories: list[str], score_hint: float) -> NCUProfile:
    return NCUProfile(
        arithmetic_intensity=-1.0,
        memory_bound=False,
        compute_bound=False,
        dominant_stall_type="unknown",
        global_load_efficiency=-1.0,
        achieved_occupancy=-1.0,
        stall_long_sb_pct=-1.0,
        stall_mem_pct=-1.0,
        register_count=-1,
        l2_hit_rate=-1.0,
        dram_bw_utilization=-1.0,
        raw={
            "source": "kernelbot",
            "needs_profiling": True,
            "suspected_bottleneck": suspected_bottleneck,
            "ground_truth_verified": False,
            "pattern_confidence": confidence,
            "matched_patterns": categories,
            "score_hint": score_hint,
        },
    )


def make_entry(row: dict[str, Any], fallback_index: int = 0) -> KernelEntry | None:
    code = str(row.get("submission_code") or "")
    detected = detect_pattern(code)
    if detected is None:
        return None
    suspected_bottleneck, misleading_signal, category = detected
    matches = detect_patterns(code)
    confidence = len(matches)
    problem_id = str(row.get("problem_id") or f"mock_problem_{fallback_index:03d}")
    hardware = str(row.get("hardware") or "unknown")
    score_hint = float(row.get("score") or 0.0)
    return KernelEntry(
        id=f"kernelbot_{problem_id.replace('/', '_').replace(' ', '_').lower()}",
        source="kernelbot",
        code=code,
        pytorch_reference=None,
        true_bottleneck=suspected_bottleneck,
        misleading_signal=misleading_signal,
        category=category,
        difficulty="medium" if confidence == 1 else "hard",
        hardware=hardware,
        ncu_profile=make_sentinel_profile(suspected_bottleneck, confidence, matches, score_hint),
        task_id=problem_id,
    )


def load_dataset_entries() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    for row in dataset[split_name]:
        yield dict(row)


def build_mock_rows(limit: int) -> list[dict[str, Any]]:
    base_rows = [
        {
            "problem_id": "shared_00",
            "score": 1.0,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ __shared__ float tile[32][32]; int i = threadIdx.x; tile[i][0] = x[i]; } int main(){ k<<<256, 64>>>(0); }",
        },
        {
            "problem_id": "shared_01",
            "score": 1.1,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ __shared__ int tile[64]; x[threadIdx.x] = tile[threadIdx.x & 63]; } int main(){ k<<<128, 32>>>(0); }",
        },
        {
            "problem_id": "shared_02",
            "score": 1.2,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ __shared__ float scratch[128]; x[threadIdx.x] = scratch[threadIdx.x]; } int main(){ k<<<64, 48>>>(0); }",
        },
        {
            "problem_id": "shared_03",
            "score": 1.3,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ __shared__ float tile[16][16]; x[threadIdx.x] = tile[threadIdx.x & 15][0]; } int main(){ k<<<512, 64>>>(0); }",
        },
        {
            "problem_id": "stride_00",
            "score": 0.9,
            "hardware": "AMD MI300X",
            "submission_code": "__global__ void k(float* x, float* y){ int i = blockIdx.x * blockDim.x + threadIdx.x; x[i] = y[i * 97] + 1.0f; }",
        },
        {
            "problem_id": "stride_01",
            "score": 1.0,
            "hardware": "AMD MI300X",
            "submission_code": "__global__ void k(float* x){ int t = threadIdx.x; x[t] = x[t * 45] * 2.0f; }",
        },
        {
            "problem_id": "stride_02",
            "score": 1.1,
            "hardware": "AMD MI300X",
            "submission_code": "__global__ void k(float* x, float* y){ int lane = threadIdx.x; y[lane] = x[lane * 63] + x[lane]; }",
        },
        {
            "problem_id": "stride_03",
            "score": 1.2,
            "hardware": "AMD MI300X",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; x[i] = x[i * 51] + x[i * 3]; }",
        },
        {
            "problem_id": "dep_00",
            "score": 0.8,
            "hardware": "NVFP4",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float v = x[i]; v = v * 1.1f + 1.0f; v = v * 1.2f + 2.0f; v = v * 1.3f + 3.0f; v = v * 1.4f + 4.0f; v = v * 1.5f + 5.0f; v = v * 1.6f + 6.0f; x[i] = v; }",
        },
        {
            "problem_id": "dep_01",
            "score": 0.81,
            "hardware": "NVFP4",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float acc = x[i]; acc = acc + 1.0f; acc = acc * 0.5f; acc = acc + 2.0f; acc = acc / 1.1f; acc = acc + 3.0f; acc = acc * 0.75f; x[i] = acc; }",
        },
        {
            "problem_id": "dep_02",
            "score": 0.82,
            "hardware": "NVFP4",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float r = x[i]; r = r - 0.5f; r = r * r; r = r + 0.25f; r = r / 1.5f; r = r + 0.125f; r = r * 1.05f; x[i] = r; }",
        },
        {
            "problem_id": "dep_03",
            "score": 0.83,
            "hardware": "NVFP4",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float z = x[i]; z = z + 1.0f; z = z * 1.01f; z = z + 2.0f; z = z * 1.02f; z = z + 3.0f; z = z * 1.03f; x[i] = z; }",
        },
        {
            "problem_id": "reg_00",
            "score": 1.4,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float tmp[16]; for(int j=0;j<16;++j){ tmp[j] = x[i + j]; } x[i] = tmp[0]; }",
        },
        {
            "problem_id": "reg_01",
            "score": 1.5,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20; x[i] = a0 + a20; }",
        },
        {
            "problem_id": "reg_02",
            "score": 1.6,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; int idx[16]; for(int j=0;j<16;++j){ idx[j] = i + j; } x[i] = x[idx[0]]; }",
        },
        {
            "problem_id": "reg_03",
            "score": 1.7,
            "hardware": "NVIDIA A100",
            "submission_code": "__global__ void k(float* x){ int i = threadIdx.x; float cache[24]; for(int j=0;j<24;++j){ cache[j] = x[i + j]; } x[i] = cache[0] + cache[23]; }",
        },
    ]
    rows: list[dict[str, Any]] = []
    for index in range(max(limit, len(base_rows))):
        template = dict(base_rows[index % len(base_rows)])
        template["problem_id"] = f"{template['problem_id']}_{index:03d}"
        template["score"] = float(template["score"]) + index * 0.001
        rows.append(template)
    return rows


def mine_candidates(rows: Iterable[dict[str, Any]], limit: int) -> list[KernelEntry]:
    entries = [entry for index, entry in enumerate(make_entry(row, index) for index, row in enumerate(rows)) if entry is not None]
    ranked = sorted(
        entries,
        key=lambda entry: (
            int(entry.ncu_profile.raw.get("pattern_confidence", 0)),
            -float(entry.ncu_profile.raw.get("score_hint", 0.0)),
        ),
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
