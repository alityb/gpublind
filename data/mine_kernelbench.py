from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from registry.kernel_entry import KernelEntry, NCUProfile, kernel_entry_to_dict

DATASET_NAME = "ScalingIntelligence/KernelBench"
DEFAULT_OUTPUT = Path("data/kernelbench_compute_candidates.jsonl")
RIDGE_POINT = 62.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine compute-bound candidates from KernelBench")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--levels", default="2,3")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def parse_levels(raw_levels: str) -> list[str]:
    return [f"level_{level.strip()}" for level in raw_levels.split(",") if level.strip()]


def count_pattern(code: str, patterns: list[str]) -> int:
    lowered = code.lower()
    return sum(lowered.count(pattern) for pattern in patterns)


def estimate_output_elements(code: str) -> int:
    candidates = [int(match) for match in re.findall(r"=\s*(\d+)(?:\s*\*\s*(\d+))?", code) for match in match if match]
    if not candidates:
        return 0
    candidates = [value for value in candidates if 1 < value <= 65536]
    candidates.sort(reverse=True)
    estimate = 1
    for value in candidates[:4]:
        estimate *= value
        if estimate > 16_000_000:
            break
    return estimate


def estimate_arithmetic_intensity(code: str) -> tuple[float, dict[str, int]]:
    stats = {
        "matmul": count_pattern(code, ["torch.matmul", "torch.mm", "torch.bmm", "gemm", "matmul"]),
        "linear": count_pattern(code, ["nn.linear", "f.linear"]),
        "attention": count_pattern(code, ["scaled_dot_product_attention", "multiheadattention", "attention"]),
        "conv": count_pattern(code, ["nn.conv1d", "nn.conv2d", "nn.conv3d", "convtranspose"]),
        "elementwise": count_pattern(code, ["relu", "gelu", "sigmoid", "tanh", "add(", "multiply", "sum(", "mean(", "clamp", "softmax"]),
    }
    ai = (
        stats["matmul"] * 110.0
        + stats["linear"] * 90.0
        + stats["attention"] * 140.0
        + stats["conv"] * 80.0
        - stats["elementwise"] * 2.0
    )
    return max(ai, 1.0), stats


def is_compute_candidate(code: str, level: int, estimated_ai: float, stats: dict[str, int], output_elements: int) -> bool:
    compute_ops = stats["matmul"] + stats["linear"] + stats["attention"] + stats["conv"]
    if level < 2:
        return False
    if estimated_ai <= RIDGE_POINT:
        return False
    if output_elements <= 1_000_000:
        return False
    if stats["attention"] > 0:
        return True
    return compute_ops >= 2 and stats["elementwise"] <= max(compute_ops * 3, 8)


def make_cu_template(name: str, problem_id: int, code: str) -> str:
    lowered = code.lower()
    if "attention" in lowered:
        body = "  // TODO: materialize Q/K/V tensors and call a fused attention kernel.\n"
    elif "conv" in lowered:
        body = "  // TODO: materialize activation/filter tensors and call cuDNN or CUTLASS convolution.\n"
    else:
        body = "  // TODO: materialize large matrices and call cuBLAS GEMM or grouped GEMM.\n"
    return (
        "#include <cuda_runtime.h>\n"
        "#include <cstdio>\n\n"
        f"// KernelBench candidate {problem_id}: {name}\n"
        "int main() {\n"
        "  // Standalone profiling harness template generated from KernelBench PyTorch reference.\n"
        f"{body}"
        "  cudaDeviceSynchronize();\n"
        '  std::puts("TODO: fill in generated harness before profiling.");\n'
        "  return 0;\n"
        "}\n"
    )


def make_sentinel_profile(estimated_ai: float, candidate_reason: str, cu_template: str) -> NCUProfile:
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
            "source": "kernelbench",
            "ground_truth_verified": False,
            "needs_profiling": True,
            "estimated_ai": estimated_ai,
            "candidate_reason": candidate_reason,
            "cu_template": cu_template,
        },
    )


def make_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    code = str(row["code"])
    level = int(row["level"])
    name = str(row["name"])
    problem_id = int(row["problem_id"])
    estimated_ai, stats = estimate_arithmetic_intensity(code)
    output_elements = estimate_output_elements(code)
    if not is_compute_candidate(code, level, estimated_ai, stats, output_elements):
        return None
    candidate_reason = (
        f"level={level}, compute_ops={stats['matmul'] + stats['linear'] + stats['attention'] + stats['conv']}, "
        f"estimated_ai={estimated_ai:.1f}, output_elements~{output_elements}"
    )
    cu_template = make_cu_template(name, problem_id, code)
    entry = KernelEntry(
        id=f"kernelbench_{problem_id:03d}",
        source="mined",
        code=cu_template,
        pytorch_reference=code,
        true_bottleneck="compute-bound",
        misleading_signal="framework-level fusion and activation code can obscure that large GEMMs dominate the workload",
        category="kernelbench_compute_candidate",
        difficulty="medium" if level == 2 else "hard",
        hardware="A10G",
        correct_explanation="multiple large matrix-style operators dominate arithmetic work and likely exceed the roofline ridge point",
        ncu_profile=make_sentinel_profile(estimated_ai, candidate_reason, cu_template),
        task_id=str(problem_id),
        reasoning_rubric=None,
    )
    payload = kernel_entry_to_dict(entry)
    payload.update(
        {
            "task_id": str(problem_id),
            "name": name,
            "pytorch_code": code,
            "estimated_ai": estimated_ai,
            "candidate_reason": candidate_reason,
            "cu_template": cu_template,
            "level": level,
        }
    )
    return payload


def build_mock_rows() -> list[dict[str, Any]]:
    return [
        {
            "problem_id": 1,
            "level": 3,
            "name": "MockTransformerBlock",
            "code": (
                "import torch\n"
                "class Model(torch.nn.Module):\n"
                "  def forward(self, q, k, v, w1, w2):\n"
                "    x = torch.matmul(q, k.transpose(-1, -2))\n"
                "    y = torch.matmul(x, v)\n"
                "    z = torch.matmul(y, w1)\n"
                "    return torch.matmul(z, w2)\n"
                "batch_size = 32\nsequence_length = 1024\nhidden = 4096\n"
            ),
        },
        {
            "problem_id": 2,
            "level": 2,
            "name": "MockElementwise",
            "code": "import torch\nclass Model(torch.nn.Module):\n  def forward(self, x): return torch.relu(x) + torch.sigmoid(x)\nbatch_size = 4096\nhidden = 256\n",
        },
    ]


def iter_rows(levels: list[str], mock: bool) -> list[dict[str, Any]]:
    if mock:
        return build_mock_rows()
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME)
    rows: list[dict[str, Any]] = []
    for split in levels:
        if split not in dataset:
            continue
        for row in dataset[split]:
            rows.append(dict(row))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    levels = parse_levels(args.levels)
    rows = iter_rows(levels, args.mock)
    items: list[dict[str, Any]] = []
    for row in rows:
        entry = make_entry(row)
        if entry is not None:
            items.append(entry)
    items.sort(key=lambda item: (-float(item["estimated_ai"]), str(item["task_id"])))
    selected = items[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item) + "\n")
    print(f"Wrote {len(selected)} KernelBench compute candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
