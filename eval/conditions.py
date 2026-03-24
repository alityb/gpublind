from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus import CorpusEntry, load_entry

QUESTION_TEXT = """What is the primary performance bottleneck?
Choose exactly one label:

memory-bound: kernel throughput is limited by memory bandwidth
  (DRAM utilization should be high, >50% of peak)
compute-bound: kernel throughput is limited by arithmetic units
  (arithmetic intensity above ridge point)
latency-bound: kernel is stalled waiting for dependent operations
  (long scoreboard stalls high, DRAM utilization LOW — distinct
  from memory-bound)
occupancy-limited: insufficient active warps to hide latency
  (achieved occupancy <35%, causing stalls)
register-spill: register pressure forces spilling to local memory
  (register count at max, local memory traffic visible)

BOTTLENECK: <exactly one label>
CONFIDENCE: <HIGH, MEDIUM, or LOW>
REASONING: <2-3 sentences citing specific metric values
            or code patterns as evidence>"""


@dataclass(frozen=True)
class Condition:
    id: int
    name: str


CONDITION_0 = Condition(0, "code_only")
CONDITION_1 = Condition(1, "metrics_only")
CONDITION_2 = Condition(2, "code_plus_metrics")
CONDITION_3 = Condition(3, "code_plus_metrics_plus_context")
CONDITION_4 = Condition(4, "correction")

CONDITIONS = {
    0: CONDITION_0,
    1: CONDITION_1,
    2: CONDITION_2,
    3: CONDITION_3,
    4: CONDITION_4,
}


def metrics_block(entry: CorpusEntry) -> str:
    p = entry.profile
    hardware = p["hardware"]
    return (
        f"Arithmetic Intensity: {p['arithmetic_intensity_flop_per_byte']:.2f} FLOP/byte\n"
        f"  (Ridge point: {hardware['ridge_point_flop_per_byte']:.0f} FLOP/byte — above=compute-bound tendency,\n"
        f"   below=memory-bound tendency, but check DRAM utilization)\n"
        f"Achieved Occupancy: {p['achieved_occupancy_pct']:.1f}%\n"
        f"DRAM Bandwidth Utilization: {p['dram_bw_utilization_pct']:.1f}% of peak ({hardware['peak_bw_gbps']:.0f} GB/s)\n"
        f"Global Load Efficiency: {p['global_load_efficiency_pct']:.1f}%\n"
        f"Stall — Long Scoreboard: {p['stall_long_scoreboard_pct']:.1f}% of active cycles\n"
        f"Stall — Memory Dependency: {p['stall_memory_pct']:.1f}% of active cycles\n"
        f"L2 Hit Rate: {p['l2_hit_rate_pct']:.1f}%\n"
        f"Registers per Thread: {p['register_count_per_thread']}"
    )


def roofline_context(entry: CorpusEntry) -> str:
    p = entry.profile
    ai = float(p["arithmetic_intensity_flop_per_byte"])
    ridge = float(p["hardware"]["ridge_point_flop_per_byte"])
    return (
        "Roofline position:\n"
        f"  This kernel's arithmetic intensity ({ai:.2f} FLOP/byte) is\n"
        f"  {'ABOVE' if ai > ridge else 'BELOW'} the ridge point ({ridge:.0f} FLOP/byte),\n"
        f"  suggesting {'compute' if ai > ridge else 'memory'}-bound tendency.\n\n"
        "  However: if DRAM utilization is LOW (<10%) despite being below\n"
        "  the ridge point, the kernel is likely latency-bound (waiting on\n"
        "  instruction dependencies), NOT memory-bound (not saturating BW)."
    )


def contradiction_hint(entry: CorpusEntry, wrong_label: str) -> str:
    p = entry.profile
    if wrong_label == "memory-bound":
        return f"DRAM utilization is only {p['dram_bw_utilization_pct']:.1f}% of peak, which contradicts a memory-bandwidth bottleneck."
    if wrong_label == "compute-bound":
        return f"Arithmetic intensity is {p['arithmetic_intensity_flop_per_byte']:.2f} FLOP/byte versus a ridge point of {p['hardware']['ridge_point_flop_per_byte']:.0f}."
    if wrong_label == "occupancy-limited":
        return f"Achieved occupancy is {p['achieved_occupancy_pct']:.1f}%, which may not support an occupancy-limited diagnosis."
    if wrong_label == "register-spill":
        return f"Registers per thread are {p['register_count_per_thread']}, which may not support a register-spill diagnosis."
    return f"Long scoreboard stalls are {p['stall_long_scoreboard_pct']:.1f}% while DRAM utilization is {p['dram_bw_utilization_pct']:.1f}%."


def render_condition(entry: CorpusEntry, condition_id: int, *, prior_result: dict | None = None) -> dict[str, str]:
    hardware = entry.profile["hardware"]
    code_block = f"```cuda\n{entry.kernel_code}\n```"
    metrics = metrics_block(entry)
    if condition_id == 0:
        user = (
            f"Diagnose the primary performance bottleneck in this CUDA kernel\n"
            f"running on an {hardware['name']} GPU (ridge point: {hardware['ridge_point_flop_per_byte']:.0f} FLOP/byte).\n"
            f"{code_block}\n\n{QUESTION_TEXT}"
        )
    elif condition_id == 1:
        user = (
            f"Diagnose the bottleneck of a CUDA kernel on {hardware['name']} given\n"
            "these hardware performance counters. You will NOT see the source.\n\n"
            f"{metrics}\n\n{QUESTION_TEXT}"
        )
    elif condition_id == 2:
        user = (
            "Diagnose the primary performance bottleneck in this CUDA kernel.\n\n"
            f"Hardware performance counters ({hardware['name']}):\n{metrics}\n\n"
            f"Source code:\n{code_block}\n\n{QUESTION_TEXT}"
        )
    elif condition_id == 3:
        user = (
            "Diagnose the primary performance bottleneck in this CUDA kernel.\n\n"
            f"Hardware performance counters ({hardware['name']}):\n{metrics}\n\n"
            f"{roofline_context(entry)}\n\nSource code:\n{code_block}\n\n{QUESTION_TEXT}"
        )
    elif condition_id == 4:
        if prior_result is None:
            raise ValueError("Condition 4 requires a prior wrong C2 result.")
        user = (
            f"Your previous diagnosis:\nBOTTLENECK: {prior_result.get('predicted_label', 'unknown')}\n"
            f"REASONING: {prior_result.get('reasoning', '')}\n\n"
            "Here is additional evidence that may change your answer:\n"
            f"{contradiction_hint(entry, str(prior_result.get('predicted_label', 'unknown')))}\n\n"
            "Please reconsider. Does this evidence change your diagnosis?\n\n"
            "BOTTLENECK: <label>\n"
            "CONFIDENCE: <HIGH, MEDIUM, or LOW>\n"
            "REASONING: <updated reasoning>"
        )
    else:
        raise KeyError(f"Unknown condition {condition_id}")
    return {
        "system": "You are a GPU performance engineer. Follow the requested format exactly.",
        "user": user,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GPUBlind v2 prompts")
    parser.add_argument("--kernel", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entry = load_entry(args.kernel)
    for condition_id in range(4):
        rendered = render_condition(entry, condition_id)
        print(f"=== C{condition_id}: {CONDITIONS[condition_id].name} ===")
        print(rendered["user"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
