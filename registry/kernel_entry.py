from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

BottleneckLabel = Literal[
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "occupancy-limited",
    "register-spill"
]

# Confusion severity matrix - how wrong is each misprediction?
# 0 = correct, 1 = adjacent mistake, 2 = completely wrong direction
SEVERITY: dict[tuple[BottleneckLabel, BottleneckLabel], int] = {
    ("memory-bound",      "latency-bound"):       1,
    ("memory-bound",      "compute-bound"):        2,
    ("memory-bound",      "occupancy-limited"):    1,
    ("memory-bound",      "register-spill"):       1,
    ("compute-bound",     "memory-bound"):         2,
    ("compute-bound",     "latency-bound"):        2,
    ("compute-bound",     "occupancy-limited"):    1,
    ("compute-bound",     "register-spill"):       1,
    ("latency-bound",     "memory-bound"):         1,
    ("latency-bound",     "compute-bound"):        2,
    ("latency-bound",     "occupancy-limited"):    1,
    ("latency-bound",     "register-spill"):       1,
    ("occupancy-limited", "memory-bound"):         1,
    ("occupancy-limited", "compute-bound"):        1,
    ("occupancy-limited", "latency-bound"):        1,
    ("occupancy-limited", "register-spill"):       1,
    ("register-spill",    "memory-bound"):         1,
    ("register-spill",    "compute-bound"):        2,
    ("register-spill",    "latency-bound"):        1,
    ("register-spill",    "occupancy-limited"):    1,
}


@dataclass
class NCUProfile:
    arithmetic_intensity: float        # FLOP/byte
    memory_bound: bool
    compute_bound: bool
    dominant_stall_type: str           # raw ncu stall category
    global_load_efficiency: float      # 0-1
    achieved_occupancy: float          # 0-1
    stall_long_sb_pct: float           # % warps stalled on long scoreboard
    stall_mem_pct: float               # % warps stalled on memory
    register_count: int
    l2_hit_rate: float                 # 0-1
    dram_bw_utilization: float         # 0-1
    raw: dict                          # full ncu output, unparsed
    verification: Optional[dict[str, Any]] = None


@dataclass
class KernelEntry:
    id: str                            # unique, slug format e.g. "sakana_l1_042"
    source: Literal["mined", "handwritten", "kernelbot"]
    code: str                          # full CUDA/Triton source
    pytorch_reference: Optional[str]   # PyTorch equivalent, if available
    true_bottleneck: BottleneckLabel   # ground truth from NCU, never human judgment
    misleading_signal: str             # human-readable: "high FLOP count suggests
                                       # compute-bound but memory access is strided"
    category: str                      # e.g. "compute_looks_memory"
    difficulty: Literal["easy", "medium", "hard"]
    hardware: str                      # e.g. "A10G", "A100", "H100"
    correct_explanation: Optional[str] # plain-English statement of the true bottleneck
    ncu_profile: NCUProfile
    task_id: Optional[str]             # original source task_id if mined
    reasoning_rubric: Optional[dict[str, Any]] = None


def ncu_profile_from_dict(data: dict[str, Any]) -> NCUProfile:
    verification = data.get("verification")
    if verification is None:
        verification = dict(data.get("raw", {})).get("verification")
    profile = NCUProfile(
        arithmetic_intensity=float(data["arithmetic_intensity"]),
        memory_bound=bool(data["memory_bound"]),
        compute_bound=bool(data["compute_bound"]),
        dominant_stall_type=str(data["dominant_stall_type"]),
        global_load_efficiency=float(data["global_load_efficiency"]),
        achieved_occupancy=float(data["achieved_occupancy"]),
        stall_long_sb_pct=float(data["stall_long_sb_pct"]),
        stall_mem_pct=float(data["stall_mem_pct"]),
        register_count=int(data["register_count"]),
        l2_hit_rate=float(data["l2_hit_rate"]),
        dram_bw_utilization=float(data["dram_bw_utilization"]),
        raw=dict(data.get("raw", {})),
        verification=dict(verification) if isinstance(verification, dict) else None,
    )
    if profile.verification is None:
        profile.verification = infer_verification(profile)
    return profile


def infer_verification(profile: NCUProfile) -> dict[str, str]:
    roof = profile.raw.get("roof", {})
    ridge_point = float(profile.raw.get("ridge_point", 0.0))
    if ridge_point <= 0.0 and isinstance(roof, dict):
        peak_bw = float(roof.get("peak_bw_tbps", 0.0))
        peak_flops = float(roof.get("peak_flops_tflops", 0.0))
        if peak_bw > 0.0:
            ridge_point = peak_flops / peak_bw
    if profile.register_count >= 255 and profile.stall_long_sb_pct > 0.20:
        roofline = "register-spill"
    elif profile.achieved_occupancy < 0.10:
        roofline = "occupancy-limited"
    elif profile.arithmetic_intensity < ridge_point:
        roofline = "memory-bound"
    else:
        roofline = "compute-bound"
    if profile.register_count >= 255 and profile.stall_long_sb_pct > 0.20 and profile.dram_bw_utilization < 0.25:
        bandwidth = "register-spill"
    elif profile.achieved_occupancy < 0.10:
        bandwidth = "occupancy-limited"
    elif profile.dram_bw_utilization > 0.50:
        bandwidth = "memory-bound"
    elif profile.stall_long_sb_pct > 0.30 and profile.dram_bw_utilization < 0.10:
        bandwidth = "latency-bound"
    else:
        bandwidth = "compute-bound"
    dominant = profile.dominant_stall_type.lower()
    if "register" in dominant:
        stall = "register-spill"
    elif "occupancy" in dominant:
        stall = "occupancy-limited"
    elif "memory" in dominant:
        stall = "memory-bound"
    elif "arithmetic" in dominant or dominant in {"latency-bound", "long_scoreboard"}:
        stall = "latency-bound"
    elif dominant == "compute":
        stall = "compute-bound"
    elif profile.achieved_occupancy < 0.40:
        stall = "occupancy-limited"
    else:
        stall = "compute-bound"
    votes = [roofline, bandwidth, stall]
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote] = counts.get(vote, 0) + 1
    consensus_label, consensus_count = max(counts.items(), key=lambda item: item[1])
    consensus = consensus_label if consensus_count >= 2 else "ambiguous"
    confidence = "high" if consensus_count == 3 else "medium" if consensus_count == 2 else "low"
    return {
        "roofline": roofline,
        "bandwidth": bandwidth,
        "stall": stall,
        "consensus": consensus,
        "confidence": confidence,
    }


def kernel_entry_from_dict(data: dict[str, Any]) -> KernelEntry:
    return KernelEntry(
        id=str(data["id"]),
        source=data["source"],
        code=str(data["code"]),
        pytorch_reference=data.get("pytorch_reference"),
        true_bottleneck=data["true_bottleneck"],
        misleading_signal=str(data["misleading_signal"]),
        category=str(data["category"]),
        difficulty=data["difficulty"],
        hardware=str(data["hardware"]),
        correct_explanation=data.get("correct_explanation"),
        reasoning_rubric=dict(data["reasoning_rubric"]) if isinstance(data.get("reasoning_rubric"), dict) else None,
        ncu_profile=ncu_profile_from_dict(dict(data["ncu_profile"])),
        task_id=data.get("task_id"),
    )


def kernel_entry_to_dict(entry: KernelEntry) -> dict[str, Any]:
    return asdict(entry)
