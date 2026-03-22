from __future__ import annotations

from eval.baselines import frequency_baseline, random_baseline, roofline_baseline
from registry.kernel_entry import KernelEntry, NCUProfile


def make_entry(kernel_id: str, label: str) -> KernelEntry:
    return KernelEntry(
        id=kernel_id,
        source="handwritten",
        code="__global__ void k(float* x){ int i = threadIdx.x; x[i] = x[i] * 2.0f + 1.0f; }",
        pytorch_reference=None,
        true_bottleneck=label,
        misleading_signal="test",
        category=label,
        difficulty="easy",
        hardware="A100",
        ncu_profile=NCUProfile(
            arithmetic_intensity=1.0,
            memory_bound=label == "memory-bound",
            compute_bound=label == "compute-bound",
            dominant_stall_type="memory_dependency",
            global_load_efficiency=0.5,
            achieved_occupancy=0.5,
            stall_long_sb_pct=0.2,
            stall_mem_pct=0.3,
            register_count=64,
            l2_hit_rate=0.5,
            dram_bw_utilization=0.4,
            raw={"ridge_point": 1.0},
        ),
        task_id=None,
    )


def test_baseline_computations() -> None:
    entries = [
        make_entry("a", "memory-bound"),
        make_entry("b", "memory-bound"),
        make_entry("c", "compute-bound"),
    ]
    random_stats = random_baseline(entries, trials=20, seed=1)
    assert 0.0 <= random_stats["mean"] <= 1.0
    frequency = frequency_baseline(entries)
    assert frequency["label"] == "memory-bound"
    roofline = roofline_baseline(entries, roof={"peak_flops_tflops": 1.0, "peak_bw_tbps": 1.0})
    assert "predictions" in roofline
