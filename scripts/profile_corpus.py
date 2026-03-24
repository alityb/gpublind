from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus.verify import verify_profile

METRICS = [
    "dram__bytes.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "l2__global_load_requests.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_global_ld_sectors_miss.sum",
    "launch__registers_per_thread",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile corpus kernels on A10G with ncu")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--failure-log", type=Path, default=Path("scripts/profile_failures.txt"))
    parser.add_argument("--conflict-log", type=Path, default=Path("scripts/label_conflicts.txt"))
    parser.add_argument("--filter", default="", help="Only profile kernel ids containing this substring")
    return parser.parse_args(argv)


def matches_filter(name: str, raw_filter: str) -> bool:
    if not raw_filter:
        return True
    tokens = [token.strip() for token in raw_filter.split(",") if token.strip()]
    return any(token in name for token in tokens)


def parse_metric_csv(raw: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    csv_lines = [line for line in raw.splitlines() if line.strip() and line.startswith('"')]
    if len(csv_lines) < 3:
        return metrics
    reader = csv.DictReader(io.StringIO("\n".join([csv_lines[0]] + csv_lines[2:])))
    for row in reader:
        for col, value in row.items():
            if not col or not value:
                continue
            cleaned = str(value).replace(",", "").strip()
            try:
                metrics[col] = float(cleaned)
            except ValueError:
                continue
        break
    return metrics


def derive_profile(metrics: dict[str, float], existing: dict[str, Any]) -> dict[str, Any]:
    hardware = existing["hardware"]
    total_flops = metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0) + metrics.get(
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0
    )
    global_load_bytes = metrics.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 0.0)
    global_store_bytes = metrics.get("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", 0.0)
    local_load_bytes = metrics.get("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", 0.0)
    local_store_bytes = metrics.get("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum", 0.0)
    logical_bytes = max(global_load_bytes + global_store_bytes + local_load_bytes + local_store_bytes, 1.0)
    dram_bytes = metrics.get("dram__bytes.sum", global_load_bytes + global_store_bytes)
    requests = max(metrics.get("l2__global_load_requests.sum", 1.0), 1.0)
    misses = metrics.get("sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_global_ld_sectors_miss.sum", 0.0)
    duration_ns = max(metrics.get("gpu__time_duration.sum", 0.0), 1.0)
    duration_s = duration_ns * 1.0e-9
    dram_bw_gbps = (dram_bytes / duration_s) / 1.0e9 if duration_s > 0.0 else 0.0
    profile = {
        "needs_profiling": False,
        "arithmetic_intensity_flop_per_byte": round(total_flops / logical_bytes, 4),
        "achieved_occupancy_pct": round(metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0), 2),
        "dram_bw_utilization_pct": round((dram_bw_gbps / hardware["peak_bw_gbps"]) * 100.0, 4),
        "stall_long_scoreboard_pct": round(metrics.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 0.0), 2),
        "stall_memory_pct": round(metrics.get("smsp__warp_issue_stalled_membar_per_warp_active.pct", 0.0), 2),
        "global_load_efficiency_pct": round(max(0.0, min(100.0, (1.0 - (misses / requests)) * 100.0)), 2),
        "l2_hit_rate_pct": round(max(0.0, min(100.0, (1.0 - (misses / requests)) * 100.0)), 2),
        "register_count_per_thread": int(metrics.get("launch__registers_per_thread", 0.0)),
        "local_memory_bytes": round(local_load_bytes + local_store_bytes, 2),
        "gpu_time_us": round(metrics.get("gpu__time_duration.sum", 0.0) / 1000.0, 3),
        "hardware": hardware,
    }
    verification = verify_profile(profile)
    profile["verification"] = verification
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profiled = 0
    failed = 0
    confidence_counts: dict[str, int] = {}
    failure_lines: list[str] = []
    conflict_lines: list[str] = []
    for kernel_dir in sorted(path for path in args.kernels.iterdir() if path.is_dir()):
        if not matches_filter(kernel_dir.name, args.filter):
            continue
        profile_path = kernel_dir / "profile.json"
        meta_path = kernel_dir / "meta.json"
        kernel_path = kernel_dir / "kernel.cu"
        if not (profile_path.exists() and meta_path.exists() and kernel_path.exists()):
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not profile.get("needs_profiling", False):
            continue
        if profile.get("compile_status") != "success":
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            with tempfile.TemporaryDirectory(prefix=f"{kernel_dir.name}_", dir="/tmp") as temp_dir:
                binary_path = Path(temp_dir) / kernel_dir.name
                subprocess.run(["nvcc", "-O2", "-arch=sm_80", str(kernel_path), "-o", str(binary_path)], check=True, capture_output=True, text=True)
                result = subprocess.run(
                    ["ncu", "--csv", "--page", "raw", "--metrics", ",".join(METRICS), str(binary_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            derived = derive_profile(parse_metric_csv(result.stdout), profile)
            consensus = derived["verification"]["consensus"]
            if consensus != meta["true_bottleneck"]:
                derived["label_conflict"] = True
                derived["ncu_says"] = consensus
                derived["verification"]["confidence"] = "low"
                conflict_lines.append(f"{kernel_dir.name}: meta={meta['true_bottleneck']} ncu={consensus}")
            profile_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
            profiled += 1
            confidence = derived["verification"]["confidence"]
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            failure_lines.append(f"{kernel_dir.name}: {exc.stderr.strip()}")
    args.failure_log.write_text("\n".join(failure_lines) + ("\n" if failure_lines else ""), encoding="utf-8")
    args.conflict_log.write_text("\n".join(conflict_lines) + ("\n" if conflict_lines else ""), encoding="utf-8")
    print(f"Profiled: {profiled}")
    print(f"Failed: {failed}")
    print(f"Confidence breakdown: {dict(sorted(confidence_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
