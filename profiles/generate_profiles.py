from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from registry.kernel_entry import NCUProfile

METRICS = [
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
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
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NCU profiles for handwritten kernels")
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--roof", type=Path, default=Path("profiles/hardware_roof.json"))
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def load_roof(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ridge_point_for(roof: dict[str, Any]) -> float:
    return float(roof["peak_flops_tflops"]) / float(roof["peak_bw_tbps"])


def parse_metric_csv(csv_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Metric Name") or row.get("metric") or row.get("ID")
            value = row.get("Metric Value") or row.get("value") or row.get("Metric Value (raw)")
            if not name or value is None:
                continue
            cleaned = str(value).replace(",", "").replace("%", "")
            try:
                metrics[str(name)] = float(cleaned)
            except ValueError:
                continue
    return metrics


def derive_profile(metrics: dict[str, float], roof: dict[str, Any]) -> NCUProfile:
    total_flops = metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0) + metrics.get(
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0
    )
    total_bytes = max(metrics.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 1.0), 1.0)
    arithmetic_intensity = total_flops / total_bytes
    ridge_point = ridge_point_for(roof)
    stall_long = metrics.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 0.0) / 100.0
    stall_short = metrics.get("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", 0.0) / 100.0
    stall_mem = metrics.get("smsp__warp_issue_stalled_membar_per_warp_active.pct", 0.0) / 100.0
    dominant_stall = max(
        {
            "long_scoreboard": stall_long,
            "short_scoreboard": stall_short,
            "memory_dependency": stall_mem,
        }.items(),
        key=lambda item: item[1],
    )[0]
    miss_count = metrics.get("sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_global_ld_sectors_miss.sum", 0.0)
    request_count = max(metrics.get("l2__global_load_requests.sum", 1.0), 1.0)
    load_efficiency = max(0.0, min(1.0, 1.0 - (miss_count / request_count)))
    dram_bw_utilization = max(0.0, min(1.0, total_bytes / (float(roof["peak_bw_tbps"]) * 1.0e12)))
    return NCUProfile(
        arithmetic_intensity=arithmetic_intensity,
        memory_bound=arithmetic_intensity < ridge_point,
        compute_bound=arithmetic_intensity >= ridge_point,
        dominant_stall_type=dominant_stall,
        global_load_efficiency=load_efficiency,
        achieved_occupancy=metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0) / 100.0,
        stall_long_sb_pct=stall_long,
        stall_mem_pct=stall_mem,
        register_count=int(metrics.get("launch__registers_per_thread", 64.0)),
        l2_hit_rate=load_efficiency,
        dram_bw_utilization=dram_bw_utilization,
        raw={**metrics, "ridge_point": ridge_point, "roof": roof},
    )


def compile_kernel(source_path: Path, binary_path: Path) -> None:
    subprocess.run(["nvcc", "-O2", "-arch=sm_80", str(source_path), "-o", str(binary_path)], check=True)


def run_ncu(binary_path: Path, output_prefix: Path) -> Path:
    subprocess.run(
        [
            "ncu",
            "--csv",
            "--page",
            "raw",
            "--metrics",
            ",".join(METRICS),
            "-o",
            str(output_prefix),
            str(binary_path),
        ],
        check=True,
    )
    csv_candidate = output_prefix.with_suffix(".csv")
    if csv_candidate.exists():
        return csv_candidate
    return output_prefix.parent / f"{output_prefix.name}.csv"


def fixture_payloads() -> dict[str, dict[str, Any]]:
    roof = {"gpu_name": "Mock A100", "peak_bw_tbps": 1.85, "peak_flops_tflops": 68.0, "measured": True}
    ridge_point = ridge_point_for(roof)
    return {
        "hw_A": {
            "arithmetic_intensity": 14.2,
            "memory_bound": True,
            "compute_bound": False,
            "dominant_stall_type": "memory_dependency",
            "global_load_efficiency": 0.31,
            "achieved_occupancy": 0.64,
            "stall_long_sb_pct": 0.18,
            "stall_mem_pct": 0.42,
            "register_count": 64,
            "l2_hit_rate": 0.22,
            "dram_bw_utilization": 0.84,
            "raw": {"ridge_point": ridge_point, "latency_ms": 2.4, "roof": roof},
        },
        "hw_B": {
            "arithmetic_intensity": 1.8,
            "memory_bound": False,
            "compute_bound": False,
            "dominant_stall_type": "long_scoreboard",
            "global_load_efficiency": 0.94,
            "achieved_occupancy": 0.58,
            "stall_long_sb_pct": 0.41,
            "stall_mem_pct": 0.12,
            "register_count": 72,
            "l2_hit_rate": 0.88,
            "dram_bw_utilization": 0.29,
            "raw": {"ridge_point": ridge_point, "latency_ms": 1.7, "roof": roof},
        },
        "hw_C": {
            "arithmetic_intensity": 5.1,
            "memory_bound": True,
            "compute_bound": False,
            "dominant_stall_type": "memory_dependency",
            "global_load_efficiency": 0.43,
            "achieved_occupancy": 0.61,
            "stall_long_sb_pct": 0.14,
            "stall_mem_pct": 0.37,
            "register_count": 56,
            "l2_hit_rate": 0.34,
            "dram_bw_utilization": 0.77,
            "raw": {"ridge_point": ridge_point, "latency_ms": 2.0, "roof": roof},
        },
        "hw_D": {
            "arithmetic_intensity": 9.8,
            "memory_bound": False,
            "compute_bound": False,
            "dominant_stall_type": "long_scoreboard",
            "global_load_efficiency": 0.91,
            "achieved_occupancy": 0.22,
            "stall_long_sb_pct": 0.34,
            "stall_mem_pct": 0.16,
            "register_count": 88,
            "l2_hit_rate": 0.79,
            "dram_bw_utilization": 0.36,
            "raw": {"ridge_point": ridge_point, "latency_ms": 2.8, "roof": roof},
        },
        "hw_E": {
            "arithmetic_intensity": 4.0,
            "memory_bound": False,
            "compute_bound": False,
            "dominant_stall_type": "register_spill",
            "global_load_efficiency": 0.52,
            "achieved_occupancy": 0.49,
            "stall_long_sb_pct": 0.25,
            "stall_mem_pct": 0.28,
            "register_count": 164,
            "l2_hit_rate": 0.41,
            "dram_bw_utilization": 0.48,
            "raw": {"ridge_point": ridge_point, "latency_ms": 2.2, "roof": roof},
        },
    }


def write_mock_fixtures(profile_dir: Path) -> None:
    fixtures_dir = profile_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for kernel_id, payload in fixture_payloads().items():
        (fixtures_dir / f"{kernel_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_mock_roof(roof_path: Path) -> None:
    roof_path.parent.mkdir(parents=True, exist_ok=True)
    roof_path.write_text(
        json.dumps({"gpu_name": "Mock A100", "peak_bw_tbps": 1.85, "peak_flops_tflops": 68.0, "measured": True}, indent=2),
        encoding="utf-8",
    )


def load_kernel_metadata(kernels_dir: Path) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    for meta_path in sorted(kernels_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["kernel_dir"] = str(meta_path.parent)
        kernels.append(meta)
    return kernels


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.profiles.mkdir(parents=True, exist_ok=True)
    write_mock_fixtures(args.profiles)
    if args.mock:
        write_mock_roof(args.roof)
        return 0
    if not args.roof.exists():
        raise RuntimeError("Missing hardware roof file. Run `python -m profiles.measure_roof` first.")
    if shutil.which("nvcc") is None or shutil.which("ncu") is None:
        raise RuntimeError("nvcc and ncu are required unless --mock is used")
    roof = load_roof(args.roof)
    for meta in load_kernel_metadata(args.kernels):
        kernel_id = str(meta["id"])
        profile_path = args.profiles / f"{kernel_id}.json"
        if profile_path.exists():
            continue
        kernel_dir = Path(meta["kernel_dir"])
        source_path = kernel_dir / "kernel.cu"
        binary_path = kernel_dir / kernel_id
        output_prefix = kernel_dir / kernel_id
        compile_kernel(source_path, binary_path)
        csv_path = run_ncu(binary_path, output_prefix)
        metrics = parse_metric_csv(csv_path)
        profile = derive_profile(metrics, roof)
        profile_path.write_text(json.dumps(profile.__dict__, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
