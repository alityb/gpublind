from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class HardwareRoof:
    gpu_name: str
    peak_bw_tbps: float
    peak_flops_tflops: float
    measured: bool


BW_SOURCE = r'''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void stream_copy(const float* __restrict__ src,
                             float* __restrict__ dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    // 512MB buffer — well above A10G L2 (6MB) to force DRAM traffic
    const int n = 1 << 27;
    const int warmup = 5;
    const int iters = 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *src, *dst;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMemset(src, 1, bytes);
    cudaMemset(dst, 0, bytes);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // warmup
    for (int i = 0; i < warmup; ++i)
        stream_copy<<<blocks, threads>>>(src, dst, n);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        stream_copy<<<blocks, threads>>>(src, dst, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double moved = static_cast<double>(bytes) * 2.0 * iters;
    double tbps = moved / (static_cast<double>(ms) / 1000.0) / 1.0e12;
    std::printf("%0.6f\n", tbps);
    cudaFree(src);
    cudaFree(dst);
    return 0;
}
'''

FLOPS_SOURCE = r'''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void fma_bench(float* out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + tid * 0.00001f;
    float b = 2.0f;
    float c = 3.0f;
    #pragma unroll 8
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
        a = fmaf(a, b, c);
    }
    out[tid] = a;
}

int main() {
    const int threads = 256;
    const int blocks = 4096;
    const int iters = 4096;
    float* out;
    cudaMalloc(&out, static_cast<size_t>(threads) * blocks * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    fma_bench<<<blocks, threads>>>(out, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double flops = static_cast<double>(blocks) * threads * iters * 8.0 * 2.0;
    double tflops = flops / (static_cast<double>(ms) / 1000.0) / 1.0e12;
    std::printf("%0.6f\n", tflops);
    cudaFree(out);
    return 0;
}
'''


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure empirical GPU roofline")
    parser.add_argument("--output", type=Path, default=Path("profiles/hardware_roof.json"))
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def detect_gpu_name() -> str:
    if shutil.which("nvidia-smi") is None:
        return "Mock GPU"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return names[0] if names else "Unknown GPU"


def build_and_run(source: str, workdir: Path, stem: str) -> float:
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is required to measure the empirical roof")
    source_path = workdir / f"{stem}.cu"
    binary_path = workdir / stem
    source_path.write_text(source, encoding="utf-8")
    subprocess.run(["nvcc", "-O3", "-arch=sm_80", str(source_path), "-o", str(binary_path)], check=True)
    result = subprocess.run([str(binary_path)], check=True, capture_output=True, text=True)
    return float(result.stdout.strip().splitlines()[-1])


def measure_hardware_roof() -> HardwareRoof:
    with tempfile.TemporaryDirectory(prefix="gpublind_roof_") as temp_dir:
        workdir = Path(temp_dir)
        peak_bw = build_and_run(BW_SOURCE, workdir, "bw_bench")
        peak_flops = build_and_run(FLOPS_SOURCE, workdir, "flops_bench")
    return HardwareRoof(
        gpu_name=detect_gpu_name(),
        peak_bw_tbps=peak_bw,
        peak_flops_tflops=peak_flops,
        measured=True,
    )


def mock_roof() -> HardwareRoof:
    return HardwareRoof(gpu_name="Mock A100", peak_bw_tbps=1.85, peak_flops_tflops=68.0, measured=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roof = mock_roof() if args.mock else measure_hardware_roof()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(roof), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
