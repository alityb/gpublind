from __future__ import annotations

import json
from pathlib import Path

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
    "gpu_time_us": -1,
    "hardware": HARDWARE,
    "verification": {"confidence": "unverified"},
}

META_TEMPLATE = {
    "source": "handwritten",
    "true_bottleneck": "memory-bound",
    "difficulty": "easy",
    "category": "memory-bound",
    "hardware": "A10G",
    "reasoning_rubric": {
        "must_cite_one_of": ["dram bandwidth", "dram_bw", "arithmetic intensity", "global memory", "bandwidth"],
        "must_not_cite_as_primary": ["long scoreboard", "dependency chain", "register spill", "occupancy collapse"],
    },
}

KERNELS = {
    "hw_mem_01": {
        "signal": "simple streaming elementwise scale over a 256MB vector with negligible compute",
        "explanation": "one load and one store per element over 64M floats drives DRAM throughput while arithmetic intensity stays near zero",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(const float* __restrict__ in, float* __restrict__ out, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = in[idx] * scale;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 1, bytes);
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0f, n);
    cudaDeviceSynchronize();
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0f, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_02": {
        "signal": "vector add streams two large inputs and one output with no data reuse",
        "explanation": "two reads and one write per element produce a textbook bandwidth-bound streaming kernel on large vectors",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void vadd_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = a[idx] + b[idx];
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);
    vadd_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    vadd_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_03": {
        "signal": "coalesced streaming copy touches each element once with almost no arithmetic",
        "explanation": "the kernel behaves like a memcpy with one read and one write per element, so DRAM bandwidth dominates runtime",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = in[idx];
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 3, bytes);
    copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_04": {
        "signal": "grid-stride partial reduction streams a huge array with only one add per load",
        "explanation": "each thread performs a long streaming sum over global memory with minimal arithmetic, keeping arithmetic intensity far below the ridge point",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void reduce_kernel(const float* __restrict__ in, float* __restrict__ partial, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (int idx = tid; idx < n; idx += stride) {
        acc += in[idx];
    }
    if (tid < gridDim.x * blockDim.x) {
        partial[tid] = acc;
    }
}

int main() {
    const int n = 1 << 26;
    const int threads = 256;
    const int blocks = 4096;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t partial_bytes = static_cast<size_t>(threads * blocks) * sizeof(float);
    float *d_in, *d_partial;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_partial, partial_bytes);
    cudaMemset(d_in, 1, bytes);
    reduce_kernel<<<blocks, threads>>>(d_in, d_partial, n);
    cudaDeviceSynchronize();
    reduce_kernel<<<blocks, threads>>>(d_in, d_partial, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_partial);
    return 0;
}
''',
    },
    "hw_mem_05": {
        "signal": "streaming copy with a fixed offset defeats tiny-cache reuse and keeps traffic on DRAM",
        "explanation": "the kernel streams a large shifted window through memory, so every output requires a fresh global read and write",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void offset_copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n - 32) {
        out[idx] = in[idx + 32];
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 4, bytes);
    offset_copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    offset_copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_06": {
        "signal": "saxpy-style multiply-add still performs only two FLOPs for three memory operations",
        "explanation": "the fused multiply-add does not create enough arithmetic work to offset the cost of streaming two input vectors and one output vector",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void madd_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = alpha * a[idx] + b[idx];
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 5, bytes);
    cudaMemset(d_b, 6, bytes);
    madd_kernel<<<4096, 256>>>(d_a, d_b, d_out, 1.5f, n);
    cudaDeviceSynchronize();
    madd_kernel<<<4096, 256>>>(d_a, d_b, d_out, 1.5f, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_07": {
        "signal": "double-precision elementwise scale halves element count but still streams a 512MB working set",
        "explanation": "the kernel touches enough double-precision data to saturate DRAM while doing only one multiply per loaded element",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(const double* __restrict__ in, double* __restrict__ out, double scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = in[idx] * scale;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    double *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 1, bytes);
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0, n);
    cudaDeviceSynchronize();
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0, n);
    cudaDeviceSynchronize();
    double h_out = 0.0;
    cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_08": {
        "signal": "row-sum over a huge dense matrix streams one row at a time with one add per element",
        "explanation": "each row sum performs minimal arithmetic relative to the number of bytes read from DRAM, so bandwidth remains the limiting resource",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void row_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (row < rows) {
        float acc = 0.0f;
        int base = row * cols;
        for (int col = 0; col < cols; ++col) {
            acc += in[base + col];
        }
        out[row] = acc;
        row += stride;
    }
}

int main() {
    const int cols = 1024;
    const int rows = 1 << 16;
    const int n = rows * cols;
    const size_t in_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(rows) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemset(d_in, 1, in_bytes);
    row_sum_kernel<<<2048, 256>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    row_sum_kernel<<<2048, 256>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
''',
    },
    "hw_mem_09": {
        "signal": "broadcast add reads one scalar and a huge vector, so the vector stream dominates all cost",
        "explanation": "the scalar stays cached while the 256MB vector still has to be streamed through DRAM with almost no arithmetic",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void broadcast_add_kernel(const float* __restrict__ in, const float* __restrict__ scalar, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = scalar[0];
    while (idx < n) {
        out[idx] = in[idx] + s;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out, *d_scalar;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_scalar, sizeof(float));
    cudaMemset(d_in, 1, bytes);
    cudaMemset(d_scalar, 0, sizeof(float));
    broadcast_add_kernel<<<4096, 256>>>(d_in, d_scalar, d_out, n);
    cudaDeviceSynchronize();
    broadcast_add_kernel<<<4096, 256>>>(d_in, d_scalar, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_scalar);
    return 0;
}
''',
    },
    "hw_mem_10": {
        "signal": "elementwise max streams two input vectors and one output vector with only a compare in between",
        "explanation": "the compare is negligible relative to the amount of data moved, so DRAM bandwidth remains the binding resource",
        "code": r'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void max_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        float av = a[idx];
        float bv = b[idx];
        out[idx] = av > bv ? av : bv;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 7, bytes);
    cudaMemset(d_b, 8, bytes);
    max_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    max_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
''',
    },
}


def write_kernel(root: Path, kernel_id: str, payload: dict[str, str]) -> None:
    kernel_dir = root / kernel_id
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.cu").write_text(payload["code"], encoding="utf-8")
    meta = {
        "id": kernel_id,
        **META_TEMPLATE,
        "misleading_signal": payload["signal"],
        "correct_explanation": payload["explanation"],
    }
    (kernel_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (kernel_dir / "profile.json").write_text(json.dumps(PLACEHOLDER_PROFILE, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = Path("corpus/kernels")
    for kernel_id, payload in KERNELS.items():
        write_kernel(root, kernel_id, payload)
    print(f"Wrote {len(KERNELS)} memory-bound kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
