from __future__ import annotations

import json
from pathlib import Path

PLACEHOLDER = {
    "needs_profiling": True,
    "arithmetic_intensity_flop_per_byte": -1,
    "achieved_occupancy_pct": -1,
    "dram_bw_utilization_pct": -1,
    "stall_long_scoreboard_pct": -1,
    "stall_memory_pct": -1,
    "global_load_efficiency_pct": -1,
    "l2_hit_rate_pct": -1,
    "register_count_per_thread": -1,
    "local_memory_bytes": -1,
    "gpu_time_us": -1,
    "hardware": {
        "name": "NVIDIA A10G",
        "peak_bw_gbps": 496.0,
        "peak_flops_tflops": 30.77,
        "ridge_point_flop_per_byte": 62.07,
    },
    "verification": {"confidence": "unverified"},
}


def kernel_source(name: str, table_bits: int, stride: int, steps: int, mix: str) -> str:
    table_size = 1 << table_bits
    mask = table_size - 1
    return f'''#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void {name}(const int* next_idx, const float* values, float* out, int table_mask, int out_mask) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid * {stride} + threadIdx.x * 17 + blockIdx.x * 13) & table_mask;
    float acc = values[idx];

    #pragma unroll 1
    for (int iter = 0; iter < {steps}; ++iter) {{
        idx = next_idx[idx];
        float sample = values[idx];
{mix}
    }}

    out[tid & out_mask] = acc;
}}

int main() {{
    const int table_size = 1 << {table_bits};
    const int out_size = 1 << 20;
    const int table_mask = table_size - 1;
    const int out_mask = out_size - 1;
    const size_t index_bytes = static_cast<size_t>(table_size) * sizeof(int);
    const size_t value_bytes = static_cast<size_t>(table_size) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_size) * sizeof(float);

    std::vector<int> h_next(table_size);
    std::vector<float> h_values(table_size);
    for (int i = 0; i < table_size; ++i) {{
        h_next[i] = (i + {stride}) & {mask};
        h_values[i] = static_cast<float>((i * 37) & 255) * 0.00390625f;
    }}

    int* d_next = nullptr;
    float* d_values = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_next, index_bytes);
    cudaMalloc(&d_values, value_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_next, h_next.data(), index_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), value_bytes, cudaMemcpyHostToDevice);

    {name}<<<4096, 256>>>(d_next, d_values, d_out, table_mask, out_mask);
    cudaDeviceSynchronize();

    float sink = 0.0f;
    cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("%f\\n", sink);

    cudaFree(d_next);
    cudaFree(d_values);
    cudaFree(d_out);
    return 0;
}}
'''


def main() -> int:
    variants = {
        "hw_lat_01": (18, 97, 256, "        acc = acc * 0.99951172f + sample;"),
        "hw_lat_02": (18, 131, 320, "        acc = (acc + sample) * 0.99804688f;"),
        "hw_lat_03": (17, 193, 384, "        acc = acc * 1.00097656f + sample * 0.5f;"),
        "hw_lat_04": (18, 257, 288, "        acc = (acc - sample * 0.25f) * 1.00195312f;"),
        "hw_lat_05": (19, 65, 224, "        acc = acc + sample * 0.75f + 0.00195312f;"),
        "hw_lat_06": (18, 385, 352, "        acc = (acc * 0.5f + sample) * 1.00048828f;"),
        "hw_lat_07": (17, 449, 416, "        acc = acc * 0.99902344f + sample * 0.625f;"),
        "hw_lat_08": (18, 513, 272, "        acc = (acc + sample * 0.875f) * 0.99951172f;"),
        "hw_lat_09": (19, 769, 240, "        acc = acc * 1.00146484f + sample * 0.375f;"),
        "hw_lat_10": (18, 897, 336, "        acc = (acc - sample * 0.125f) * 1.00097656f;"),
    }
    for kid, (table_bits, stride, steps, mix) in variants.items():
        root = Path("corpus/kernels") / kid
        meta_path = root / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["correct_explanation"] = (
            "dependent pointer-chasing loads keep warps waiting on serialized cache accesses, "
            "so long-scoreboard stalls stay high while DRAM utilization remains low"
        )
        meta["misleading_signal"] = (
            "the kernel performs many global loads, but the small pointer table fits in cache so the bottleneck is latency, not DRAM bandwidth"
        )
        meta["reasoning_rubric"] = {
            "must_cite_one_of": [
                "stall_long_scoreboard_pct",
                "long scoreboard",
                "dram_bw_utilization_pct",
                "dependency chain",
                "pointer chasing",
            ],
            "must_not_cite_as_primary": [
                "memory bandwidth",
                "global load efficiency",
                "coalescing",
                "dram saturation",
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (root / "kernel.cu").write_text(kernel_source(kid + "_kernel", table_bits, stride, steps, mix), encoding="utf-8")
        (root / "profile.json").write_text(json.dumps(PLACEHOLDER, indent=2) + "\n", encoding="utf-8")
    print("Overwrote hw_lat_01..10 with pointer-chasing latency kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
