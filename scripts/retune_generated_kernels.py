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


def write_kernel(kernel_id: str, label: str, explanation: str, code: str) -> None:
    root = Path("corpus/kernels") / kernel_id
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    meta["true_bottleneck"] = label
    meta["category"] = label
    meta["correct_explanation"] = explanation
    (root / "kernel.cu").write_text(code, encoding="utf-8")
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (root / "profile.json").write_text(json.dumps(PLACEHOLDER_PROFILE, indent=2) + "\n", encoding="utf-8")


def latency_code(name: str, body: str) -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

__global__ void {name}(const float* in, float* out, int n) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {{
        return;
    }}
    float x = in[tid];
    float acc = x * 0.5f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < 32; ++i) {{
{body}
    }}
    out[tid] = acc;
}}

int main() {{
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    for (int i = 0; i < n; ++i) {{
        h[i] = static_cast<float>((i % 97) + 1) * 0.015625f;
    }}
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);
    {name}<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    {name}<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\\n", h[11]);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h;
    return 0;
}}
'''


def register_code(name: str, s0: int, s1: int, taps_n: int, mix_n: int) -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 128;
constexpr int STATE0 = {s0};
constexpr int STATE1 = {s1};
constexpr int TAPS = {taps_n};
constexpr int MIX = {mix_n};

__global__ __launch_bounds__(BLOCK_SIZE, 1) void {name}(const float* x, const float* coeff, float* out, int n) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {{
        return;
    }}
    int mask = n - 1;
    float state0[STATE0];
    float state1[STATE1];
    float taps[TAPS];
    float mix[MIX];

    #pragma unroll
    for (int i = 0; i < TAPS; ++i) {{
        taps[i] = coeff[(blockIdx.x * 17 + i) & mask] * (0.5f + 0.001f * static_cast<float>(i));
    }}
    #pragma unroll
    for (int i = 0; i < STATE0; ++i) {{
        state0[i] = x[(tid + i) & mask] * (0.75f + 0.0005f * static_cast<float>(i & 31));
    }}
    #pragma unroll
    for (int i = 0; i < STATE1; ++i) {{
        state1[i] = coeff[(tid + i * 3 + 7) & mask] * (0.625f + 0.00025f * static_cast<float>(i & 63));
    }}
    #pragma unroll
    for (int i = 0; i < MIX; ++i) {{
        mix[i] = state0[(i * 5) % STATE0] + state1[(i * 7) % STATE1];
    }}

    float acc = 0.0f;
    #pragma unroll
    for (int phase = 0; phase < 12; ++phase) {{
        #pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {{
            float a = state0[(phase * 13 + tap) % STATE0];
            float b = state1[(phase * 19 + tap + 11) % STATE1];
            float c = mix[(tap + phase * 7) & (MIX - 1)];
            acc += a * taps[tap];
            acc += b * taps[(tap + phase + 3) & (TAPS - 1)];
            acc += c * 0.015625f;
        }}
    }}
    out[tid] = acc;
}}

int main() {{
    const int n = 1 << 18;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *x, *coeff, *out;
    cudaMalloc(&x, bytes);
    cudaMalloc(&coeff, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {{
        h[i] = static_cast<float>((i * 13) % 257) * 0.0078125f;
    }}
    cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(coeff, h, bytes, cudaMemcpyHostToDevice);
    {name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
    cudaDeviceSynchronize();
    {name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\\n", h[9]);
    cudaFree(x);
    cudaFree(coeff);
    cudaFree(out);
    delete[] h;
    return 0;
}}
'''


def main() -> int:
    latency_bodies = {
        "hw_lat_01": "        acc = __fdividef(acc + 0.03125f, acc + 1.0001f);",
        "hw_lat_02": "        float prev = acc; acc = acc + x * 0.125f; x = prev * 0.5f + 0.25f;",
        "hw_lat_03": "        acc = 0.5f * (acc + x / (acc + 0.001f));",
        "hw_lat_04": "        int bits = __float_as_int(acc); bits ^= (bits << 5) + i; acc = __int_as_float((bits & 0x007fffff) | 0x3f000000);",
        "hw_lat_05": "        acc = __sinf(acc); acc = __cosf(acc + 0.125f);",
        "hw_lat_06": "        acc = ((i & 1) ? (acc / 1.0007f) : (acc * 1.0003f)) + 0.03125f;",
        "hw_lat_07": "        acc = fmaf(acc, 1.0009765625f, 0.03125f);",
        "hw_lat_08": "        acc = __fdividef(acc + 0.125f, 1.0f + acc * 0.03125f);",
        "hw_lat_09": "        acc = sqrtf(acc + 1.0f);",
        "hw_lat_10": "        acc = fmaf(acc, acc, 0.03125f) / (acc + 1.0f);",
    }
    for kernel_id, body in latency_bodies.items():
        write_kernel(
            kernel_id,
            "latency-bound",
            "the kernel performs a serial dependent chain with low DRAM pressure, so long-scoreboard latency dominates",
            latency_code(kernel_id + "_kernel", body),
        )

    write_kernel(
        "hw_mem_12",
        "memory-bound",
        "the kernel streams a large source vector into a strided destination, so bandwidth dominates despite the irregular write pattern",
        '''#include <cuda_runtime.h>
#include <cstdio>

__global__ void transpose_like_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        int dst = (idx * 33) & (n - 1);
        out[dst] = a[idx] + b[idx] * 0.5f;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);
    cudaMemset(d_c, 0, bytes);
    transpose_like_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    transpose_like_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    float h = 0.0f;
    cudaMemcpy(&h, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
''',
    )

    write_kernel(
        "hw_occ_02",
        "occupancy-limited",
        "the kernel carries moderate register pressure plus large shared-memory state, collapsing occupancy before bandwidth saturates",
        '''#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 128;
constexpr int SHARED_FLOATS = 16384;
constexpr int EXTRA = 64;

__global__ void occ_reg_heavy(const float* x, const float* y, float* out, int n) {
    extern __shared__ float scratch[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    scratch[threadIdx.x] = x[tid] + y[(tid + 17) & (n - 1)];
    for (int i = blockDim.x + threadIdx.x; i < SHARED_FLOATS; i += blockDim.x) {
        scratch[i] = scratch[i & (blockDim.x - 1)] * (1.0f + 0.000244140625f * static_cast<float>(i & 31));
    }
    __syncthreads();
    float regs[EXTRA];
    #pragma unroll
    for (int i = 0; i < EXTRA; ++i) {
        regs[i] = scratch[(threadIdx.x + i * 7) & (SHARED_FLOATS - 1)];
    }
    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < EXTRA; ++iter) {
        acc += regs[iter] * (0.25f + 0.001f * static_cast<float>(iter));
    }
    out[tid] = acc;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t shared_bytes = static_cast<size_t>(SHARED_FLOATS) * sizeof(float);
    float* h = new float[n];
    float *x, *y, *out;
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>((i * 11) % 193) * 0.015625f;
    }
    cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h, bytes, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(occ_reg_heavy, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    occ_reg_heavy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    occ_reg_heavy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\n", h[5]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}
''',
    )

    reg_params = {
        "hw_reg_01": (240, 192, 128, 64),
        "hw_reg_02": (224, 176, 128, 64),
        "hw_reg_03": (256, 160, 128, 64),
        "hw_reg_04": (208, 208, 128, 64),
    }
    for kernel_id, params in reg_params.items():
        write_kernel(
            kernel_id,
            "register-spill",
            "the kernel keeps hundreds of live values in flight, forcing spill traffic into local memory instead of DRAM saturation",
            register_code(kernel_id + "_kernel", *params),
        )

    print("Retuned latency/register/occupancy kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
