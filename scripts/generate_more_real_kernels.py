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


def write_entry(root: Path, kernel_id: str, meta: dict, code: str) -> None:
    kernel_dir = root / kernel_id
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.cu").write_text(code, encoding="utf-8")
    (kernel_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (kernel_dir / "profile.json").write_text(json.dumps(PLACEHOLDER_PROFILE, indent=2) + "\n", encoding="utf-8")


def meta(kernel_id: str, label: str, misleading_signal: str, explanation: str) -> dict:
    must_cite = {
        "memory-bound": ["dram bandwidth", "bandwidth", "arithmetic intensity", "global memory"],
        "latency-bound": ["long scoreboard", "dependency", "stall", "dram utilization"],
        "occupancy-limited": ["occupancy", "shared memory", "blocks per SM", "active warps"],
        "register-spill": ["register", "spill", "local memory", "register pressure"],
    }[label]
    must_not = {
        "memory-bound": ["dependency chain", "register spill", "occupancy collapse"],
        "latency-bound": ["dram bandwidth", "global memory saturation", "shared memory tiling"],
        "occupancy-limited": ["memory bandwidth saturation", "dependency chain", "cache miss"],
        "register-spill": ["dram saturation", "occupancy collapse", "shared memory pressure"],
    }[label]
    return {
        "id": kernel_id,
        "source": "handwritten",
        "true_bottleneck": label,
        "misleading_signal": misleading_signal,
        "correct_explanation": explanation,
        "difficulty": "medium",
        "category": label,
        "hardware": "A10G",
        "reasoning_rubric": {
            "must_cite_one_of": must_cite,
            "must_not_cite_as_primary": must_not,
        },
    }


def memory_code(kernel_name: str, body: str, setup: str) -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>

__global__ void {kernel_name}({body}) {{
{setup}
}}

int main() {{
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);
    cudaMemset(d_c, 0, bytes);
    {kernel_name}<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    {kernel_name}<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    float h = 0.0f;
    cudaMemcpy(&h, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\\n", h);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}}
'''


def latency_code(kernel_name: str, inner: str) -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

__global__ void {kernel_name}(const float* in, float* out, int n) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {{
        return;
    }}
    float x = in[tid];
    float acc = x * 0.5f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < 256; ++i) {{
{inner}
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
    {kernel_name}<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    {kernel_name}<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\\n", h[7]);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h;
    return 0;
}}
'''


def occupancy_code(kernel_name: str, block_size: int, shared_floats: int, extra_decl: str, extra_body: str, n_expr: str = "1 << 20") -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = {block_size};
constexpr int SHARED_FLOATS = {shared_floats};

{extra_decl}

__global__ void {kernel_name}(const float* x, const float* y, float* out, int n) {{
    extern __shared__ float scratch[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {{
        return;
    }}
    scratch[threadIdx.x] = x[tid] + y[(tid + 17) & (n - 1)];
    for (int i = blockDim.x + threadIdx.x; i < SHARED_FLOATS; i += blockDim.x) {{
        scratch[i] = scratch[i & (blockDim.x - 1)] * (1.0f + 0.000244140625f * static_cast<float>(i & 31));
    }}
    __syncthreads();
{extra_body}
}}

int main() {{
    const int n = {n_expr};
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t shared_bytes = static_cast<size_t>(SHARED_FLOATS) * sizeof(float);
    float* h = new float[n];
    float *x, *y, *out;
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {{
        h[i] = static_cast<float>((i * 11) % 193) * 0.015625f;
    }}
    cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h, bytes, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute({kernel_name}, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    {kernel_name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    {kernel_name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\\n", h[5]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}}
'''


def register_code(kernel_name: str, state0: int, state1: int, taps: int, mix: int, body_scale: str) -> str:
    return f'''#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 128;
constexpr int STATE0 = {state0};
constexpr int STATE1 = {state1};
constexpr int TAPS = {taps};
constexpr int MIX = {mix};

__global__ __launch_bounds__(BLOCK_SIZE, 1) void {kernel_name}(const float* x, const float* coeff, float* out, int n) {{
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
        taps[i] = coeff[(blockIdx.x * 31 + i) & mask] * ({body_scale} + 0.001953125f * static_cast<float>(i));
    }}
    #pragma unroll
    for (int i = 0; i < STATE0; ++i) {{
        float xv = x[(tid + i) & mask];
        float yv = coeff[(tid + i * 7) & mask];
        state0[i] = xv * (0.75f + 0.0009765625f * static_cast<float>(i & 31)) + yv * 0.25f;
    }}
    #pragma unroll
    for (int i = 0; i < STATE1; ++i) {{
        float xv = x[(tid + i * 3 + 11) & mask];
        float yv = coeff[(tid + i * 5 + 19) & mask];
        state1[i] = xv * (0.625f + 0.00048828125f * static_cast<float>(i & 63)) + yv * 0.375f;
    }}
    #pragma unroll
    for (int i = 0; i < MIX; ++i) {{
        mix[i] = state0[(i * 3) % STATE0] + state1[(i * 5) % STATE1];
    }}
    float acc = 0.0f;
    #pragma unroll
    for (int phase = 0; phase < 8; ++phase) {{
        #pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {{
            float s0 = state0[(phase * 19 + tap) % STATE0];
            float s1 = state1[(phase * 23 + tap + 37) % STATE1];
            float m0 = mix[(tap + phase * 7) & (MIX - 1)];
            float c0 = taps[tap];
            float c1 = taps[(tap + phase * 11 + 7) & (TAPS - 1)];
            acc += s0 * c0;
            acc += s1 * c1;
            acc += m0 * 0.015625f;
        }}
    }}
    out[tid] = acc;
}}

int main() {{
    const int n = 1 << 20;
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
    {kernel_name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
    cudaDeviceSynchronize();
    {kernel_name}<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
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
    root = Path("corpus/kernels")

    memory_entries = {
        "hw_mem_11": memory_code(
            "tile_read_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n - 1024) {
        int tile = idx & ~1023;
        int lane = idx & 1023;
        out[idx] = a[tile + ((lane * 17) & 1023)] + b[idx] * 0.25f;
        idx += stride;
    }
""",
        ),
        "hw_mem_12": memory_code(
            "transpose_like_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int width = 8192;
    while (idx < n) {
        int row = idx / width;
        int col = idx - row * width;
        int transposed = col * width + row;
        out[transposed & (n - 1)] = a[idx] + b[idx] * 0.5f;
        idx += stride;
    }
""",
        ),
        "hw_mem_13": memory_code(
            "histogram_like_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        int bucket = ((int)a[idx] * 1315423911u + idx) & (n - 1);
        out[bucket] = b[idx] + 1.0f;
        idx += stride;
    }
""",
        ),
        "hw_mem_14": memory_code(
            "gather_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        int gather = (idx * 97 + 17) & (n - 1);
        out[idx] = a[gather] + b[(gather * 3) & (n - 1)] * 0.125f;
        idx += stride;
    }
""",
        ),
        "hw_mem_15": memory_code(
            "prefix_like_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n - 8) {
        float acc = a[idx];
        #pragma unroll
        for (int offset = 1; offset <= 8; ++offset) {
            acc += a[idx + offset] * 0.0625f;
        }
        out[idx] = acc + b[idx] * 0.125f;
        idx += stride;
    }
""",
        ),
        "hw_mem_16": memory_code(
            "conv3x3_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float k0 = 0.0625f, k1 = 0.125f, k2 = 0.25f;
    while (idx < n - 4) {
        float acc = a[idx - 1 + (idx == 0)] * k0 + a[idx] * k1 + a[idx + 1] * k2;
        acc += b[idx] * k0 + b[idx + 1] * k1;
        out[idx] = acc;
        idx += stride;
    }
""",
        ),
        "hw_mem_17": memory_code(
            "embedding_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int table_width = 128;
    while (idx < n - table_width) {
        int token = ((int)b[idx] + idx * 17) & ((n / table_width) - 1);
        int base = token * table_width;
        out[idx] = a[base + (idx & (table_width - 1))];
        idx += stride;
    }
""",
        ),
        "hw_mem_18": memory_code(
            "batched_mxv_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int width = 64;
    while (idx < n - width) {
        float acc = 0.0f;
        int base = (idx / width) * width;
        #pragma unroll
        for (int j = 0; j < width; ++j) {
            acc += a[base + j] * b[j] * 0.015625f;
        }
        out[idx] = acc;
        idx += stride;
    }
""",
        ),
        "hw_mem_19": memory_code(
            "fill_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = 3.0f + a[idx] * 0.0f + b[idx] * 0.0f;
        idx += stride;
    }
""",
        ),
        "hw_mem_20": memory_code(
            "double_buffer_kernel",
            "const float* a, const float* b, float* out, int n",
            """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n - 1) {
        float v0 = a[idx];
        float v1 = b[idx + 1];
        out[idx] = v0 + v1 * 0.5f;
        idx += stride;
    }
""",
        ),
    }

    latency_entries = {
        "hw_lat_01": latency_code("pointer_chase_kernel", "        int next = ((int)acc * 17 + i) & 1023;\n        acc = __fdividef(acc + in[next], 1.0001f + 0.0001f * next);"),
        "hw_lat_02": latency_code("fibonacci_kernel", "        float prev = acc;\n        acc = acc + x;\n        x = prev;"),
        "hw_lat_03": latency_code("newton_kernel", "        acc = 0.5f * (acc + x / (acc + 0.001f));"),
        "hw_lat_04": latency_code("bit_chain_kernel", "        int bits = __float_as_int(acc);\n        bits ^= (bits << 5) + i;\n        acc = __int_as_float((bits & 0x007fffff) | 0x3f000000);"),
        "hw_lat_05": latency_code("transcendental_kernel", "        acc = __sinf(acc) + __cosf(acc * 0.5f);\n        acc = __expf(acc * 0.125f);"),
        "hw_lat_06": latency_code("branch_chain_kernel", "        if (((int)acc + i) & 1) acc = acc * 1.0003f + 0.5f; else acc = acc / 1.0007f + 0.25f;"),
        "hw_lat_07": latency_code("shuffle_style_kernel", "        acc = __fadd_rn(acc, x * 0.125f);\n        x = __fadd_rn(acc, x * 0.0625f);"),
        "hw_lat_08": latency_code("atomic_style_kernel", "        acc = __fadd_rn(acc, 0.03125f);\n        acc = __fdividef(acc, 1.0001f + acc * 0.0001f);"),
        "hw_lat_09": latency_code("sqrt_chain_kernel", "        acc = sqrtf(acc + 1.0f);"),
        "hw_lat_10": latency_code("fma_chain_kernel", "        acc = fmaf(acc, 1.0009765625f, 0.03125f);"),
    }

    occupancy_entries = {
        "hw_occ_01": occupancy_code(
            "occ_shmem_48k",
            256,
            12288,
            "",
            """
    float acc = scratch[(threadIdx.x * 13) & (SHARED_FLOATS - 1)];
    #pragma unroll
    for (int iter = 0; iter < 256; ++iter) {
        int base = (threadIdx.x * 97 + iter * 29) & (SHARED_FLOATS - 1);
        acc += scratch[base] * 0.5f + y[(tid + iter * 17) & (n - 1)] * 0.25f;
        acc += scratch[(base + 257) & (SHARED_FLOATS - 1)] * 0.125f;
    }
    out[tid] = acc;
""",
        ),
        "hw_occ_02": occupancy_code(
            "occ_reg_heavy",
            256,
            8192,
            "constexpr int EXTRA = 96;",
            """
    float regs[EXTRA];
    #pragma unroll
    for (int i = 0; i < EXTRA; ++i) {
        regs[i] = scratch[(threadIdx.x + i * 7) & (SHARED_FLOATS - 1)] + x[(tid + i * 13) & (n - 1)];
    }
    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < EXTRA; ++iter) {
        acc += regs[iter] * (0.25f + 0.001f * static_cast<float>(iter));
    }
    out[tid] = acc;
""",
        ),
        "hw_occ_03": occupancy_code(
            "occ_small_block",
            32,
            16384,
            "",
            """
    float acc = scratch[(threadIdx.x * 31) & (SHARED_FLOATS - 1)];
    #pragma unroll
    for (int iter = 0; iter < 512; ++iter) {
        int base = (threadIdx.x * 19 + iter * 7) & (SHARED_FLOATS - 1);
        acc += scratch[base] * 0.5f + y[(tid + iter) & (n - 1)] * 0.125f;
    }
    out[tid] = acc;
""",
        ),
        "hw_occ_04": occupancy_code(
            "occ_double_pressure",
            192,
            16384,
            "constexpr int EXTRA = 64;",
            """
    float regs[EXTRA];
    #pragma unroll
    for (int i = 0; i < EXTRA; ++i) {
        regs[i] = scratch[(threadIdx.x + i * 11) & (SHARED_FLOATS - 1)] + y[(tid + i * 17) & (n - 1)];
    }
    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < EXTRA; ++iter) {
        acc += regs[iter] * (0.5f + 0.001f * static_cast<float>(iter));
    }
    out[tid] = acc;
""",
        ),
        "hw_occ_05": occupancy_code(
            "occ_barrier_loop",
            256,
            12288,
            "",
            """
    float acc = scratch[threadIdx.x];
    #pragma unroll 1
    for (int iter = 0; iter < 96; ++iter) {
        scratch[threadIdx.x] = acc + y[(tid + iter * 9) & (n - 1)] * 0.25f;
        __syncthreads();
        acc = scratch[(threadIdx.x + 37) & (blockDim.x - 1)];
        __syncthreads();
    }
    out[tid] = acc;
""",
        ),
    }

    register_entries = {
        "hw_reg_01": register_code("spill_variant_01", 224, 192, 128, 64, "0.5f"),
        "hw_reg_02": register_code("spill_variant_02", 208, 176, 128, 64, "0.75f"),
        "hw_reg_03": register_code("spill_variant_03", 240, 160, 128, 64, "0.625f"),
        "hw_reg_04": register_code("spill_variant_04", 192, 192, 128, 64, "0.875f"),
    }

    for idx, code in memory_entries.items():
        write_entry(root, idx, meta(idx, "memory-bound", f"{idx} looks like a mixed access pattern, but the real bottleneck is sustained DRAM traffic on a large working set", "the kernel streams hundreds of megabytes with only a handful of FLOPs per element, so bandwidth dominates"), code)
    for idx, code in latency_entries.items():
        write_entry(root, idx, meta(idx, "latency-bound", f"{idx} looks arithmetic-light, but the real issue is a long serialized dependency path with low DRAM pressure", "the kernel keeps DRAM utilization low while scoreboard stalls dominate due to serial dependent operations"), code)
    for idx, code in occupancy_entries.items():
        write_entry(root, idx, meta(idx, "occupancy-limited", f"{idx} appears compute-heavy, but large shared-memory or launch geometry collapses active warps", "the kernel cannot keep enough warps resident to hide latency, so occupancy becomes the binding constraint"), code)
    for idx, code in register_entries.items():
        write_entry(root, idx, meta(idx, "register-spill", f"{idx} looks like a dense filter, but the real bottleneck is excessive live state forcing spills to local memory", "the kernel carries so much live state that register pressure triggers spilling and local-memory latency"), code)

    total = len(memory_entries) + len(latency_entries) + len(occupancy_entries) + len(register_entries)
    print(f"Wrote {total} additional kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
