#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 128;
constexpr int SHARED_FLOATS = 24576;
constexpr int OUTER_ITERS = 96;
constexpr int REPEAT_ITERS = 64;

__device__ __forceinline__ float counted_mul(float a, float b) {
    float out;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float counted_add(float a, float b) {
    float out;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__global__ void occupancy_reference(const float* x, const float* y, float* out, int n) {
    extern __shared__ float scratch[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    // optimized tiling for L1 reuse
    scratch[threadIdx.x] = x[tid] + y[(tid + 17) & (n - 1)];
    for (int i = blockDim.x + threadIdx.x; i < SHARED_FLOATS; i += blockDim.x) {
        scratch[i] = scratch[i & (blockDim.x - 1)] * (1.0f + 0.000244140625f * static_cast<float>(i & 31));
    }
    __syncthreads();

    float acc = scratch[(threadIdx.x * 13) & (SHARED_FLOATS - 1)];
    #pragma unroll
    for (int iter = 0; iter < OUTER_ITERS; ++iter) {
        int base = (threadIdx.x * 97 + iter * 29) & (SHARED_FLOATS - 1);
        int global_idx = (tid + iter * 257) & (n - 1);
        float g0 = x[global_idx];
        float g1 = y[(global_idx + 17) & (n - 1)];
        float s0 = scratch[base];
        float s1 = scratch[(base + 257) & (SHARED_FLOATS - 1)];
        float s2 = scratch[(base + 1023) & (SHARED_FLOATS - 1)];
        #pragma unroll
        for (int repeat = 0; repeat < REPEAT_ITERS; ++repeat) {
            float scale = 1.0f + 0.0009765625f * static_cast<float>(repeat + iter);
            float prod0 = counted_mul(counted_add(s0, g0), scale);
            float prod1 = counted_mul(counted_add(s1, g1), 0.5f * scale);
            float prod2 = counted_mul(counted_add(s2, g0), 0.25f);
            float prod3 = counted_mul(counted_add(g1, s0), 0.125f);
            acc = counted_add(acc, prod0);
            acc = counted_add(acc, prod1);
            acc = counted_add(acc, prod2);
            acc = counted_add(acc, prod3);
        }
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
    cudaFuncSetAttribute(occupancy_reference, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    occupancy_reference<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[7]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}
