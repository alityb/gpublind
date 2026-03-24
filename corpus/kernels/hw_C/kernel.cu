#include <cuda_runtime.h>
#include <cstdio>

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

__global__ void suspicious_sync(const float* values, const int* next_slot, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    if (blockIdx.x & 1) {
        // WARNING: sync inside branch
        __syncthreads();
    } else {
        __syncthreads();
    }

    int mask = n - 1;
    int lane = threadIdx.x & 31;
    int cursor = (tid * 37 + lane * 97) & mask;
    float acc = values[(cursor + lane * 13) & mask];
    float scale0 = 0.25f + 0.03125f * static_cast<float>(lane & 7);
    float scale1 = 0.75f - 0.015625f * static_cast<float>(lane & 15);

    #pragma unroll 1
    for (int step = 0; step < 16; ++step) {
        cursor = next_slot[(cursor + step * 119 + lane * 17) & mask];
        float sample0 = values[cursor];
        float sample1 = values[(cursor + 257) & mask];

        #pragma unroll
        for (int mix = 0; mix < 8; ++mix) {
            float coeff0 = scale0 + 0.0078125f * static_cast<float>(mix + 1);
            float coeff1 = scale1 - 0.00390625f * static_cast<float>((mix + step) & 7);
            float t0 = counted_mul(sample0, coeff0);
            float t1 = counted_mul(sample1, coeff1);
            acc = counted_add(acc, t0);
            acc = counted_add(acc, t1);
        }

        scale0 = counted_add(counted_mul(scale0, 1.0009765625f), 0.00390625f);
        scale1 = counted_add(counted_mul(scale1, 0.9990234375f), 0.001953125f);
    }

    out[tid] = acc;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t index_bytes = static_cast<size_t>(n) * sizeof(int);
    float* h = new float[n];
    int* idx = new int[n];
    float *values, *out;
    int* next_slot;
    cudaMalloc(&values, bytes);
    cudaMalloc(&out, bytes);
    cudaMalloc(&next_slot, index_bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>((i * 7) % 193) * 0.02f;
        idx[i] = (i * 131 + 17) & (n - 1);
    }
    cudaMemcpy(values, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(next_slot, idx, index_bytes, cudaMemcpyHostToDevice);
    suspicious_sync<<<(n + 255) / 256, 256>>>(values, next_slot, out, n);
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[2]);
    cudaFree(values);
    cudaFree(out);
    cudaFree(next_slot);
    delete[] h;
    delete[] idx;
    return 0;
}
