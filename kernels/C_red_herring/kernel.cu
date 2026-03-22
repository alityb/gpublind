#include <cuda_runtime.h>
#include <cstdio>

__global__ void suspicious_sync(const float* values, const int* next_slot, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    int halo_lane = threadIdx.x & 7;
    if (halo_lane == 0) {
        // WARNING: sync inside branch
        __syncthreads();
    }

    int cursor = next_slot[tid];
    float acc = 0.0f;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        cursor = next_slot[(cursor + step * 17) & (n - 1)];
        acc += values[cursor] * (0.5f + 0.125f * step);
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
