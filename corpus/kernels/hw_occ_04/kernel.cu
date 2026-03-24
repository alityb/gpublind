#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 192;
constexpr int SHARED_FLOATS = 16384;

constexpr int EXTRA = 64;

__global__ void occ_double_pressure(const float* x, const float* y, float* out, int n) {
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
        regs[i] = scratch[(threadIdx.x + i * 11) & (SHARED_FLOATS - 1)] + y[(tid + i * 17) & (n - 1)];
    }
    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < EXTRA; ++iter) {
        acc += regs[iter] * (0.5f + 0.001f * static_cast<float>(iter));
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
    cudaFuncSetAttribute(occ_double_pressure, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    occ_double_pressure<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    occ_double_pressure<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\n", h[5]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}
