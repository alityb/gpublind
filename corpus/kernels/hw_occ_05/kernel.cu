#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 256;
constexpr int SHARED_FLOATS = 12288;



__global__ void occ_barrier_loop(const float* x, const float* y, float* out, int n) {
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

    float acc = scratch[threadIdx.x];
    #pragma unroll 1
    for (int iter = 0; iter < 96; ++iter) {
        scratch[threadIdx.x] = acc + y[(tid + iter * 9) & (n - 1)] * 0.25f;
        __syncthreads();
        acc = scratch[(threadIdx.x + 37) & (blockDim.x - 1)];
        __syncthreads();
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
    cudaFuncSetAttribute(occ_barrier_loop, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    occ_barrier_loop<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    occ_barrier_loop<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_bytes>>>(x, y, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\n", h[5]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}
