#include <cuda_runtime.h>
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
