#include <cuda_runtime.h>
#include <cstdio>

// ===== KERNEL CODE START =====
__global__ void k(float* x){ __shared__ int tile[64]; x[threadIdx.x] = tile[threadIdx.x & 63]; }
// ===== KERNEL CODE END =====

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 1, bytes);

    k<<<(n + 255) / 256, 256>>>(d_a);
    cudaDeviceSynchronize();

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
