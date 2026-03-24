#include <cuda_runtime.h>
#include <cstdio>

__global__ void max_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        float av = a[idx];
        float bv = b[idx];
        out[idx] = av > bv ? av : bv;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_a, 7, bytes);
    cudaMemset(d_b, 8, bytes);
    max_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    max_kernel<<<4096, 256>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
