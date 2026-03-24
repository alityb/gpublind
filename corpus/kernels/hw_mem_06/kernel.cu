#include <cuda_runtime.h>
#include <cstdio>

__global__ void madd_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = alpha * a[idx] + b[idx];
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
    cudaMemset(d_a, 5, bytes);
    cudaMemset(d_b, 6, bytes);
    madd_kernel<<<4096, 256>>>(d_a, d_b, d_out, 1.5f, n);
    cudaDeviceSynchronize();
    madd_kernel<<<4096, 256>>>(d_a, d_b, d_out, 1.5f, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
