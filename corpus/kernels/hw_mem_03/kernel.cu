#include <cuda_runtime.h>
#include <cstdio>

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = in[idx];
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 3, bytes);
    copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    copy_kernel<<<4096, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
