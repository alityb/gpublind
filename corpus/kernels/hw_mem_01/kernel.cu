#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(const float* __restrict__ in, float* __restrict__ out, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        out[idx] = in[idx] * scale;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 1, bytes);
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0f, n);
    cudaDeviceSynchronize();
    scale_kernel<<<4096, 256>>>(d_in, d_out, 2.0f, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
