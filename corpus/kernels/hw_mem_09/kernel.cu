#include <cuda_runtime.h>
#include <cstdio>

__global__ void broadcast_add_kernel(const float* __restrict__ in, const float* __restrict__ scalar, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = scalar[0];
    while (idx < n) {
        out[idx] = in[idx] + s;
        idx += stride;
    }
}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_in, *d_out, *d_scalar;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_scalar, sizeof(float));
    cudaMemset(d_in, 1, bytes);
    cudaMemset(d_scalar, 0, sizeof(float));
    broadcast_add_kernel<<<4096, 256>>>(d_in, d_scalar, d_out, n);
    cudaDeviceSynchronize();
    broadcast_add_kernel<<<4096, 256>>>(d_in, d_scalar, d_out, n);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_scalar);
    return 0;
}
