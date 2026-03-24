#include <cuda_runtime.h>
#include <cstdio>

__global__ void double_buffer_kernel(const float* a, const float* b, float* out, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n - 1) {
        float v0 = a[idx];
        float v1 = b[idx + 1];
        out[idx] = v0 + v1 * 0.5f;
        idx += stride;
    }

}

int main() {
    const int n = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);
    cudaMemset(d_c, 0, bytes);
    double_buffer_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double_buffer_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    float h = 0.0f;
    cudaMemcpy(&h, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
