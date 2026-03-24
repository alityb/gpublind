#include <cuda_runtime.h>
#include <cstdio>

__global__ void conv3x3_kernel(const float* a, const float* b, float* out, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float k0 = 0.0625f, k1 = 0.125f, k2 = 0.25f;
    while (idx < n - 4) {
        float acc = a[idx - 1 + (idx == 0)] * k0 + a[idx] * k1 + a[idx + 1] * k2;
        acc += b[idx] * k0 + b[idx + 1] * k1;
        out[idx] = acc;
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
    conv3x3_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    conv3x3_kernel<<<4096, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    float h = 0.0f;
    cudaMemcpy(&h, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
