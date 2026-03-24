#include <cuda_runtime.h>
#include <cstdio>

__global__ void row_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (row < rows) {
        float acc = 0.0f;
        int base = row * cols;
        for (int col = 0; col < cols; ++col) {
            acc += in[base + col];
        }
        out[row] = acc;
        row += stride;
    }
}

int main() {
    const int cols = 1024;
    const int rows = 1 << 16;
    const int n = rows * cols;
    const size_t in_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(rows) * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemset(d_in, 1, in_bytes);
    row_sum_kernel<<<2048, 256>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    row_sum_kernel<<<2048, 256>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
