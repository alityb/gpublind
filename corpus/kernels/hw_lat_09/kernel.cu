#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void hw_lat_09_kernel(const int* next_idx, const float* values, float* out, int table_mask, int out_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid * 769 + threadIdx.x * 17 + blockIdx.x * 13) & table_mask;
    float acc = values[idx];

    #pragma unroll 1
    for (int iter = 0; iter < 240; ++iter) {
        idx = next_idx[idx];
        float sample = values[idx];
        acc = acc * 1.00146484f + sample * 0.375f;
    }

    out[tid & out_mask] = acc;
}

int main() {
    const int table_size = 1 << 19;
    const int out_size = 1 << 20;
    const int table_mask = table_size - 1;
    const int out_mask = out_size - 1;
    const size_t index_bytes = static_cast<size_t>(table_size) * sizeof(int);
    const size_t value_bytes = static_cast<size_t>(table_size) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_size) * sizeof(float);

    std::vector<int> h_next(table_size);
    std::vector<float> h_values(table_size);
    for (int i = 0; i < table_size; ++i) {
        h_next[i] = (i + 769) & 524287;
        h_values[i] = static_cast<float>((i * 37) & 255) * 0.00390625f;
    }

    int* d_next = nullptr;
    float* d_values = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_next, index_bytes);
    cudaMalloc(&d_values, value_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_next, h_next.data(), index_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), value_bytes, cudaMemcpyHostToDevice);

    hw_lat_09_kernel<<<4096, 256>>>(d_next, d_values, d_out, table_mask, out_mask);
    cudaDeviceSynchronize();

    float sink = 0.0f;
    cudaMemcpy(&sink, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("%f\n", sink);

    cudaFree(d_next);
    cudaFree(d_values);
    cudaFree(d_out);
    return 0;
}
