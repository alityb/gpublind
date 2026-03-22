#include <cuda_runtime.h>
#include <cstdio>

__global__ void traversal_kernel(const float* values, const float* weights, float* out, int n, int tile_span) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    int tile_id = tid >> 5;
    int lane = tid & 31;
    int tile_base = (tile_id * tile_span + ((lane & 7) << 1) + (lane >> 3)) & (n - 1);
    float acc = 0.0f;

    // process tile elements
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        int edge_slot = (tile_base + step * tile_span + (lane & 3)) & (n - 1);
        float x = values[edge_slot];
        float y = weights[edge_slot];
        acc = acc + x * y;
        acc = acc + x * 0.375f;
        acc = acc - y * 0.125f;
        acc = acc + x * y * 0.50f;
        acc = acc + x * x;
        acc = acc - y * y;
        acc = acc + (x + y) * 0.75f;
        acc = acc - (x - y) * 0.25f;
    }

    out[tid] = acc;
}

int main() {
    const int n = 1 << 20;
    const int tile_span = 96;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *values, *weights, *out;
    cudaMalloc(&values, bytes);
    cudaMalloc(&weights, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i % 251) * 0.01f;
    }
    cudaMemcpy(values, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(weights, h, bytes, cudaMemcpyHostToDevice);
    traversal_kernel<<<(n + 255) / 256, 256>>>(values, weights, out, n, tile_span);
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[0]);
    cudaFree(values);
    cudaFree(weights);
    cudaFree(out);
    delete[] h;
    return 0;
}
