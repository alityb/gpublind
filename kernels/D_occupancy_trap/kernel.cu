#include <cuda_runtime.h>
#include <cstdio>

constexpr int TILE = 32;
constexpr int STAGES = 4;

__global__ void tiled_update(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[STAGES][TILE][TILE];
    __shared__ float tile_b[STAGES][TILE][TILE];
    __shared__ float tile_accum[STAGES][TILE][TILE];
    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    if (row >= n) {
        return;
    }

    // optimized tiling for L1 reuse
    for (int stage = 0; stage < STAGES; ++stage) {
        int col = (tid + stage * 7) & (TILE - 1);
        tile_a[stage][tid & (TILE - 1)][col] = a[(row + stage * TILE + col) & (n - 1)];
        tile_b[stage][tid & (TILE - 1)][col] = b[(row + stage * TILE + col * 3) & (n - 1)];
        tile_accum[stage][tid & (TILE - 1)][col] = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float lhs = tile_a[stage][tid & (TILE - 1)][k];
            float rhs = tile_b[stage][k][tid & (TILE - 1)];
            acc += lhs * rhs;
        }
    }
    c[row] = acc;
}

int main() {
    const int n = 1 << 16;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i % 71) * 0.1f;
    }
    cudaMemcpy(a, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, h, bytes, cudaMemcpyHostToDevice);
    tiled_update<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaMemcpy(h, c, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[3]);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete[] h;
    return 0;
}
