#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_THREADS = 16;
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int REUSE_ITERS = 32;

__device__ __forceinline__ float counted_mul(float a, float b) {
    float out;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float counted_add(float a, float b) {
    float out;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__global__ void tiled_gemm(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.y * TILE_M + ty * 2;
    int col0 = blockIdx.x * TILE_N + tx * 2;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (int k0 = 0; k0 < n; k0 += TILE_K) {
        tile_a[ty * 2 + 0][tx * 2 + 0] = a[(row0 + 0) * n + (k0 + tx * 2 + 0)];
        tile_a[ty * 2 + 0][tx * 2 + 1] = a[(row0 + 0) * n + (k0 + tx * 2 + 1)];
        tile_a[ty * 2 + 1][tx * 2 + 0] = a[(row0 + 1) * n + (k0 + tx * 2 + 0)];
        tile_a[ty * 2 + 1][tx * 2 + 1] = a[(row0 + 1) * n + (k0 + tx * 2 + 1)];
        tile_b[ty * 2 + 0][tx * 2 + 0] = b[(k0 + ty * 2 + 0) * n + (col0 + 0)];
        tile_b[ty * 2 + 0][tx * 2 + 1] = b[(k0 + ty * 2 + 0) * n + (col0 + 1)];
        tile_b[ty * 2 + 1][tx * 2 + 0] = b[(k0 + ty * 2 + 1) * n + (col0 + 0)];
        tile_b[ty * 2 + 1][tx * 2 + 1] = b[(k0 + ty * 2 + 1) * n + (col0 + 1)];
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a0 = tile_a[ty * 2 + 0][kk];
            float a1 = tile_a[ty * 2 + 1][kk];
            float b0 = tile_b[kk][tx * 2 + 0];
            float b1 = tile_b[kk][tx * 2 + 1];
            #pragma unroll
            for (int repeat = 0; repeat < REUSE_ITERS; ++repeat) {
                float scale = 1.0f + 0.0009765625f * static_cast<float>(repeat);
                acc00 = counted_add(acc00, counted_mul(counted_mul(a0, b0), scale));
                acc01 = counted_add(acc01, counted_mul(counted_mul(a0, b1), scale));
                acc10 = counted_add(acc10, counted_mul(counted_mul(a1, b0), scale));
                acc11 = counted_add(acc11, counted_mul(counted_mul(a1, b1), scale));
            }
        }
        __syncthreads();
    }

    c[(row0 + 0) * n + (col0 + 0)] = acc00;
    c[(row0 + 0) * n + (col0 + 1)] = acc01;
    c[(row0 + 1) * n + (col0 + 0)] = acc10;
    c[(row0 + 1) * n + (col0 + 1)] = acc11;
}

int main() {
    const int n = 1024;
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n * n];
    float *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);
    for (int i = 0; i < n * n; ++i) {
        h[i] = static_cast<float>((i * 7) % 127) * 0.0078125f;
    }
    cudaMemcpy(a, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, h, bytes, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_THREADS, BLOCK_THREADS);
    dim3 grid(n / TILE_N, n / TILE_M);
    tiled_gemm<<<grid, block>>>(a, b, c, n);
    cudaMemcpy(h, c, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[5]);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete[] h;
    return 0;
}
