#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 256;
constexpr int STAGES = 2;
constexpr int HISTORY = 16;
constexpr int REUSE_ITERS = 12;

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

__global__ void tiled_update(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[STAGES][BLOCK_SIZE];
    __shared__ float tile_b[STAGES][BLOCK_SIZE];
    __shared__ float tile_accum[STAGES][HISTORY][BLOCK_SIZE];

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    if (row >= n) {
        return;
    }

    int mask = n - 1;

    // optimized tiling for L1 reuse
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        int base = (blockIdx.x * (blockDim.x * STAGES) + stage * blockDim.x + tid) & mask;
        float av = a[base];
        float bv = b[(base + stage * 97) & mask];
        tile_a[stage][tid] = av;
        tile_b[stage][tid] = bv;

        #pragma unroll
        for (int tap = 0; tap < HISTORY; ++tap) {
            float seeded0 = counted_mul(av, 0.125f + 0.015625f * static_cast<float>(tap));
            float seeded1 = counted_mul(bv, 0.5f - 0.0078125f * static_cast<float>(tap));
            tile_accum[stage][tap][tid] = counted_add(seeded0, seeded1);
        }
    }
    __syncthreads();

    float acc = 0.0f;
    float carry = 0.0f;
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        #pragma unroll
        for (int repeat = 0; repeat < REUSE_ITERS; ++repeat) {
            #pragma unroll
            for (int tap = 0; tap < HISTORY; ++tap) {
                int neighbor = (tid + tap * 13 + repeat * 7) & (BLOCK_SIZE - 1);
                float lhs = tile_accum[stage][tap][neighbor];
                float rhs = tile_accum[stage][(tap + repeat + 3) & (HISTORY - 1)][tid];
                float prod0 = counted_mul(lhs, 0.03125f * static_cast<float>(tap + 1));
                float prod1 = counted_mul(rhs, 0.015625f * static_cast<float>(repeat + 1));
                float prod2 = counted_mul(tile_a[stage][neighbor], 0.0625f);
                float prod3 = counted_mul(tile_b[stage][tid], 0.03125f);
                acc = counted_add(acc, prod0);
                acc = counted_add(acc, prod1);
                carry = counted_add(carry, prod2);
                carry = counted_add(carry, prod3);
            }
        }
    }

    c[row] = counted_add(acc, carry);
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
    tiled_update<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, n);
    cudaMemcpy(h, c, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[3]);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete[] h;
    return 0;
}
