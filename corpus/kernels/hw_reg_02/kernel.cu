#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 128;
constexpr int STATE0 = 208;
constexpr int STATE1 = 176;
constexpr int TAPS = 128;
constexpr int MIX = 64;

__global__ __launch_bounds__(BLOCK_SIZE, 1) void spill_variant_02(const float* x, const float* coeff, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    int mask = n - 1;
    float state0[STATE0];
    float state1[STATE1];
    float taps[TAPS];
    float mix[MIX];

    #pragma unroll
    for (int i = 0; i < TAPS; ++i) {
        taps[i] = coeff[(blockIdx.x * 31 + i) & mask] * (0.75f + 0.001953125f * static_cast<float>(i));
    }
    #pragma unroll
    for (int i = 0; i < STATE0; ++i) {
        float xv = x[(tid + i) & mask];
        float yv = coeff[(tid + i * 7) & mask];
        state0[i] = xv * (0.75f + 0.0009765625f * static_cast<float>(i & 31)) + yv * 0.25f;
    }
    #pragma unroll
    for (int i = 0; i < STATE1; ++i) {
        float xv = x[(tid + i * 3 + 11) & mask];
        float yv = coeff[(tid + i * 5 + 19) & mask];
        state1[i] = xv * (0.625f + 0.00048828125f * static_cast<float>(i & 63)) + yv * 0.375f;
    }
    #pragma unroll
    for (int i = 0; i < MIX; ++i) {
        mix[i] = state0[(i * 3) % STATE0] + state1[(i * 5) % STATE1];
    }
    float acc = 0.0f;
    #pragma unroll
    for (int phase = 0; phase < 8; ++phase) {
        #pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {
            float s0 = state0[(phase * 19 + tap) % STATE0];
            float s1 = state1[(phase * 23 + tap + 37) % STATE1];
            float m0 = mix[(tap + phase * 7) & (MIX - 1)];
            float c0 = taps[tap];
            float c1 = taps[(tap + phase * 11 + 7) & (TAPS - 1)];
            acc += s0 * c0;
            acc += s1 * c1;
            acc += m0 * 0.015625f;
        }
    }
    out[tid] = acc;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *x, *coeff, *out;
    cudaMalloc(&x, bytes);
    cudaMalloc(&coeff, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>((i * 13) % 257) * 0.0078125f;
    }
    cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(coeff, h, bytes, cudaMemcpyHostToDevice);
    spill_variant_02<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
    cudaDeviceSynchronize();
    spill_variant_02<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, coeff, out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    printf("%f\n", h[9]);
    cudaFree(x);
    cudaFree(coeff);
    cudaFree(out);
    delete[] h;
    return 0;
}
