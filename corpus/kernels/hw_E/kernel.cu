#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 1) void streamed_filter(const float* x, const float* y, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    int mask = n - 1;
    float taps[32];
    float state[96];
    float partial[32];

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int coeff_idx = (blockIdx.x * 17 + i) & mask;
        taps[i] = y[coeff_idx] * (0.125f + 0.00390625f * static_cast<float>(i));
        partial[i] = 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < 96; ++i) {
        int sample_idx = (tid + i) & mask;
        float xv = x[sample_idx];
        float yv = y[(sample_idx + 9) & mask];
        float mix0 = __fmul_rn(xv, 0.5f + 0.00390625f * static_cast<float>(i & 15));
        float mix1 = __fmul_rn(yv, 0.25f - 0.001953125f * static_cast<float>(i & 7));
        state[i] = __fadd_rn(mix0, mix1);
    }

    #pragma unroll
    for (int phase = 0; phase < 3; ++phase) {
        #pragma unroll
        for (int tap = 0; tap < 32; ++tap) {
            float s0 = state[phase * 16 + tap];
            float s1 = state[phase * 16 + tap + 16];
            float coeff0 = taps[tap];
            float coeff1 = taps[(tap + phase * 5 + 7) & 31];
            float prod0 = __fmul_rn(s0, coeff0);
            float prod1 = __fmul_rn(s1, coeff1);
            partial[tap] = __fadd_rn(partial[tap], prod0);
            partial[tap] = __fadd_rn(partial[tap], prod1);
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < 32; ++tap) {
        float blend0 = __fmul_rn(partial[tap], 1.0f / static_cast<float>(tap + 1));
        float blend1 = __fmul_rn(state[tap + 48], 0.03125f);
        acc = __fadd_rn(acc, blend0);
        acc = __fadd_rn(acc, blend1);
    }

    out[tid] = acc;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *x, *y, *out;
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>((i * 5) % 257) * 0.01f;
    }
    cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h, bytes, cudaMemcpyHostToDevice);
    streamed_filter<<<(n + 255) / 256, 256>>>(x, y, out, n);
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[4]);
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    delete[] h;
    return 0;
}
