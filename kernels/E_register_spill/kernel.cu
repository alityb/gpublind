#include <cuda_runtime.h>
#include <cstdio>

__global__ void streamed_filter(const float* x, const float* y, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    float moments[24];
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        moments[i] = 0.0f;
    }

    #pragma unroll
    for (int tap = 0; tap < 24; ++tap) {
        int idx = (tid + tap) & (n - 1);
        float xv = x[idx];
        float yv = y[idx];
        moments[tap] = xv * (0.25f + 0.015625f * tap) + yv * (0.5f - 0.0078125f * tap);
    }

    float acc = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < 24; ++tap) {
        acc += moments[tap] * (1.0f / (tap + 1.0f));
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
