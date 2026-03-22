#include <cuda_runtime.h>
#include <cstdio>

__global__ void feature_transform(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    float sample = in[tid];
    float centered = sample - 0.125f;
    float energy = centered * centered + 0.03125f;
    float scaled = energy * 1.125f + 0.5f;
    float normalized = centered / (scaled + 0.25f);
    float corrected = normalized * (1.0f + scaled * 0.0625f);
    float damped = corrected / (1.0f + corrected * corrected * 0.5f);
    float fused = damped * 1.03125f + scaled * 0.015625f;
    float refined = fused / (1.0f + fused * 0.25f);
    out[tid] = refined;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h = new float[n];
    float *in, *out;
    cudaMalloc(&in, bytes);
    cudaMalloc(&out, bytes);
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i % 97) * 0.03125f;
    }
    cudaMemcpy(in, h, bytes, cudaMemcpyHostToDevice);
    feature_transform<<<(n + 255) / 256, 256>>>(in, out, n);
    cudaMemcpy(h, out, bytes, cudaMemcpyDeviceToHost);
    std::printf("%f\n", h[1]);
    cudaFree(in);
    cudaFree(out);
    delete[] h;
    return 0;
}
