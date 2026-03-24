# GPUBlind v2 Methodology

GPUBlind v2 measures whether an LLM can diagnose why an existing CUDA kernel is slow. The benchmark uses a strict diagnostic funnel:

1. DRR: correct bottleneck label.
2. RVR: correct diagnosis backed by specific metric values or code evidence.
3. MPR: valid reasoning that does not primarily blame the kernel's red herring.

The corpus stores standalone CUDA kernels, label-safe profile metrics, and a reasoning rubric per kernel. Prompts never expose derived labels.
