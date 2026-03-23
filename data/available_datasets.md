# Available CUDA / Triton Datasets for GPUBlind

This note records the external datasets and repos inspected for expanding GPUBlind's category coverage.

## 1. GPU MODE reference-kernels

- URL: `https://github.com/gpu-mode/reference-kernels`
- License: MIT
- Size: ~30 reference problems across PMPP, AMD, NVIDIA, distributed, and BioML tracks
- Programmatic access:
  - `git clone --depth 1 https://github.com/gpu-mode/reference-kernels`
  - inspect `problems/*/*`

### Problem inventory

Compute-heavy candidates:

- `amd/fp8-mm`
- `amd/mla-decode`
- `amd/moe`
- `amd_202602/mixed-mla`
- `amd_202602/moe-mxfp4`
- `amd_202602/mxfp4-mm`
- `amd_distributed/ag-gemm`
- `amd_distributed/gemm-rs`
- `bioml/trimul`
- `nvidia/modal_nvfp4_dual_gemm`
- `nvidia/nvfp4_dual_gemm`
- `nvidia/nvfp4_gemm`
- `nvidia/nvfp4_group_gemm`
- `pmpp/matmul_py`
- `pmpp_v2/matmul_py`

Memory-heavy candidates:

- `amd/identity`
- `pmpp/grayscale_py`
- `pmpp/histogram_py`
- `pmpp/sort_py`
- `pmpp/vectoradd_py`
- `pmpp/vectorsum_py`
- `pmpp_v2/grayscale_py`
- `pmpp_v2/histogram_py`
- `pmpp_v2/sort_py`
- `pmpp_v2/vectoradd_py`
- `pmpp_v2/vectorsum_py`

Mixed or ambiguous:

- `amd_distributed/all2all`
- `helion/causal_conv1d_py`
- `helion/fp8_quant_py`
- `helion/gated_deltanet_chunk_fwd_h_py`
- `helion/gated_deltanet_chunk_fwd_o_py`
- `helion/gated_deltanet_recompute_w_u_py`
- `nvidia/nvfp4_gemv`
- `pmpp/conv2d_py`
- `pmpp/prefixsum_py`
- `pmpp_v2/conv2d_py`
- `pmpp_v2/prefixsum_py`

Expected bottleneck distribution:

- strong source for compute-bound GEMM / MoE / MLA workloads
- good source for memory-bound basics from PMPP
- weaker for true latency-bound and register-spill cases

## 2. GPUMODE / kernelbot-data

- URL: `https://huggingface.co/datasets/GPUMODE/kernelbot-data`
- License: CC-BY-4.0
- Size:
  - Hugging Face tags mark it as `100K < n < 1M`
  - configs include `amd_submissions`, `amd_successful_submissions`, `nvidia_nvfp4_submissions`, `leaderboards`
  - `amd_successful_submissions` currently has 60,357 rows
  - `nvidia_nvfp4_submissions` is substantially larger
- Programmatic access:
  - `load_dataset('GPUMODE/kernelbot-data', 'amd_successful_submissions')`
  - `load_dataset('GPUMODE/kernelbot-data', 'nvidia_nvfp4_submissions')`

Expected bottleneck distribution:

- broad but noisy
- good source for occupancy, register-pressure, and strided-access patterns
- weaker as a direct verified corpus because it needs static pattern mining plus later profiling

Notes:

- The current `data/mine_kernelbot.py` integration already mines this dataset by suspicious pattern.
- Hardware metadata is present via leaderboard / run metadata rather than a single clean `hardware` field.

## 3. ScalingIntelligence / KernelBench

- URL: `https://huggingface.co/datasets/ScalingIntelligence/KernelBench`
- Upstream repo: `https://github.com/ScalingIntelligence/KernelBench`
- License:
  - Hugging Face card does not currently expose a license field
  - upstream GitHub repo is MIT
- Size:
  - `level_1`: 100 tasks
  - `level_2`: 100 tasks
  - `level_3`: 50 tasks
  - `level_4`: 20 tasks
- Programmatic access:
  - `load_dataset('ScalingIntelligence/KernelBench')`
  - splits are exposed directly as `level_1`, `level_2`, `level_3`, `level_4`

Expected bottleneck distribution:

- `level_1`: mostly memory-bound / single-op workloads
- `level_2`: mixed fused kernels, including matmul- and conv-heavy candidates
- `level_3`: best source for compute-bound candidates because it contains full architectures such as ResNet, VGG, MobileNet, MiniGPT, and Mamba-style tasks

How GPUBlind uses it:

- `data/mine_kernelbench.py` mines likely compute-bound candidates from levels 2 and 3
- outputs standalone CUDA harness templates that still require real GPU profiling before they can be treated as verified

## 4. triton-lang / kernels

- URL: `https://github.com/triton-lang/kernels`
- License: MIT
- Size: focused repository of reference Triton kernels plus tests and profiling utilities
- Programmatic access:
  - `git clone --depth 1 https://github.com/triton-lang/kernels`
  - inspect `kernels/*.py`

Relevant kernels:

- `kernels/matmul.py`
- `kernels/flash_attention.py`
- `kernels/blocksparse/matmul.py`
- `kernels/blocksparse/softmax.py`
- `kernels/cross_entropy.py`

Expected bottleneck distribution:

- strong compute-bound and mixed attention workloads
- useful as a future source of real Triton kernels with benchmark harnesses
- less helpful for latency-bound or register-spill examples

## Recommended integration order

1. KernelBench level 2 / 3 for compute-bound candidate mining
2. KernelBot for occupancy / register-pressure / memory-pathology candidates
3. reference-kernels for curated compute-heavy and memory-heavy handcrafted tasks
4. triton-lang/kernels for future Triton-native benchmark expansion
