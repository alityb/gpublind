# GPUBlind v2 Benchmark Card

- Task: Diagnose primary CUDA kernel bottleneck.
- Target labels: memory-bound, compute-bound, latency-bound, occupancy-limited, register-spill.
- Primary metric: Diagnostic funnel (DRR, RVR, MPR).
- Hardware: NVIDIA A10G.
- Risks: class imbalance, label leakage, overreliance on judge coverage, small-sample uncertainty.
