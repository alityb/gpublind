# GPUBlind v2 Report

## 1. The Diagnostic Funnel

| Model | Condition | DRR | RVR | MPR |
| --- | --- | --- | --- | --- |
| claude-opus-4-6 | C0 | 91.7% [74.2%, 97.7%] | 91.7% [74.2%, 97.7%] | 62.5% [42.7%, 78.8%] |
| claude-opus-4-6 | C1 | 75.0% [55.1%, 88.0%] | 75.0% [55.1%, 88.0%] | 75.0% [55.1%, 88.0%] |
| claude-opus-4-6 | C2 | 79.2% [59.5%, 90.8%] | 79.2% [59.5%, 90.8%] | 79.2% [59.5%, 90.8%] |
| gpt-5.4 | C0 | 87.5% [69.0%, 95.7%] | 87.5% [69.0%, 95.7%] | 87.5% [69.0%, 95.7%] |
| gpt-5.4 | C1 | 70.8% [50.8%, 85.1%] | 70.8% [50.8%, 85.1%] | 70.8% [50.8%, 85.1%] |
| gpt-5.4 | C2 | 79.2% [59.5%, 90.8%] | 79.2% [59.5%, 90.8%] | 79.2% [59.5%, 90.8%] |
| Random | C2 | 16.7% | — | — |
| Frequency (always memory-bound) | C2 | 41.7% | — | — |
| Roofline (AI vs ridge) | C2 | 43.8% | — | — |
| Rule-Based Expert | C2 | 52.1% | — | — |

## 2. Per-category breakdown (C2)

| Model | memory | compute | latency | occupancy | register |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 50.0% (N=10) | 100.0% (N=1) | 100.0% (N=9) | 100.0% (N=3) | 100.0% (N=1) |
| gpt-5.4 | 50.0% (N=10) | 100.0% (N=1) | 100.0% (N=9) | 100.0% (N=3) | 100.0% (N=1) |

## 3. Information sensitivity

| Metric | C0→C1 | C0→C2 | C1→C2 | C0→C3 |
| --- | --- | --- | --- | --- |
| claude-opus-4-6 | -16.7 pp | -12.5 pp | +4.2 pp | -91.7 pp |
| gpt-5.4 | -16.7 pp | -8.3 pp | +8.3 pp | -87.5 pp |

## 4. One-shot correction rate

| Model | Wrong at C2 | Corrected after hint | Correction rate |
| --- | --- | --- | --- |
| claude-opus-4-6 | 5 | 0 | 0.0% |
| gpt-5.4 | 5 | 0 | 0.0% |

## 5. Confidence calibration

| Model | When HIGH confidence | When MEDIUM | When LOW |
| --- | --- | --- | --- |
| claude-opus-4-6 | 81.9% | — | — |
| gpt-5.4 | 78.9% | 100.0% | — |

## 6. The hw_B case study

#### C0
BOTTLENECK: latency-bound
CONFIDENCE: HIGH
REASONING: The kernel processes only n=1<<20 (1M) elements, launching 4096 blocks of 256 threads. Each thread reads one float and writes one float (8 bytes) with ~15 FLOPs, giving an arithmetic intensity of ~1.9 FLOP/byte — far below the ridge point of 62 FLOP/byte, so it is not compute-bound. However, the total data footprint is only 8 MB, which is too small to saturate the A10G's DRAM bandwidth (>500 GB/s), so DRAM utilization will be low, ruling out memory-bound. The long dependent chain of arithmetic operations (each line depends on the previous result) creates instruction-level serialization within each thread, causing scoreboard/dependency stalls while DRAM utilization remains low — the hallmark of a latency-bound kernel with insufficient arithmetic or memory-level parallelism to hide the pipeline latency.

#### C1
BOTTLENECK: latency-bound
CONFIDENCE: HIGH
REASONING: The kernel exhibits 57.1% of active cycles stalled on long scoreboard waits, which typically indicates warps waiting for memory or dependent instruction results to return. Despite this, DRAM bandwidth utilization is essentially 0.0% and the L2 hit rate is 100.0%, meaning all data is served from the L2 cache rather than DRAM — so this is not a classical memory-bound scenario. The arithmetic intensity of 1.25 FLOP/byte is far below the ridge point of 62, confirming the kernel is not compute-bound, and the high long scoreboard stalls with low DRAM utilization clearly point to latency-bound behavior where warps are stalled waiting on L2 cache responses or dependent operations that occupancy (74.2%) cannot fully hide.

#### C2
BOTTLENECK: latency-bound
CONFIDENCE: HIGH
REASONING: The kernel shows 57.1% of active cycles stalled on Long Scoreboard, which indicates warps are waiting for the results of dependent operations (likely the long chain of dependent floating-point arithmetic: divisions and multiply-adds forming a serial dependency chain). DRAM bandwidth utilization is 0.0%, ruling out memory-bound behavior — the 100% L2 hit rate means all data is served from cache, yet the kernel still stalls because each instruction depends on the previous result. The arithmetic intensity of 1.25 FLOP/byte is far below the ridge point of 62, but the near-zero DRAM utilization confirms this is not a classical memory-bandwidth bottleneck; rather, it is instruction-level latency in the serial computation chain that the 74.2% occupancy cannot fully hide.

