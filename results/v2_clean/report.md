# GPUBlind v2 Report

## 1. The Diagnostic Funnel

| Model | Condition | DRR | RVR | MPR |
| --- | --- | --- | --- | --- |
| claude-opus-4-6 | C0 | 90.2% [79.0%, 95.7%] | 90.2% [79.0%, 95.7%] | 78.4% [65.4%, 87.5%] |
| claude-opus-4-6 | C1 | 76.5% [63.2%, 86.0%] | 76.5% [63.2%, 86.0%] | 76.5% [63.2%, 86.0%] |
| claude-opus-4-6 | C2 | 88.2% [76.6%, 94.5%] | 88.2% [76.6%, 94.5%] | 88.2% [76.6%, 94.5%] |
| gpt-5.4 | C0 | 82.4% [69.7%, 90.4%] | 82.4% [69.7%, 90.4%] | 82.4% [69.7%, 90.4%] |
| gpt-5.4 | C1 | 76.5% [63.2%, 86.0%] | 76.5% [63.2%, 86.0%] | 76.5% [63.2%, 86.0%] |
| gpt-5.4 | C2 | 88.2% [76.6%, 94.5%] | 88.2% [76.6%, 94.5%] | 88.2% [76.6%, 94.5%] |
| Random | C2 | 18.7% | — | — |
| Frequency (always memory-bound) | C2 | 40.0% | — | — |
| Roofline (AI vs ridge) | C2 | 41.3% | — | — |
| Rule-Based Expert | C2 | 64.0% | — | — |

## 2. Per-category breakdown (C2)

| Model | memory | compute | latency | occupancy | register |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 70.0% (N=20) | 100.0% (N=1) | 100.0% (N=18) | 100.0% (N=7) | 100.0% (N=5) |
| gpt-5.4 | 70.0% (N=20) | 100.0% (N=1) | 100.0% (N=18) | 100.0% (N=7) | 100.0% (N=5) |

## 3. Information sensitivity

| Metric | C0→C1 | C0→C2 | C1→C2 | C0→C3 |
| --- | --- | --- | --- | --- |
| claude-opus-4-6 | -13.7 pp | -2.0 pp | +11.8 pp | -90.2 pp |
| gpt-5.4 | -5.9 pp | +5.9 pp | +11.8 pp | -82.4 pp |

## 4. One-shot correction rate

| Model | Wrong at C2 | Corrected after hint | Correction rate |
| --- | --- | --- | --- |
| claude-opus-4-6 | 6 | 0 | 0.0% |
| gpt-5.4 | 6 | 0 | 0.0% |

## 5. Confidence calibration

| Model | When HIGH confidence | When MEDIUM | When LOW |
| --- | --- | --- | --- |
| claude-opus-4-6 | 85.0% | — | — |
| gpt-5.4 | 82.4% | — | — |

## 6. The hw_B case study

No hw_B case study results available.
