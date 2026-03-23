# Ground Truth Methodology

## Goal

GPUBlind labels each kernel with a primary bottleneck label intended to reflect the most defensible explanation of the observed performance on the measured hardware. Labels are not meant to be purely intuitive judgments from source code. They are derived from profiler features and then checked with a three-signal verification procedure.

## Hardware context

The handwritten benchmark kernels in this repo are profiled against an empirically measured roof file:

- GPU: NVIDIA A10G
- Peak bandwidth: 0.495687 TB/s
- Peak FLOPS: 30.769767 TFLOPS
- Ridge point: `peak_flops_tflops / peak_bw_tbps`

The roof is stored in `profiles/hardware_roof.json` and should be treated as the source of truth for hardware-specific thresholds in the benchmark.

## Derived profile features

`profiles/generate_profiles.py` derives the following from Nsight Compute output:

- arithmetic intensity
- memory-bound vs compute-bound roofline side
- dominant stall type
- global load efficiency
- achieved occupancy
- long-scoreboard stall fraction
- memory-stall fraction
- register count
- L2 hit rate
- DRAM bandwidth utilization

These features are written into each profile JSON and retained for downstream auditing.

## Triple-verification procedure

Every profile should carry a `verification` field with:

```json
{
  "roofline": "memory-bound",
  "bandwidth": "latency-bound",
  "stall": "latency-bound",
  "consensus": "latency-bound",
  "confidence": "medium"
}
```

The three verification signals are:

### 1. Roofline test

This is the simplest check.

- If `arithmetic_intensity < ridge_point`, vote `memory-bound`
- Else, vote `compute-bound`

This test is useful but insufficient on its own because it cannot separate latency-bound, occupancy-limited, or register-spill cases cleanly.

### 2. Bandwidth-and-latency test

This checks whether the measured execution actually looks bandwidth-limited or latency-limited:

- If `dram_bw_utilization > 0.50`, vote `memory-bound`
- Else if `stall_long_sb_pct > 0.30` and `dram_bw_utilization < 0.10`, vote `latency-bound`
- Else if `achieved_occupancy < 0.40`, vote `occupancy-limited`
- Else, vote `compute-bound`

This test is designed to avoid the common mistake of calling every long-scoreboard kernel “memory-bound.”

### 3. Stall-classification test

This maps the derived `dominant_stall_type` to a benchmark label:

- memory-related dominant stalls -> `memory-bound`
- arithmetic-dependency / long-scoreboard dominant stalls -> `latency-bound`
- occupancy-collapse dominant stalls -> `occupancy-limited`
- register-spill dominant stalls -> `register-spill`
- otherwise, conservative fallback behavior is used

## Consensus and confidence

The benchmark uses the following confidence policy:

- `high`: all 3 verification signals agree
- `medium`: 2 of 3 agree
- `low`: no 2-of-3 consensus
- `ambiguous`: the consensus label is explicitly set to `ambiguous`

For evaluation:

- `--min-confidence high` keeps only fully agreed labels
- `--min-confidence medium` keeps high and medium confidence labels
- `--min-confidence any` includes ambiguous cases

## What counts as ambiguous

A kernel is treated as ambiguous when the three verification signals do not reach a 2-of-3 agreement. This is intentional. The benchmark should not force a single “ground truth” label when the profiler evidence does not support one cleanly.

Ambiguous kernels can still be useful for exploratory analysis, but they should be excluded from headline benchmark claims unless explicitly justified.

## Verified vs unverified kernels

For corpus statistics and benchmark filtering, a kernel is considered verified only if:

- Nsight Compute profiling succeeded
- arithmetic intensity is non-zero
- `dominant_stall_type` is not `unknown`
- no sentinel `-1` profile values remain
- the entry is marked as `ground_truth_verified`

This separates profiler-backed cases from pattern-detected but unprofiled candidates.

## Known limitations

- The verification procedure is heuristic, not a formal proof of causality.
- Labels are hardware-specific and may change across GPUs.
- Some kernels legitimately exhibit mixed behavior; forcing a single primary bottleneck remains a simplification.
- Mined kernels can inherit upstream profiler artifacts that are less controlled than the handwritten A10G cases.
