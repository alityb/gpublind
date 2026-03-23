# GPUBlind Benchmark Card

## Purpose

GPUBlind measures whether a model can identify the primary performance bottleneck of a GPU kernel from code and profiler evidence. The target capability is diagnostic performance reasoning, not code generation and not full kernel optimization.

## Unit of evaluation

- One kernel profile and its associated prompt context.
- The primary scored task is predicting the bottleneck label.
- Secondary scored tasks include consistency across question formats, adversarial robustness, parse reliability, and evidence-grounded reasoning.

## Supported labels

- `memory-bound`
- `compute-bound`
- `latency-bound`
- `occupancy-limited`
- `register-spill`

## Inputs shown to the model

Depending on level and question format, the model may see:

- kernel source code
- derived Nsight Compute metrics
- adversarial framing or colleague claims
- metrics-only summaries without code

## Ground truth

- Handwritten kernels are labeled from real A10G Nsight Compute runs.
- Each stored profile includes triple-verification fields: roofline, bandwidth, stall, consensus, and confidence.
- Ambiguous kernels are allowed in the registry but can be filtered out of evaluation with `--min-confidence`.

## Primary metrics

- Label accuracy with Wilson 95% confidence intervals
- Parse error rate
- Consistency across question formats
- Sycophancy / adversarial-framing susceptibility
- Evidence-grounded correctness

## Evidence-grounded correctness

GPUBlind distinguishes between:

- `correct`: the predicted label matches the stored ground truth
- `grounded`: the reasoning cites expected profiler evidence and avoids anchoring on the kernel's primary red herring

This prevents the benchmark from rewarding lucky guesses too aggressively.

## Intended uses

- Comparing frontier models on GPU performance diagnosis
- Stress-testing whether extra profiler evidence improves reasoning
- Measuring robustness to misleading bottleneck cues
- Producing case studies for blog posts and papers on systems reasoning

## Out-of-scope uses

- Claiming end-to-end GPU optimization ability
- Claiming transfer across GPU architectures without rerunning profiling
- Treating the benchmark as a substitute for expert kernel review
- Measuring tool-use or iterative debugging ability beyond the prompt protocol

## Known risks and limitations

- Labels are hardware-specific and currently centered on A10G.
- Heuristic label verification can still disagree with expert judgment on borderline kernels.
- Historical result trees may contain partial optional-format coverage from older runs.
- API model drift can change results over time even with fixed prompts.

## Reproducibility artifacts

GPUBlind should be released with:

- kernel source
- profile JSON files
- mining / profiling / validation scripts
- prompt templates
- parsed model outputs
- benchmark docs and readiness checks

## Governance and updates

- Add new kernels rather than silently rewriting old benchmark slices.
- Keep result files single-entry and trial-aware.
- Run validation and readiness checks before publishing new tables.
- Treat new adversarial kernels as benchmark expansions, not quiet test-set replacements.
