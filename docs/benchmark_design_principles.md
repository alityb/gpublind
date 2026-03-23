# GPUBlind Benchmark Design Principles

## Why this document exists

GPUBlind is intended to become a credible benchmark for testing whether LLMs can reason about GPU kernel performance bottlenecks. For a benchmark like this to be publishable, it must do more than report a single accuracy number on a small, static set of examples. The benchmark needs clear ground truth, adequate statistical power, reproducible procedures, transparent artifacts, and evidence that success on the benchmark tracks the real capability of interest.

This document summarizes the design principles drawn from the requested sources and translates them into concrete requirements for GPUBlind.

## Key principles from prior work

### 1. Broad coverage, while being explicit about incompleteness

HELM argues that credible evaluation needs broad coverage across the space of relevant scenarios and should explicitly document what is not yet covered. A benchmark should not silently present a narrow slice of the problem as if it were complete.

Implication for GPUBlind:
- The benchmark should contain more than a handful of handwritten kernels.
- It should include diverse kernel sources, bottleneck types, and difficulty levels.
- It should document what is still missing, such as additional GPU architectures, additional bottleneck families, and multi-kernel workloads.

### 2. Multi-metric measurement, not single-metric scoring

HELM emphasizes that evaluation should use multiple metrics because accuracy alone hides important tradeoffs. KernelBench makes the same point in a different way by separating functional correctness from performance and combining them in a unified metric. A benchmark is stronger when it measures both whether a model gets the answer right and how robust, consistent, or efficient that reasoning is.

Implication for GPUBlind:
- Report accuracy, parse error rate, consistency across question formats, adversarial susceptibility, and confidence intervals.
- Preserve qualitative reasoning and prompt artifacts so errors can be inspected, not just counted.
- Keep profiling-derived metadata available for auditability.

### 2a. Separate correctness from groundedness

Exa's WebCode writeup is useful here because it distinguishes between an answer being correct and an answer being well grounded in the evidence actually available to the model. That distinction matters for GPUBlind: a model can guess the right bottleneck label for the wrong reasons, especially on adversarial kernels where a misleading signal is intentionally present.

Implication for GPUBlind:
- Report label accuracy separately from evidence-grounded reasoning.
- Mark whether a correct answer cited the expected profiler evidence.
- Penalize answers that anchor on the red herring even when the final label is right.

### 3. Standardization and transparent comparison

HELM’s core claim is that model comparisons become unreliable when prompts, formatting, and evaluation rules vary across models. KernelBench similarly standardizes task format, evaluation pipeline, and scoring so models are compared under the same protocol.

Implication for GPUBlind:
- Use standardized prompt templates and output parsing.
- Keep one canonical result schema across models and humans.
- Version the evaluation protocol in documentation and code.
- Avoid result files that append heterogeneous retries into ambiguous lists.

### 4. Automatic verifiability and auditable artifacts

KernelBench is especially relevant because it focuses on GPU kernels and emphasizes automatic verification, repeated trials, and programmatic measurement of success. For GPUBlind, the analogue is profiler-backed labeling and full retention of the evidence used to derive those labels.

Implication for GPUBlind:
- Every kernel label should be backed by stored profiler evidence.
- Label assignment should be programmatic and auditable, not purely hand-authored.
- The benchmark should expose both the final label and the intermediate verification signals that support it.

### 5. Dynamic and adversarial evaluation matters

Dynabench argues that static benchmarks saturate, miss simple failure cases, and often fail to reflect real-world robustness. Its human-and-model-in-the-loop design is a response to those issues. For GPUBlind, this does not require a full Dynabench-style platform, but it does require active pressure against benchmark gaming and prompt overfitting.

Implication for GPUBlind:
- Maintain multiple prompt formats, not one fixed question.
- Include adversarial framings and misleading cues.
- Track consistency across formats and levels of evidence.
- Allow iterative expansion of the corpus as failure modes become known.

### 6. Statistical validity and adequate sample size

Work on validity challenges in ML benchmarks shows that narrow benchmarks are fragile and can yield misleading conclusions, especially when small distribution changes produce large performance swings. Even when adaptive overfitting is not the primary problem, external validity and underpowered reporting remain major threats.

Implication for GPUBlind:
- Five kernels are not enough for publishable accuracy claims.
- Accuracy tables must include confidence intervals.
- The benchmark needs a substantially larger verified corpus and should flag underpowered analyses.

### 7. Reproducibility requires environment, code, and workflow discipline

Reproducibility guidance in ML consistently emphasizes publishing exact code, data artifacts, environment details, and workflow steps. A benchmark is much more credible when a third party can rerun it without reverse-engineering hidden assumptions.

Implication for GPUBlind:
- Document Python, CUDA, Nsight Compute, GPU, and model settings.
- Keep scripts for mining, profiling, cleaning, validating, and analyzing results in-repo.
- Make reruns idempotent and trial-aware.
- Treat transient API failures as operational noise, not benchmark data.

### 7a. Benchmarks need a structured benchmark card

Recent benchmark documentation work, including BenchmarkCards and evaluation factsheet proposals, argues that benchmarks should ship with a structured description of scope, intended use, exclusions, failure modes, and governance. This is not just paperwork; it reduces ambiguity about what the benchmark actually measures.

Implication for GPUBlind:
- Maintain a benchmark card that states the benchmark goal, unit of evaluation, scoring dimensions, exclusions, contamination risks, and release/update policy.
- Make the card specific enough that a third party could tell whether a claimed use of GPUBlind is in-scope or out-of-scope.

### 8. Validity depends on alignment between the measured task and the claimed capability

A benchmark is only meaningful if the ground truth labels actually correspond to the capability being measured. For GPUBlind, that means the task is not “repeat the metadata label,” but “infer the true bottleneck from profiler evidence and code.” This requires careful label methodology and explicit handling of ambiguous cases.

Implication for GPUBlind:
- Introduce multi-signal verification for each bottleneck label.
- Mark ambiguous kernels explicitly rather than forcing a debatable single label.
- Filter analyses by verification confidence.

### 9. Human baselines improve interpretability

KernelBench and broader benchmark practice both benefit from anchor baselines. Random and frequency baselines are useful sanity checks, but they do not answer whether the task is meaningfully hard for qualified humans.

Implication for GPUBlind:
- Add a human evaluation path with the same schema as model outputs.
- Record time-to-answer so performance can be interpreted against expert effort, not just against trivial baselines.

## What these principles mean for GPUBlind

To be publishable, GPUBlind should satisfy the following:

1. Corpus breadth
- At least dozens of verified kernels, not only five handwritten cases.
- Coverage across bottleneck families, sources, and difficulty levels.

2. Ground-truth transparency
- Programmatic label verification stored in each profile.
- Confidence levels and an explicit ambiguous category.

3. Statistical reporting
- Confidence intervals on all headline accuracy numbers.
- Warnings when sample size is too small for stable claims.

4. Reproducible execution
- One result per file, trial-aware output layout, and deterministic resume logic.
- Separate operational errors from benchmark measurements.

5. Multi-view evaluation
- Multiple prompt formats, adversarial framing, metrics-only settings, and consistency scoring.
- Distinct reporting of label correctness versus evidence-groundedness.

6. Auditability
- Persist prompts, raw responses, parsed outputs, profiler features, and validation reports.
- Publish a benchmark card that captures scope, limitations, and update policy.

7. Human interpretability
- Human baseline interface and documentation that explains exactly how labels and scores are computed.

## Current repo gaps identified from these principles

At the start of this pass, the repo already had a useful skeleton:
- handwritten kernels with stored profiles
- profiler parsing and derived features
- multi-level prompt evaluation
- multiple question formats
- analysis scripts and tests

The main gaps were:
- too few verified kernels for meaningful statistical claims
- no explicit multi-signal label verification and confidence field
- result-file structure that still mixed legacy and new layouts
- no confidence intervals in analysis outputs
- no human baseline interface
- incomplete benchmark documentation
- no corpus or result validation script

## Known limitations that will remain even after this cleanup

Even after the planned fixes, GPUBlind will still have important limitations:
- labels remain hardware-specific and should not be assumed to transfer across GPUs
- profiler-derived labels are only as strong as the chosen metrics and heuristics
- the benchmark still measures reasoning about bottlenecks, not end-to-end kernel optimization ability
- prompt-based evaluations can drift as API models change over time

These limitations should be documented explicitly in the paper and blog post rather than hidden.

## Sources consulted

- HELM: Holistic Evaluation of Language Models
- Dynabench: Rethinking Benchmarking in NLP
- KernelBench: Can LLMs Write Efficient GPU Kernels?
- Exa WebCode benchmark writeup
- BenchmarkCards / evaluation-factsheet style benchmark documentation
- Validity Challenges in Machine Learning Benchmarks
- Reproducibility standards for machine learning in the life sciences
