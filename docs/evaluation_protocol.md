# Evaluation Protocol

## Task

GPUBlind measures whether a model can identify the primary performance bottleneck of a GPU kernel using source code, profiler evidence, or adversarial framing.

The valid labels are:

- memory-bound
- compute-bound
- latency-bound
- occupancy-limited
- register-spill

## Information levels

The benchmark currently uses five information levels:

1. code only
2. code plus basic metrics
3. code plus full profiler output
4. adversarial wrong framing
5. adversarial correct framing

These are defined in `eval/prompts.py`.

## Question formats

GPUBlind supports seven question formats:

### 1. `label`

Standard label prediction.

Required output:

```text
BOTTLENECK: <label>
REASONING: <2-3 sentences>
```

Scoring:

- exact-match label accuracy
- severity-weighted confusion tracking
- evidence-grounded correctness tracking when the reasoning cites expected profiler evidence

### 2. `yesno_memory`

Binary test for whether the kernel is memory-bound.

Required output:

```text
MEMORY_BOUND: <YES or NO>
REASONING: <explanation>
```

Scoring:

- YES maps to `memory-bound`
- NO maps to `not-memory-bound`

### 3. `rank`

Rank all bottleneck types from most to least likely.

Required output:

```text
RANK_1: <label>
RANK_2: <label>
RANK_3: <label>
RANK_4: <label>
RANK_5: <label>
REASONING: <2-3 sentences>
```

Scoring:

- top-1 rank is used as the predicted label

### 4. `junior_wrong`

Adversarial agreement test with an intentionally wrong memory-bandwidth claim.

Required output:

```text
ASSESSMENT: <AGREE or DISAGREE>
REASONING: <2-3 sentences>
```

Scoring:

- `AGREE` is interpreted as endorsing `memory-bound`
- `DISAGREE` is interpreted as rejecting that framing

### 5. `junior_right`

Agreement test with a plain-English statement of the correct bottleneck.

Required output:

```text
ASSESSMENT: <AGREE or DISAGREE>
REASONING: <2-3 sentences>
```

Scoring:

- agreement is expected
- this format is used more for calibration than for label extraction

### 6. `fix`

Optimization recommendation.

Required output:

```text
CHANGE: <one sentence>
EXPECTED_IMPROVEMENT: <metric change and why>
```

Scoring:

- no label accuracy score
- stored as qualitative analysis only

### 7. `metrics_only`

Bottleneck prediction without source code, using only compact metrics.

Required output:

```text
BOTTLENECK: <label>
REASONING: <2-3 sentences>
```

Scoring:

- exact-match label accuracy

## Parse errors

A response counts as a parse error when the required structured field cannot be extracted or the extracted label is not one of the valid labels.

Examples:

- missing `BOTTLENECK:`
- invalid label spelling
- missing `MEMORY_BOUND:`
- malformed `RANK_1` ... `RANK_5`

## Consistency score

Consistency is computed per `kernel × model × level`:

- collect the inferred labels from `label`, `yesno_memory`, `rank`, and `junior_wrong`
- ignore `fix` and `junior_right`
- majority label = most frequent inferred label
- consistency = fraction of formats agreeing with that majority label

The benchmark reports aggregate consistency per model and uses bootstrap confidence intervals for the mean.

## Evidence-grounded correctness

Following the benchmark-design principle that correctness and groundedness should be separated, GPUBlind also tracks whether a correct answer is justified by the right evidence.

For handwritten kernels, this uses `reasoning_rubric` metadata from `meta.json`:

- the reasoning must mention at least one required profiler signal
- the first sentence must not anchor on the kernel's primary red herring
- the final label must still be correct

This metric is reported separately from plain label accuracy in `results/groundedness_table.md`.

## Confidence filtering

Benchmark runs can be filtered by profile verification confidence:

- `high`: only 3-of-3 consensus kernels
- `medium`: 2-of-3 or 3-of-3 consensus kernels
- `any`: include ambiguous kernels too

## Result schema

Each result file stores one evaluation only and includes:

- kernel id
- model
- trial
- level
- question format
- true bottleneck
- predicted label
- correctness
- severity
- raw response
- parsed fields
- rendered prompt
- timestamp

## Human evaluation

`eval/human_eval.py` writes human answers using the same schema so human and model results can be analyzed together.
