# Reproducibility

## Environment

Verified local environment:

- Python: 3.12.3
- CUDA toolkit (`nvcc`): 12.0.140
- Nsight Compute CLI: 2025.4.1.0
- GPU roof file: `profiles/hardware_roof.json`
- Measured GPU in roof file: NVIDIA A10G

## Core artifacts

- handwritten kernels: `kernels/`
- mined corpus: `data/mined_kernels.jsonl`
- kernelbot corpus: `data/kernelbot_kernels.jsonl`
- handwritten profiles: `profiles/hw_*.json`
- roof file: `profiles/hardware_roof.json`
- evaluation prompts: `eval/prompts.py`
- evaluation runner: `eval/run_eval.py`
- analysis pipeline: `eval/analyze_results.py`

## Reproduce from scratch

### 1. Install the project

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 2. Verify hardware roof

If needed:

```bash
python3 -m profiles.measure_roof
cat profiles/hardware_roof.json
```

Expected:

- `gpu_name` matches the target GPU
- `peak_bw_tbps` and `peak_flops_tflops` are non-zero
- `measured` is `true`

### 3. Generate or refresh handwritten profiles

```bash
sudo -E env "PATH=$PATH" python3 -m profiles.generate_profiles
```

Expected:

- `profiles/hw_A.json` through `profiles/hw_E.json` exist
- each contains derived profile fields and a `verification` object

### 4. Inspect corpus health

```bash
python3 scripts/corpus_stats.py
```

Expected:

- a total kernel count
- verified count
- breakdowns by bottleneck, source, difficulty, and confidence

### 5. Run evaluation

Mock example:

```bash
python3 -m eval.run_eval \
  --model claude-opus-4-6 \
  --levels 1 \
  --filter source=handwritten \
  --min-confidence high \
  --question-formats label,yesno_memory,rank,junior_wrong \
  --trial 1 \
  --mock
```

Real example:

```bash
python3 -m eval.run_eval \
  --model claude-opus-4-6 \
  --levels 1,2,3,4,5 \
  --question-formats label \
  --min-confidence medium \
  --trial 1
```

Expected:

- one JSON result per `model × trial × level × format × kernel`
- outputs under `results/{model}/trial_{N}/...`

### 6. Clean operational failures if needed

```bash
python3 scripts/clean_results.py
```

### 7. Analyze results

```bash
python3 -m eval.analyze_results --include-formats --min-confidence medium
```

Expected:

- `results/accuracy_table.md`
- `results/summary.md`
- `results/consistency_scores.md`
- confidence intervals in reported accuracy tables

### 8. Validate result integrity

```bash
python3 scripts/validate_results.py --min-confidence medium
```

Expected:

- `PASS`, or a fixable list of specific integrity failures

### 9. Run tests

```bash
pytest -q
```

## Known sources of variance

- Nsight Compute measurements can vary slightly across runs.
- API model behavior can drift over time as hosted models are updated.
- Human evaluation timing depends on evaluator experience and local workflow.
- LLM outputs can vary if non-greedy decoding settings change.

## Sanity checks

- `api_error` rows should not remain in final benchmark results.
- result files should not contain lists
- profile JSONs should expose `verification.confidence`
- underpowered results should be flagged in analysis output
