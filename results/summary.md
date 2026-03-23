1. claude-opus-4-6 achieved the highest aggregate accuracy at 33.33%.
2. claude-opus-4-6 improved the most from level 1 to level 3, gaining 66.67 points when profiler evidence was added.
3. claude-opus-4-6 was most vulnerable to adversarial framing at level 4, agreeing with the wrong framing 33.33% of the time.

Statistical Notes:
- All CIs computed using Wilson score interval.
- Results with CI width > 30pp are flagged as underpowered.
- Minimum recommended corpus size for 20pp CI: 96 kernels.
- Groundedness reports correct answers whose reasoning cites expected profiler evidence and avoids the primary red herring.

Consistency Scores

| Model | Mean Consistency | Std | Min | Max | 95% CI |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 0.4487 | 0.4282 | 0.0 | 1.0 | [0.288, 0.622] |
| gpt-5.4 | 0.7 | 0.2535 | 0.5 | 1.0 | [0.567, 0.833] |
