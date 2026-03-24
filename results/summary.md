1. gpt-5.4 achieved the highest aggregate accuracy at 66.48%.
2. gpt-5.4 improved the most from level 1 to level 3, gaining 8.43 points when profiler evidence was added.
3. claude-opus-4-6 was most vulnerable to adversarial framing at level 4, agreeing with the wrong framing 33.33% of the time.

Statistical Notes:
- All CIs computed using Wilson score interval.
- Results with CI width > 30pp are flagged as underpowered.
- Minimum recommended corpus size for 20pp CI: 96 kernels.
- Groundedness reports correct answers whose reasoning cites expected profiler evidence and avoids the primary red herring.
- Judge-based reasoning quality scores are produced by an auxiliary LLM judge when available.
