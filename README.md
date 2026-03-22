### What is GPUBlind?
GPUBlind is a benchmark for testing whether language models can diagnose GPU kernel bottlenecks from code and profiling evidence, especially when the code contains misleading cues that reward shallow pattern matching. It combines NCU-verified mined kernels, handwritten adversarial kernels, and pattern-detected candidates awaiting profiling so the benchmark can distinguish correct labels from correct reasoning. GPUBlind tests reasoning, not generation.

### Corpus
| Source | Count | Ground Truth | Categories |
|--------|-------|--------------|------------|
| Handwritten (adversarial) | 5 | NCU verified | A-E |
| Sakana AI CUDA Engineer | ~70 | NCU verified | mined |
| KernelBot (GPU Mode) | ~30 | Needs profiling | pattern-detected |
| **Total** | **~105** | | |

Note: KernelBot entries marked `needs_profiling=True` are excluded from default evaluation runs until profiled on hardware.

### Kernel Categories
| Category | Description | Misleading Signal | True Bottleneck |
|---|---|---|---|
| compute_looks_memory | High arithmetic density paired with wasteful global reads hidden behind tile-style traversal. | FLOP-heavy inner loop and traversal-style indexing look cache-friendly. | memory-bound |
| memory_looks_compute | Coalesced streaming loads paired with long dependent compute chains. | Access pattern looks bandwidth-optimal. | latency-bound |
| red_herring | An obvious synchronization distraction hides the real memory problem. | Branched `__syncthreads()` looks like the issue. | memory-bound |
| occupancy_trap | Shared-memory tiling looks textbook-correct but collapses resident blocks per SM. | Large tiling comment and layout look carefully optimized. | occupancy-limited |
| register_spill | Straightforward-looking streaming traffic is dominated by spill-induced local memory. | Global traffic looks like a normal bandwidth bottleneck. | register-spill |
| shared_memory_trap | Mined kernels where shared memory appears helpful but constrains execution. | Manual optimization cue dominates first impression. | compute-bound or memory-bound |
| register_pressure_decoy | Mined kernels with high register count that is not actually dominant. | Register count looks like the story. | varies by NCU |
| memory_looks_latency | Mined kernels that look throughput-limited but stall on latency. | Poor efficiency resembles bandwidth saturation. | latency-bound |

### Evaluation Levels
| Level | Name | Information Given | What It Tests |
|---|---|---|---|
| 1 | code_only | CUDA source only | Pure code reasoning under misleading cues |
| 2 | code_plus_basic_metrics | Code plus latency, occupancy, load efficiency, DRAM utilization | Whether light profiler context improves diagnosis |
| 3 | code_plus_full_ncu | Code plus full NCU JSON | Whether richer evidence improves beyond pattern matching |
| 4 | adversarial_wrong_framing | Code plus an explicitly wrong colleague claim | Susceptibility to wrong framing and sycophancy |
| 5 | adversarial_correct_framing | Code plus an explicitly correct colleague claim | Robustness when framing aligns with the truth |

### Results
| Model | L1 | L2 | L3 | L4 Sycophancy | Reasoning Quality |
|-------|----|----|----|----|---|
| (run eval to populate) | | | | | |

### Reproducing Results
1. `pip install -e .`
2. `python -m data.mine_sakana`
3. `python -m data.mine_kernelbot`
4. `python -m registry.build_registry --mined data/mined_kernels.jsonl --kernelbot data/kernelbot_kernels.jsonl --kernels kernels/ --profiles profiles/ --output registry/registry.json --mock`
5. `python -m eval.run_eval --model gpt-4o --mock`
6. `python -m profiles.measure_roof`
7. `python -m profiles.generate_profiles`
8. `python -m eval.run_eval --model gpt-4o`
9. `python -m eval.analyze_results`

### Adding a Kernel
Exact `meta.json` schema:
```json
{
  "id": "hw_X",
  "category": "category_name",
  "true_bottleneck": "memory-bound",
  "misleading_signal": "human-readable trap description",
  "difficulty": "hard",
  "hardware": "A100",
  "reasoning_rubric": {
    "must_mention": ["term one", "term two"],
    "must_not_cite_as_primary": ["term three", "term four"]
  }
}
```
Note: `true_bottleneck` must come from `ncu`, never from human judgment.

### NCU Command
```bash
ncu --csv --page raw --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,l2__global_load_requests.sum,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed,sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_global_ld_sectors_miss.sum -o <output_prefix> <binary>
```

### Dataset
Dataset available at: https://huggingface.co/datasets/alityb/gpublind
