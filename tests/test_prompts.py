from __future__ import annotations

from eval.prompts import PROMPTS


def test_prompt_rendering_all_levels() -> None:
    for level, prompt in PROMPTS.items():
        rendered = prompt.render(
            kernel_code="__global__ void k() {}",
            latency_ms=1.23,
            occupancy_pct=45.0,
            load_eff_pct=55.0,
            dram_bw_pct=66.0,
            ncu_json="{}",
            wrong_bottleneck="memory",
            correct_bottleneck="compute",
        )
        assert "system" in rendered
        assert "user" in rendered
        assert "__global__ void k() {}" in rendered["user"]
        assert prompt.expected_format.startswith("BOTTLENECK")
