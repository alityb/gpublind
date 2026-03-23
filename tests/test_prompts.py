from __future__ import annotations

from eval.prompts import PROMPTS, QUESTION_FORMATS, render_prompt


def test_prompt_rendering_all_levels() -> None:
    for level, prompt in PROMPTS.items():
        rendered = render_prompt(
            level,
            "label",
            kernel_code="__global__ void k() {}",
            latency_ms=1.23,
            occupancy_pct=45.0,
            load_eff_pct=55.0,
            dram_bw_pct=66.0,
            compute_pct=12.0,
            stall_long_pct=34.0,
            ncu_json="{}",
            wrong_bottleneck="memory",
            correct_bottleneck="compute",
            correct_explanation="dependency chains create long waits",
        )
        assert "system" in rendered
        assert "user" in rendered
        assert "__global__ void k() {}" in rendered["user"]
        assert prompt.expected_format.startswith("BOTTLENECK")


def test_metrics_only_prompt_renders_without_kernel_code() -> None:
    rendered = render_prompt(
        3,
        "metrics_only",
        kernel_code="__global__ void hidden() {}",
        latency_ms=1.23,
        occupancy_pct=45.0,
        load_eff_pct=55.0,
        dram_bw_pct=66.0,
        compute_pct=12.0,
        stall_long_pct=34.0,
        ncu_json="{}",
        wrong_bottleneck="memory-bound",
        correct_bottleneck="latency-bound",
        correct_explanation="dependency chains create long waits",
    )
    assert "__global__ void hidden() {}" not in rendered["user"]
    assert "Achieved memory bandwidth: 66.0% of peak" in rendered["user"]


def test_question_formats_cover_requested_variants() -> None:
    assert {"label", "yesno_memory", "rank", "junior_wrong", "junior_right", "fix", "metrics_only"} <= set(
        QUESTION_FORMATS
    )
