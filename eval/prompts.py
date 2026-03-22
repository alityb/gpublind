from __future__ import annotations

from dataclasses import dataclass

from jinja2 import Template

from registry.kernel_entry import BottleneckLabel

VALID_LABELS = [
    "memory-bound", "compute-bound", "latency-bound",
    "occupancy-limited", "register-spill"
]


@dataclass
class Prompt:
    level: int
    name: str
    system: str
    user_template: str
    expected_format: str

    def render(self, **kwargs: object) -> dict[str, str]:
        return {
            "system": self.system,
            "user": Template(self.user_template).render(**kwargs)
        }


SYSTEM_BASE = """You are a GPU performance engineer. When asked to diagnose
a kernel bottleneck, respond with exactly this format:
BOTTLENECK: <label>
REASONING: <2-3 sentences citing specific evidence from the code or metrics>

Valid labels: memory-bound, compute-bound, latency-bound,
occupancy-limited, register-spill"""

PROMPTS: dict[int, Prompt] = {
    1: Prompt(
        level=1,
        name="code_only",
        system=SYSTEM_BASE,
        user_template="""Diagnose the primary performance bottleneck 
in this CUDA kernel:

```cuda
{{ kernel_code }}
```""",
        expected_format="BOTTLENECK: <label>\nREASONING: <explanation>"
    ),
    2: Prompt(
        level=2,
        name="code_plus_basic_metrics",
        system=SYSTEM_BASE,
        user_template="""Diagnose the primary performance bottleneck
in this CUDA kernel.

Profiling data:
- Latency: {{ latency_ms }}ms
- Achieved occupancy: {{ occupancy_pct }}%
- Global load efficiency: {{ load_eff_pct }}%
- DRAM bandwidth utilization: {{ dram_bw_pct }}%

```cuda
{{ kernel_code }}
```""",
        expected_format="BOTTLENECK: <label>\nREASONING: <explanation>"
    ),
    3: Prompt(
        level=3,
        name="code_plus_full_ncu",
        system=SYSTEM_BASE,
        user_template="""Diagnose the primary performance bottleneck
in this CUDA kernel using the full profiler output.

NCU Profile:
{{ ncu_json }}

```cuda
{{ kernel_code }}
```""",
        expected_format="BOTTLENECK: <label>\nREASONING: <explanation>"
    ),
    4: Prompt(
        level=4,
        name="adversarial_wrong_framing",
        system=SYSTEM_BASE,
        user_template="""A colleague claims this kernel is 
{{ wrong_bottleneck }}. Do you agree?
Explain why or why not, then give your own diagnosis.

```cuda
{{ kernel_code }}
```""",
        expected_format="BOTTLENECK: <label>\nREASONING: <explanation>"
    ),
    5: Prompt(
        level=5,
        name="adversarial_correct_framing",
        system=SYSTEM_BASE,
        user_template="""A colleague claims this kernel is
{{ correct_bottleneck }}. Do you agree?
Explain why or why not, then give your own diagnosis.

```cuda
{{ kernel_code }}
```""",
        expected_format="BOTTLENECK: <label>\nREASONING: <explanation>"
    ),
}
