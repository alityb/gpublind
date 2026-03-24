from __future__ import annotations

from dataclasses import dataclass
import re

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


SYSTEM_BASE = """You are a GPU performance engineer.
Follow the user's requested output format exactly.

Valid bottleneck labels: memory-bound, compute-bound, latency-bound,
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

Hardware: {{ hardware_context }}

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

QUESTION_FORMATS = {
    "label": {
        "description": "Standard label prediction",
        "user_suffix": """What is the primary performance bottleneck? Use exactly this format:
BOTTLENECK: <label>
REASONING: <2-3 sentences>""",
    },
    "yesno_memory": {
        "description": "Binary memory-bound check",
        "user_suffix": """Is this kernel memory-bound?
Answer YES or NO, then explain in 2-3 sentences why.
MEMORY_BOUND: <YES or NO>
REASONING: <explanation>""",
    },
    "rank": {
        "description": "Rank all bottleneck types",
        "user_suffix": """Rank these bottleneck types from most to least likely for this kernel:
memory-bound, compute-bound, latency-bound, occupancy-limited, register-spill

Format:
RANK_1: <most likely>
RANK_2: <second>
RANK_3: <third>
RANK_4: <fourth>
RANK_5: <least likely>
REASONING: <2-3 sentences>""",
    },
    "junior_wrong": {
        "description": "Adversarial explanation - junior engineer wrong",
        "user_suffix": """A junior engineer says: 'This kernel is slow because it's saturating memory bandwidth.'
Are they correct? Answer AGREE or DISAGREE, then explain.
ASSESSMENT: <AGREE or DISAGREE>
REASONING: <2-3 sentences citing specific evidence>""",
    },
    "junior_right": {
        "description": "Non-adversarial explanation - junior right",
        "user_suffix": """A junior engineer says: 'This kernel is slow because {{ correct_explanation }}.'
Are they correct? Answer AGREE or DISAGREE, then explain.
ASSESSMENT: <AGREE or DISAGREE>
REASONING: <2-3 sentences>""",
    },
    "fix": {
        "description": "Optimization recommendation",
        "user_suffix": """What single code change would most improve this kernel's performance? Be specific.
CHANGE: <one sentence describing the change>
EXPECTED_IMPROVEMENT: <what metric would improve and why>""",
    },
    "metrics_only": {
        "description": "Metrics interpretation without code",
        "user_suffix": """Without seeing the kernel code, given only these metrics:
- Achieved memory bandwidth: {{ dram_bw_pct }}% of peak
- Achieved FLOPS: {{ compute_pct }}% of peak
- Warp stall (long scoreboard): {{ stall_long_pct }}%
- Achieved occupancy: {{ occupancy_pct }}%

What is the primary bottleneck?
BOTTLENECK: <label>
REASONING: <2-3 sentences>""",
    },
}


def _strip_code_block(rendered_user: str) -> str:
    stripped = re.sub(r"\n```cuda\n.*?\n```", "", rendered_user, flags=re.DOTALL)
    return stripped.strip()


def render_prompt(level: int, question_format: str = "label", **kwargs: object) -> dict[str, str]:
    prompt = PROMPTS[level]
    base_user = prompt.render(**kwargs)["user"].strip()
    if question_format == "metrics_only":
        base_user = _strip_code_block(base_user)
    suffix = Template(QUESTION_FORMATS[question_format]["user_suffix"]).render(**kwargs).strip()
    user = suffix if not base_user else f"{base_user}\n\n{suffix}"
    return {
        "system": prompt.system,
        "user": user,
    }
