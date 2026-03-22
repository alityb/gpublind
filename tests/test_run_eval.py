from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from data.mine_kernelbot import build_mock_rows as build_kernelbot_rows, mine_candidates as mine_kernelbot_candidates, write_jsonl as write_kernelbot_jsonl
from data.mine_sakana import build_mock_rows as build_sakana_rows, mine_candidates as mine_sakana_candidates, write_jsonl as write_sakana_jsonl
from eval.run_eval import call_model, main, parse_response, score_prediction, wrong_label_for
from profiles.generate_profiles import write_mock_fixtures
from registry.kernel_entry import SEVERITY


def test_parse_response_valid() -> None:
    label, reasoning = parse_response("BOTTLENECK: memory-bound\nREASONING: Because the loads are scattered.")
    assert label == "memory-bound"
    assert "scattered" in reasoning


def test_parse_response_malformed() -> None:
    label, reasoning = parse_response("REASONING: missing label")
    assert label == "parse_error"
    assert reasoning == "missing label"


def test_parse_response_missing_reasoning() -> None:
    label, reasoning = parse_response("BOTTLENECK: compute-bound")
    assert label == "compute-bound"
    assert reasoning == ""


def test_scoring_logic_uses_severity_matrix() -> None:
    score = score_prediction("memory-bound", "compute-bound", "compute-bound")
    assert score["correct"] is False
    assert score["severity"] == SEVERITY[("memory-bound", "compute-bound")]
    assert score["fell_for_adversarial"] is True


def test_wrong_label_for_uses_semantic_opposites() -> None:
    assert wrong_label_for("memory-bound") == "compute-bound"
    assert wrong_label_for("compute-bound") == "memory-bound"
    assert wrong_label_for("latency-bound") == "memory-bound"
    assert wrong_label_for("occupancy-limited") == "compute-bound"
    assert wrong_label_for("register-spill") == "memory-bound"


def test_call_model_retries_rate_limit_then_succeeds() -> None:
    class RateLimitError(Exception):
        pass

    class Response:
        def __init__(self, content: str) -> None:
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    state = {"calls": 0, "sleeps": []}

    def fake_completion(**_: object) -> Response:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RateLimitError("slow down")
        return Response("BOTTLENECK: memory-bound\nREASONING: recovered")

    raw_response, api_error = call_model(
        model_name="gpt-4o",
        rendered_prompt={"system": "s", "user": "u"},
        mock=False,
        entry=SimpleNamespace(id="kernel_x"),
        level=1,
        completion_fn=fake_completion,
        sleep_fn=lambda seconds: state["sleeps"].append(seconds),
    )

    assert api_error is False
    assert "BOTTLENECK: memory-bound" in raw_response
    assert state["calls"] == 2
    assert state["sleeps"] == [2.0, 0.5]


def test_dry_run_prints_cost_estimate_and_writes_no_results(tmp_path: Path, capsys: object) -> None:
    mined_path = tmp_path / "mined.jsonl"
    kernelbot_path = tmp_path / "kernelbot.jsonl"
    profiles_dir = tmp_path / "profiles"
    output_dir = tmp_path / "results"

    write_sakana_jsonl(mine_sakana_candidates(build_sakana_rows(70), 70), mined_path)
    write_kernelbot_jsonl(mine_kernelbot_candidates(build_kernelbot_rows(30), 30), kernelbot_path)
    write_mock_fixtures(profiles_dir)

    exit_code = main(
        [
            "--model",
            "gpt-4o",
            "--mock",
            "--dry-run",
            "--mined",
            str(mined_path),
            "--kernelbot",
            str(kernelbot_path),
            "--kernels",
            "kernels",
            "--profiles",
            str(profiles_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Estimated cost:" in captured.out
    assert "Total calls:" in captured.out
    assert not output_dir.exists()
