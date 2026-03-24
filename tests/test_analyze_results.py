from __future__ import annotations

import json
from pathlib import Path

from eval.analyze_results import compute_consistency_scores, consistency_score, load_results


def test_consistency_score_uses_majority_vote() -> None:
    rows = [
        {"question_format": "label", "predicted_label": "memory-bound"},
        {"question_format": "yesno_memory", "predicted_label": "memory-bound"},
        {"question_format": "rank", "parsed_ranking": ["latency-bound", "memory-bound"]},
        {"question_format": "junior_wrong", "parsed_assessment": "AGREE"},
        {"question_format": "fix"},
    ]
    assert consistency_score(rows) == 0.75


def test_compute_consistency_scores_aggregates_per_model() -> None:
    rows = [
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_A", "question_format": "label", "predicted_label": "memory-bound"},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_A", "question_format": "yesno_memory", "predicted_label": "memory-bound"},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_A", "question_format": "rank", "parsed_ranking": ["memory-bound"]},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_A", "question_format": "junior_wrong", "parsed_assessment": "AGREE"},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_B", "question_format": "label", "predicted_label": "latency-bound"},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_B", "question_format": "yesno_memory", "predicted_label": "not-memory-bound"},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_B", "question_format": "rank", "parsed_ranking": ["latency-bound"]},
        {"model": "gpt-5.4", "level": 1, "kernel_id": "hw_B", "question_format": "junior_wrong", "parsed_assessment": "DISAGREE"},
    ]
    import pandas as pd

    detail, summary = compute_consistency_scores(pd.DataFrame(rows))
    assert len(detail) == 2
    assert summary.iloc[0]["Model"] == "gpt-5.4"
    assert 0.0 <= summary.iloc[0]["Mean Consistency"] <= 1.0


def test_load_results_infers_question_format_from_nested_path(tmp_path: Path) -> None:
    path = tmp_path / "results" / "gpt-5.4" / "level_1" / "rank"
    path.mkdir(parents=True)
    payload = {"kernel_id": "hw_A", "model": "gpt-5.4", "level": 1, "predicted_label": "memory-bound"}
    (path / "hw_A.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = load_results(tmp_path / "results")
    assert rows[0]["question_format"] == "rank"


def test_load_results_preserves_judge_payload(tmp_path: Path) -> None:
    path = tmp_path / "results" / "gpt-5.4" / "trial_1" / "level_1"
    path.mkdir(parents=True)
    payload = {
        "kernel_id": "hw_A",
        "model": "gpt-5.4",
        "level": 1,
        "predicted_label": "memory-bound",
        "judge": {"label_correct": 1, "reasoning_grounded": 1, "mislead_resistant": 1},
    }
    (path / "hw_A.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = load_results(tmp_path / "results")
    assert rows[0]["judge"]["label_correct"] == 1
