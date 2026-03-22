from __future__ import annotations

from pathlib import Path

from data.mine_kernelbot import build_mock_rows as build_kernelbot_rows, mine_candidates as mine_kernelbot_candidates, write_jsonl as write_kernelbot_jsonl
from data.mine_sakana import build_mock_rows as build_sakana_rows, mine_candidates as mine_sakana_candidates, write_jsonl as write_sakana_jsonl
from profiles.generate_profiles import write_mock_fixtures
from registry.build_registry import build_registry


def test_full_mock_build_produces_at_least_100_entries(tmp_path: Path) -> None:
    mined_path = tmp_path / "mined.jsonl"
    kernelbot_path = tmp_path / "kernelbot.jsonl"
    profiles_dir = tmp_path / "profiles"

    write_sakana_jsonl(mine_sakana_candidates(build_sakana_rows(70), 70), mined_path)
    write_kernelbot_jsonl(mine_kernelbot_candidates(build_kernelbot_rows(30), 30), kernelbot_path)
    write_mock_fixtures(profiles_dir)

    entries = build_registry(
        mined=mined_path,
        kernelbot=kernelbot_path,
        kernels=Path("kernels"),
        profiles=profiles_dir,
        mock=True,
    )

    assert len(entries) >= 100
