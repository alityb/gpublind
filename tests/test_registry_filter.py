from __future__ import annotations

from pathlib import Path

from data.mine_kernelbot import build_mock_rows, mine_candidates as mine_kernelbot_candidates, write_jsonl as write_kernelbot_jsonl
from data.mine_sakana import build_mock_rows as build_sakana_rows, mine_candidates as mine_sakana_candidates, write_jsonl as write_sakana_jsonl
from profiles.generate_profiles import write_mock_fixtures
from registry.registry import KernelRegistry


def test_filter_ground_truth_verified_excludes_kernelbot_sentinels(tmp_path: Path) -> None:
    mined_path = tmp_path / "mined.jsonl"
    kernelbot_path = tmp_path / "kernelbot.jsonl"
    profiles_dir = tmp_path / "profiles"

    write_sakana_jsonl(mine_sakana_candidates(build_sakana_rows(70), 70), mined_path)
    write_kernelbot_jsonl(mine_kernelbot_candidates(build_mock_rows(30), 30), kernelbot_path)
    write_mock_fixtures(profiles_dir)

    registry = KernelRegistry(profile_dir=profiles_dir, mock=True)
    registry.load_mined(mined_path)
    registry.load_kernelbot(kernelbot_path)
    registry.load_handwritten(Path("kernels"))

    verified = registry.filter(ground_truth_verified=True)
    unverified = registry.filter(ground_truth_verified=False)

    assert all(entry.ncu_profile.raw.get("ground_truth_verified", True) is True for entry in verified)
    assert all(entry.source == "kernelbot" for entry in unverified)
    assert all(entry.ncu_profile.raw.get("ground_truth_verified") is False for entry in unverified)
