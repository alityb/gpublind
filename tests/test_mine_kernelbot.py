from __future__ import annotations

from data.mine_kernelbot import build_mock_rows, mine_candidates


def test_mock_kernelbot_mining_uses_sentinel_profiles() -> None:
    entries = mine_candidates(build_mock_rows(30), limit=30)

    assert len(entries) == 30
    assert all(entry.source == "kernelbot" for entry in entries)
    assert all(entry.ncu_profile.arithmetic_intensity == -1.0 for entry in entries)
    assert all(entry.ncu_profile.register_count == -1 for entry in entries)
    assert all(entry.ncu_profile.raw["needs_profiling"] is True for entry in entries)
    assert all(entry.ncu_profile.raw["ground_truth_verified"] is False for entry in entries)
