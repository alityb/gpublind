from __future__ import annotations

from pathlib import Path

from profiles.generate_profiles import write_mock_fixtures
from registry.registry import KernelRegistry


def test_kernel_registry_load_and_filter(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    write_mock_fixtures(profiles_dir)
    registry = KernelRegistry(profile_dir=profiles_dir, mock=True)
    registry.load_handwritten(Path("kernels"))
    assert len(registry) == 5
    assert registry.get("hw_A").true_bottleneck == "memory-bound"
    filtered = registry.filter(category="red_herring")
    assert [entry.id for entry in filtered] == ["hw_C"]
