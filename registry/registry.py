from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from registry.kernel_entry import BottleneckLabel, KernelEntry, kernel_entry_from_dict, ncu_profile_from_dict


def ground_truth_verified(entry: KernelEntry) -> bool:
    return bool(entry.ncu_profile.raw.get("ground_truth_verified", True))


class KernelRegistry:
    def __init__(self, profile_dir: Path, mock: bool = False):
        self.profile_dir = Path(profile_dir)
        self.mock = mock
        self._entries: dict[str, KernelEntry] = {}

    def _load_jsonl(self, jsonl_path: Path) -> None:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            return
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = kernel_entry_from_dict(json.loads(line))
                self._entries[entry.id] = entry

    def load_mined(self, jsonl_path: Path) -> None:
        self._load_jsonl(jsonl_path)

    def load_kernelbot(self, jsonl_path: Path) -> None:
        self._load_jsonl(jsonl_path)

    def load_handwritten(self, kernels_dir: Path) -> None:
        kernels_dir = Path(kernels_dir)
        for kernel_dir in sorted(path for path in kernels_dir.iterdir() if path.is_dir()):
            code_path = kernel_dir / "kernel.cu"
            meta_path = kernel_dir / "meta.json"
            if not code_path.exists() or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            kernel_id = str(meta["id"])
            profile_path = self.profile_dir / f"{kernel_id}.json"
            if self.mock and not profile_path.exists():
                fixture_path = self.profile_dir / "fixtures" / f"{kernel_id}.json"
                if fixture_path.exists():
                    profile_path = fixture_path
            if not profile_path.exists():
                raise FileNotFoundError(f"Missing profile for handwritten kernel {kernel_id}: {profile_path}")
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            profile = ncu_profile_from_dict(profile_data)
            profile.raw = {"ground_truth_verified": True, **profile.raw}
            entry = KernelEntry(
                id=kernel_id,
                source="handwritten",
                code=code_path.read_text(encoding="utf-8"),
                pytorch_reference=meta.get("pytorch_reference"),
                true_bottleneck=meta["true_bottleneck"],
                misleading_signal=meta["misleading_signal"],
                category=meta["category"],
                difficulty=meta["difficulty"],
                hardware=meta.get("hardware", "A100"),
                ncu_profile=profile,
                task_id=None,
            )
            self._entries[entry.id] = entry

    def get(self, kernel_id: str) -> KernelEntry:
        return self._entries[kernel_id]

    def filter(
        self,
        source: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        true_bottleneck: Optional[BottleneckLabel] = None,
        ground_truth_verified: Optional[bool] = None,
    ) -> list[KernelEntry]:
        entries = list(self._entries.values())
        if source is not None:
            entries = [entry for entry in entries if entry.source == source]
        if category is not None:
            entries = [entry for entry in entries if entry.category == category]
        if difficulty is not None:
            entries = [entry for entry in entries if entry.difficulty == difficulty]
        if true_bottleneck is not None:
            entries = [entry for entry in entries if entry.true_bottleneck == true_bottleneck]
        if ground_truth_verified is not None:
            entries = [entry for entry in entries if ground_truth_verified == bool(entry.ncu_profile.raw.get("ground_truth_verified", True))]
        return sorted(entries, key=lambda entry: entry.id)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[KernelEntry]:
        for entry_id in sorted(self._entries):
            yield self._entries[entry_id]
