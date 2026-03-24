from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class CorpusEntry:
    root: Path
    kernel_path: Path
    meta_path: Path
    profile_path: Path
    kernel_code: str
    meta: dict[str, Any]
    profile: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.meta["id"])

    @property
    def true_bottleneck(self) -> str:
        return str(self.meta["true_bottleneck"])

    @property
    def misleading_signal(self) -> str:
        return str(self.meta.get("misleading_signal", ""))

    @property
    def correct_explanation(self) -> str:
        return str(self.meta.get("correct_explanation", ""))

    @property
    def difficulty(self) -> str:
        return str(self.meta.get("difficulty", "unknown"))

    @property
    def category(self) -> str:
        return str(self.meta.get("category", self.true_bottleneck))

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    @property
    def hardware(self) -> str:
        return str(self.meta.get("hardware", self.profile.get("hardware", {}).get("name", "unknown")))

    @property
    def confidence(self) -> str:
        verification = self.profile.get("verification", {})
        return str(verification.get("confidence", "low"))

    @property
    def reasoning_rubric(self) -> dict[str, Any]:
        rubric = self.meta.get("reasoning_rubric", {})
        return rubric if isinstance(rubric, dict) else {}


def load_entry(kernel_dir: Path) -> CorpusEntry:
    kernel_path = kernel_dir / "kernel.cu"
    meta_path = kernel_dir / "meta.json"
    profile_path = kernel_dir / "profile.json"
    return CorpusEntry(
        root=kernel_dir,
        kernel_path=kernel_path,
        meta_path=meta_path,
        profile_path=profile_path,
        kernel_code=kernel_path.read_text(encoding="utf-8"),
        meta=json.loads(meta_path.read_text(encoding="utf-8")),
        profile=json.loads(profile_path.read_text(encoding="utf-8")),
    )


def load_corpus(root: Path = Path("corpus/kernels"), min_confidence: str = "medium") -> list[CorpusEntry]:
    minimum = CONFIDENCE_ORDER.get(min_confidence, 1)
    entries: list[CorpusEntry] = []
    for kernel_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            entry = load_entry(kernel_dir)
        except FileNotFoundError:
            continue
        if CONFIDENCE_ORDER.get(entry.confidence, 0) >= minimum:
            entries.append(entry)
    return entries
