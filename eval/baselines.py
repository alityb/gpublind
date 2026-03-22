from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from registry.kernel_entry import KernelEntry

LABELS = [
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "occupancy-limited",
    "register-spill",
]


def load_roof(path: Path = Path("profiles/hardware_roof.json")) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ridge_point_for_entry(entry: KernelEntry, roof: dict[str, Any]) -> float:
    if "ridge_point" in entry.ncu_profile.raw:
        return float(entry.ncu_profile.raw["ridge_point"])
    return float(roof["peak_flops_tflops"]) / float(roof["peak_bw_tbps"])


def estimate_flops_and_bytes(code: str) -> tuple[float, float]:
    flops = sum(len(re.findall(pattern, code)) for pattern in [r"\+", r"-", r"\*", r"/"])
    global_refs = len(re.findall(r"\[[^\]]+\]", code))
    bytes_moved = max(global_refs * 4.0, 1.0)
    return float(flops), float(bytes_moved)


def random_baseline(entries: Iterable[KernelEntry], trials: int = 1000, seed: int = 7) -> dict[str, float]:
    entry_list = list(entries)
    rng = random.Random(seed)
    accuracies: list[float] = []
    for _ in range(trials):
        correct = sum(1 for entry in entry_list if rng.choice(LABELS) == entry.true_bottleneck)
        accuracies.append(correct / max(len(entry_list), 1))
    return {"mean": mean(accuracies), "std": pstdev(accuracies)}


def frequency_baseline(entries: Iterable[KernelEntry]) -> dict[str, object]:
    entry_list = list(entries)
    counts = Counter(entry.true_bottleneck for entry in entry_list)
    if not counts:
        return {"label": None, "accuracy": 0.0, "per_category_accuracy": {}}
    label = counts.most_common(1)[0][0]
    per_category: dict[str, float] = {}
    by_category: dict[str, list[KernelEntry]] = defaultdict(list)
    for entry in entry_list:
        by_category[entry.category].append(entry)
    for category, category_entries in by_category.items():
        per_category[category] = sum(1 for entry in category_entries if entry.true_bottleneck == label) / len(category_entries)
    accuracy = sum(1 for entry in entry_list if entry.true_bottleneck == label) / len(entry_list)
    return {"label": label, "accuracy": accuracy, "per_category_accuracy": per_category}


def roofline_prediction(entry: KernelEntry, roof: dict[str, Any]) -> str:
    flops, bytes_moved = estimate_flops_and_bytes(entry.code)
    arithmetic_intensity = flops / bytes_moved
    ridge_point = ridge_point_for_entry(entry, roof)
    if entry.true_bottleneck in {"latency-bound", "occupancy-limited", "register-spill"}:
        return "unknown"
    return "compute-bound" if arithmetic_intensity > ridge_point else "memory-bound"


def roofline_baseline(entries: Iterable[KernelEntry], roof: dict[str, Any] | None = None) -> dict[str, object]:
    entry_list = list(entries)
    roof_data = roof or load_roof()
    predictions: list[dict[str, object]] = []
    correct = 0
    for entry in entry_list:
        predicted = roofline_prediction(entry, roof_data)
        is_correct = predicted == entry.true_bottleneck
        correct += int(is_correct)
        predictions.append({"kernel_id": entry.id, "predicted_label": predicted, "correct": is_correct})
    accuracy = correct / max(len(entry_list), 1)
    return {"accuracy": accuracy, "predictions": predictions}
