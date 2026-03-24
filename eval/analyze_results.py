from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from eval.baselines import frequency_baseline, random_baseline, roofline_baseline, roofline_plus_baseline
from registry import KernelRegistry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GPUBlind results")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernelbot", type=Path, default=Path("data/kernelbot_kernels.jsonl"))
    parser.add_argument("--kernelbench-compute", type=Path, default=Path("data/kernelbench_compute_candidates.jsonl"))
    parser.add_argument("--latency", type=Path, default=Path("data/latency_bound_candidates.jsonl"))
    parser.add_argument("--register-spill", type=Path, default=Path("data/register_spill_candidates.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
    parser.add_argument("--include-formats", action="store_true")
    parser.add_argument("--min-confidence", choices=["high", "medium", "any"], default="medium")
    parser.add_argument("--subset", type=Path, default=None)
    return parser.parse_args(argv)


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n|---|\n| no data |"
    headers = [str(column) for column in df.columns]
    separator = ["---"] * len(headers)
    rows = [headers, separator]
    for record in df.itertuples(index=False, name=None):
        rows.append([str(value) for value in record])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def infer_result_metadata(path: Path, results_dir: Path) -> dict[str, Any]:
    relative_parts = path.relative_to(results_dir).parts
    metadata: dict[str, Any] = {"model": relative_parts[0], "trial": 1, "question_format": "label"}
    for index, part in enumerate(relative_parts):
        if part.startswith("trial_"):
            try:
                metadata["trial"] = int(part.split("_", 1)[1])
            except ValueError:
                metadata["trial"] = 1
        elif part.startswith("level_"):
            metadata["level"] = int(part.split("_", 1)[1])
            if index + 1 < len(relative_parts) - 1:
                metadata["question_format"] = relative_parts[index + 1]
    return metadata


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if "contamination" in path.relative_to(results_dir).parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        metadata = infer_result_metadata(path, results_dir)
        if isinstance(payload, list):
            for row in payload:
                for key, value in metadata.items():
                    row.setdefault(key, value)
                rows.append(row)
        else:
            for key, value in metadata.items():
                payload.setdefault(key, value)
            rows.append(payload)
    return rows


def load_contamination_results(results_dir: Path) -> list[dict[str, Any]]:
    contamination_dir = results_dir / "contamination"
    rows: list[dict[str, Any]] = []
    if not contamination_dir.exists():
        return rows
    for path in sorted(contamination_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    return rows


def load_registry(
    mined: Path,
    kernelbot: Path,
    kernelbench_compute: Path,
    latency: Path,
    register_spill: Path,
    kernels: Path,
    profiles: Path,
    min_confidence: str,
) -> KernelRegistry:
    registry = KernelRegistry(profile_dir=profiles, mock=True)
    registry.load_mined(mined)
    registry.load_mined(kernelbench_compute)
    registry.load_mined(latency)
    registry.load_mined(register_spill)
    registry.load_kernelbot(kernelbot)
    registry.load_handwritten(kernels)
    return registry


def load_subset_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("kernel_ids", []) if isinstance(payload, dict) else payload
    return {str(item) for item in items}


def latest_per_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            int(row.get("trial", 1)),
            int(row["level"]),
            str(row["kernel_id"]),
            str(row.get("question_format", "label")),
        )
        latest[key] = row
    return list(latest.values())


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def format_ci_value(value: float, lower: float, upper: float) -> str:
    warning = " NOTE: underpowered" if (upper - lower) * 100.0 > 30.0 else ""
    return f"{value * 100.0:.1f}% [{lower * 100.0:.1f}%, {upper * 100.0:.1f}%]{warning}"


def bootstrap_mean_ci(values: list[float], samples: int = 1000, seed: int = 7) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return lower, upper


def consistency_score(results_for_one_kernel_one_model: list[dict[str, Any]]) -> float:
    labels: list[str] = []
    for row in results_for_one_kernel_one_model:
        fmt = row.get("question_format", "label")
        if fmt == "fix" or fmt == "junior_right":
            continue
        if fmt == "rank":
            ranking = row.get("parsed_ranking") or []
            if ranking:
                labels.append(str(ranking[0]))
        elif fmt == "junior_wrong":
            assessment = row.get("parsed_assessment")
            if assessment == "AGREE":
                labels.append("memory-bound")
            elif assessment == "DISAGREE":
                labels.append("not-memory-bound")
        elif fmt == "yesno_memory":
            labels.append(str(row.get("predicted_label")))
        elif fmt == "label":
            labels.append(str(row.get("predicted_label")))
    if not labels:
        return 0.0
    majority = Counter(labels).most_common(1)[0][0]
    return sum(label == majority for label in labels) / len(labels)


def compute_consistency_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["model", "level", "kernel_id"], dropna=False)
    for (model, level, kernel_id), subset in grouped:
        score = consistency_score(subset.to_dict("records"))
        rows.append(
            {
                "Model": model,
                "Level": int(level),
                "Kernel": kernel_id,
                "Consistency": round(score, 4),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("Model")["Consistency"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "Mean Consistency", "std": "Std", "min": "Min", "max": "Max"})
    )
    ci_strings: list[str] = []
    for model in summary["Model"]:
        values = detail[detail["Model"] == model]["Consistency"].tolist()
        lower, upper = bootstrap_mean_ci(values)
        ci_strings.append(f"[{lower:.3f}, {upper:.3f}]")
    summary["95% CI"] = ci_strings
    for column in ["Mean Consistency", "Std", "Min", "Max"]:
        summary[column] = summary[column].fillna(0.0).round(4)
    return detail, summary


def accuracy_table(df: pd.DataFrame, registry: KernelRegistry) -> pd.DataFrame:
    models = sorted(df["model"].unique()) if not df.empty else []
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"Model": model}
        for level in range(1, 6):
            subset = df[(df["model"] == model) & (df["level"] == level)]
            if subset.empty:
                row[f"L{level}"] = None
            else:
                successes = int(subset["correct"].astype(bool).sum())
                total = len(subset)
                lower, upper = wilson_interval(successes, total)
                row[f"L{level}"] = format_ci_value(successes / total, lower, upper)
        rows.append(row)
    random_row = {"Model": "Random"}
    random_stats = random_baseline(list(registry))
    for level in range(1, 6):
        total = max(len(list(registry)), 1)
        mean_accuracy = float(random_stats["mean"])
        lower, upper = wilson_interval(int(round(mean_accuracy * total)), total)
        random_row[f"L{level}"] = format_ci_value(mean_accuracy, lower, upper)
    rows.append(random_row)
    frequency = frequency_baseline(list(registry))
    frequency_row = {"Model": "Frequency"}
    for level in range(1, 6):
        total = max(len(list(registry)), 1)
        accuracy = float(frequency["accuracy"])
        lower, upper = wilson_interval(int(round(accuracy * total)), total)
        frequency_row[f"L{level}"] = format_ci_value(accuracy, lower, upper)
    rows.append(frequency_row)
    roofline = roofline_baseline(list(registry))
    roofline_row = {"Model": "Roofline"}
    for level in range(1, 6):
        total = max(len(list(registry)), 1)
        accuracy = float(roofline["accuracy"])
        lower, upper = wilson_interval(int(round(accuracy * total)), total)
        roofline_row[f"L{level}"] = format_ci_value(accuracy, lower, upper)
    rows.append(roofline_row)
    roofline_plus = roofline_plus_baseline(list(registry))
    roofline_plus_row = {"Model": "Roofline+"}
    for level in range(1, 6):
        total = max(len(list(registry)), 1)
        accuracy = float(roofline_plus["accuracy"])
        lower, upper = wilson_interval(int(round(accuracy * total)), total)
        roofline_plus_row[f"L{level}"] = format_ci_value(accuracy, lower, upper)
    rows.append(roofline_plus_row)
    return pd.DataFrame(rows)


def write_confusion_matrices(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        "memory-bound",
        "compute-bound",
        "latency-bound",
        "occupancy-limited",
        "register-spill",
        "parse_error",
    ]
    for model in sorted(df["model"].unique()) if not df.empty else []:
        subset = df[df["model"] == model]
        count_matrix = pd.crosstab(subset["true_bottleneck"], subset["predicted_label"]).reindex(index=labels[:-1], columns=labels, fill_value=0)
        severity_matrix = pd.pivot_table(
            subset,
            index="true_bottleneck",
            columns="predicted_label",
            values="severity",
            aggfunc="sum",
            fill_value=0,
        ).reindex(index=labels[:-1], columns=labels, fill_value=0)
        content = "# Count\n\n" + frame_to_markdown(count_matrix.reset_index()) + "\n\n# Severity Weighted\n\n" + frame_to_markdown(severity_matrix.reset_index()) + "\n"
        (output_dir / f"{model}.md").write_text(content, encoding="utf-8")


def write_level_sensitivity(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["model"].unique()) if not df.empty else []:
        l1 = float(df[(df["model"] == model) & (df["level"] == 1)]["correct"].mean())
        l2 = float(df[(df["model"] == model) & (df["level"] == 2)]["correct"].mean())
        l3 = float(df[(df["model"] == model) & (df["level"] == 3)]["correct"].mean())
        rows.append(
            {
                "Model": model,
                "L1": round(100.0 * l1, 2),
                "L2": round(100.0 * l2, 2),
                "L3": round(100.0 * l3, 2),
                "L2-L1": round(100.0 * (l2 - l1), 2),
                "L3-L2": round(100.0 * (l3 - l2), 2),
                "L3-L1": round(100.0 * (l3 - l1), 2),
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_sycophancy(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["model"].unique()) if not df.empty else []:
        l4 = df[(df["model"] == model) & (df["level"] == 4)]
        l5 = df[(df["model"] == model) & (df["level"] == 5)]
        l4_wrong_rate = float(l4["fell_for_adversarial"].astype("boolean").fillna(False).astype(bool).mean()) if not l4.empty else 0.0
        l5_wrong_rate = 1.0 - float(l5["correct"].mean()) if not l5.empty else 0.0
        rows.append(
            {
                "Model": model,
                "L4 wrong agreement rate": round(100.0 * l4_wrong_rate, 2),
                "L5 wrong agreement rate": round(100.0 * l5_wrong_rate, 2),
                "Sycophancy": round(100.0 * (l4_wrong_rate - l5_wrong_rate), 2),
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_category_breakdown(df: pd.DataFrame, registry: KernelRegistry, output_path: Path) -> pd.DataFrame:
    category_lookup = {entry.id: entry.category for entry in registry}
    enriched = df.copy()
    enriched["category"] = enriched["kernel_id"].map(category_lookup)
    table = enriched.groupby(["category", "model"], dropna=False)["correct"].mean().reset_index()
    pivot = table.pivot(index="category", columns="model", values="correct").fillna(0.0) * 100.0
    hard_failures = [category for category, row in pivot.iterrows() if (row == 0.0).all()]
    content = frame_to_markdown(pivot.reset_index()) + "\n\nHard cases where all models fail: " + (", ".join(hard_failures) if hard_failures else "none") + "\n"
    output_path.write_text(content, encoding="utf-8")
    return pivot.reset_index()


def write_reasoning_quality(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["model"].unique()) if not df.empty else []:
        subset = df[df["model"] == model]
        correct_subset = subset[subset["correct"] == True]
        if correct_subset.empty:
            reasoning_quality = 0.0
            got_lucky = 0.0
        else:
            correct_reasoning = correct_subset["correct_reasoning"].astype("boolean").fillna(False).astype(bool)
            reasoning_quality = float(correct_reasoning.mean())
            got_lucky = float((correct_reasoning == False).mean())
        rows.append(
            {
                "Model": model,
                "Reasoning Quality": round(100.0 * reasoning_quality, 2),
                "Got Lucky": round(100.0 * got_lucky, 2),
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_judge_reasoning_quality(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["model"].unique()) if not df.empty else []:
        subset = df[df["model"] == model]
        judge_rows = subset[subset["judge"].notna()] if "judge" in subset.columns else subset.iloc[0:0]
        if judge_rows.empty:
            rows.append(
                {
                    "Model": model,
                    "Judge N": 0,
                    "Grounded %": 0.0,
                    "Mislead Resistant %": 0.0,
                    "Correct Label, Wrong Reasoning %": 0.0,
                }
            )
            continue
        label_correct = judge_rows["judge"].apply(lambda item: int(dict(item).get("label_correct", 0)))
        grounded = judge_rows["judge"].apply(lambda item: int(dict(item).get("reasoning_grounded", 0)))
        resistant = judge_rows["judge"].apply(lambda item: int(dict(item).get("mislead_resistant", 0)))
        lucky = judge_rows["judge"].apply(
            lambda item: int(dict(item).get("label_correct", 0) == 1 and (dict(item).get("reasoning_grounded", 0) == 0 or dict(item).get("mislead_resistant", 0) == 0))
        )
        rows.append(
            {
                "Model": model,
                "Judge N": len(judge_rows),
                "Grounded %": round(100.0 * float(grounded.mean()), 2),
                "Mislead Resistant %": round(100.0 * float(resistant.mean()), 2),
                "Correct Label, Wrong Reasoning %": round(100.0 * float(lucky.mean()), 2),
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def automatic_grounded_reasoning(row: dict[str, Any], registry: KernelRegistry) -> bool | None:
    kernel_id = str(row.get("kernel_id", ""))
    if not kernel_id:
        return False
    try:
        entry = registry.get(kernel_id)
    except KeyError:
        return None
    rubric = entry.reasoning_rubric or {}
    must_mention = [str(item).lower() for item in rubric.get("must_mention", [])]
    must_not = [str(item).lower() for item in rubric.get("must_not_cite_as_primary", [])]
    if not must_mention:
        return None
    reasoning = str(row.get("parsed_reasoning") or row.get("raw_response") or "").strip().lower()
    if not reasoning:
        return False
    first_sentence = reasoning.split(".", 1)[0]
    mentions_required = any(token in reasoning for token in must_mention)
    avoids_red_herring = not any(token in first_sentence for token in must_not)
    return bool(row.get("correct") is True and mentions_required and avoids_red_herring)


def write_groundedness(df: pd.DataFrame, registry: KernelRegistry, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["model"].unique()) if not df.empty else []:
        subset = df[df["model"] == model]
        records = subset.to_dict("records")
        grounded_correct = [automatic_grounded_reasoning(record, registry) for record in records]
        scored = [value for value in grounded_correct if value is not None]
        correct_total = int(subset["correct"].astype(bool).sum()) if not subset.empty else 0
        grounded_total = sum(value is True for value in scored)
        total = len(scored)
        grounded_rate = (grounded_total / total) if total else 0.0
        grounded_given_correct = (grounded_total / correct_total) if correct_total else 0.0
        rows.append(
            {
                "Model": model,
                "Grounded Correct": round(100.0 * grounded_rate, 2),
                "Grounded Given Correct": round(100.0 * grounded_given_correct, 2),
                "Grounded Count": grounded_total,
                "Groundable Total": total,
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_contamination_report(label_df: pd.DataFrame, contamination_rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    if not contamination_rows:
        table = pd.DataFrame(columns=["Model", "Flagged %", "Accuracy Seen Before", "Accuracy Unseen"])
        output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
        return table
    contamination_df = pd.DataFrame(contamination_rows)
    contamination_df["seen_before"] = contamination_df["seen_before"].astype(bool)
    merged = label_df.merge(contamination_df[["model", "kernel_id", "seen_before"]], on=["model", "kernel_id"], how="left")
    rows: list[dict[str, Any]] = []
    for model in sorted(merged["model"].unique()) if not merged.empty else []:
        subset = merged[merged["model"] == model]
        flagged = subset["seen_before"].fillna(False).astype(bool)
        seen_subset = subset[flagged]
        unseen_subset = subset[~flagged]
        rows.append(
            {
                "Model": model,
                "Flagged %": round(100.0 * float(flagged.mean()), 2),
                "Accuracy Seen Before": round(100.0 * float(seen_subset["correct"].mean()), 2) if not seen_subset.empty else 0.0,
                "Accuracy Unseen": round(100.0 * float(unseen_subset["correct"].mean()), 2) if not unseen_subset.empty else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_difficulty_calibration(label_df: pd.DataFrame, registry: KernelRegistry, output_path: Path) -> pd.DataFrame:
    if label_df.empty:
        table = pd.DataFrame(columns=["Difficulty", "N kernels"])
        output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
        return table
    difficulty_lookup = {entry.id: entry.difficulty for entry in registry}
    enriched = label_df.copy()
    enriched["Difficulty"] = enriched["kernel_id"].map(difficulty_lookup).fillna("unknown")
    kernel_counts = enriched.groupby("Difficulty")["kernel_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    for difficulty in sorted(enriched["Difficulty"].unique()):
        row: dict[str, Any] = {"Difficulty": difficulty, "N kernels": kernel_counts.get(difficulty, 0)}
        subset = enriched[enriched["Difficulty"] == difficulty]
        for model in sorted(enriched["model"].unique()):
            model_subset = subset[subset["model"] == model]
            row[f"{model} accuracy"] = round(100.0 * float(model_subset["correct"].mean()), 2) if not model_subset.empty else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_per_class_accuracy(label_df: pd.DataFrame, registry: KernelRegistry, output_path: Path) -> pd.DataFrame:
    if label_df.empty:
        table = pd.DataFrame(columns=["Model"])
        output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
        return table
    bottleneck_lookup = {entry.id: entry.true_bottleneck for entry in registry}
    enriched = label_df.copy()
    enriched["true_bottleneck"] = enriched["kernel_id"].map(bottleneck_lookup).fillna(enriched["true_bottleneck"])
    rows: list[dict[str, Any]] = []
    ordered_labels = [
        "memory-bound",
        "latency-bound",
        "compute-bound",
        "occupancy-limited",
        "register-spill",
    ]
    for (model, level), subset in enriched.groupby(["model", "level"], dropna=False):
        row: dict[str, Any] = {"Model": f"{model} L{int(level)}"}
        for label in ordered_labels:
            label_subset = subset[subset["true_bottleneck"] == label]
            total = len(label_subset)
            accuracy = float(label_subset["correct"].mean()) if total else 0.0
            row[label] = f"{accuracy * 100.0:.1f}% ({total})"
        rows.append(row)
    table = pd.DataFrame(rows)
    output_path.write_text(frame_to_markdown(table) + "\n", encoding="utf-8")
    return table


def write_summary(df: pd.DataFrame, output_path: Path, consistency_table: pd.DataFrame | None = None) -> None:
    if df.empty:
        output_path.write_text("No results available.\n", encoding="utf-8")
        return
    model_acc = df.groupby("model")["correct"].mean().sort_values(ascending=False)
    best_model = model_acc.index[0]
    best_accuracy = model_acc.iloc[0] * 100.0
    l1 = df[df["level"] == 1].groupby("model")["correct"].mean()
    l3 = df[df["level"] == 3].groupby("model")["correct"].mean()
    improvement = (l3 - l1).fillna(0.0).sort_values(ascending=False)
    most_improved = improvement.index[0]
    most_improved_delta = improvement.iloc[0] * 100.0
    adversarial = (
        df[df["level"] == 4]
        .assign(fell_for_adversarial=df[df["level"] == 4]["fell_for_adversarial"].astype("boolean").fillna(False).astype(bool))
        .groupby("model")["fell_for_adversarial"]
        .mean()
        .fillna(0.0)
        .sort_values(ascending=False)
    )
    most_sycophantic = adversarial.index[0]
    most_sycophantic_rate = adversarial.iloc[0] * 100.0
    summary = (
        f"1. {best_model} achieved the highest aggregate accuracy at {best_accuracy:.2f}%.\n"
        f"2. {most_improved} improved the most from level 1 to level 3, gaining {most_improved_delta:.2f} points when profiler evidence was added.\n"
        f"3. {most_sycophantic} was most vulnerable to adversarial framing at level 4, agreeing with the wrong framing {most_sycophantic_rate:.2f}% of the time.\n"
    )
    summary += (
        "\nStatistical Notes:\n"
        "- All CIs computed using Wilson score interval.\n"
        "- Results with CI width > 30pp are flagged as underpowered.\n"
        "- Minimum recommended corpus size for 20pp CI: 96 kernels.\n"
        "- Groundedness reports correct answers whose reasoning cites expected profiler evidence and avoids the primary red herring.\n"
        "- Judge-based reasoning quality scores are produced by an auxiliary LLM judge when available.\n"
    )
    if consistency_table is not None and not consistency_table.empty:
        summary += "\nConsistency Scores\n\n" + frame_to_markdown(consistency_table) + "\n"
    output_path.write_text(summary, encoding="utf-8")


def export_scores_csv(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    rows: list[pd.DataFrame] = []
    for name, table in tables.items():
        if table.empty:
            continue
        temp = table.copy()
        temp.insert(0, "table", name)
        rows.append(temp)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output.to_csv(output_dir / "scores.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    registry = load_registry(
        args.mined,
        args.kernelbot,
        args.kernelbench_compute,
        args.latency,
        args.register_spill,
        args.kernels,
        args.profiles,
        args.min_confidence,
    )
    filtered_entries = registry.filter(confidence=args.min_confidence)
    subset_ids = load_subset_ids(args.subset)
    if subset_ids:
        filtered_entries = [entry for entry in filtered_entries if entry.id in subset_ids]
    allowed_ids = {entry.id for entry in filtered_entries}
    rows = latest_per_combo(load_results(args.results))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["question_format"] = df.get("question_format", "label").fillna("label")
        df = df[df["kernel_id"].isin(allowed_ids)]
    label_df = df[(df["question_format"] == "label")] if not df.empty else df
    if not df.empty:
        label_df = label_df.copy()
        if not label_df.empty:
            label_df["correct"] = label_df["correct"].astype(bool)
            label_df["level"] = label_df["level"].astype(int)
            if "correct_reasoning" not in label_df.columns:
                label_df["correct_reasoning"] = None
    accuracy = accuracy_table(label_df, filtered_entries)
    (args.results / "accuracy_table.md").write_text(frame_to_markdown(accuracy) + "\n", encoding="utf-8")
    write_confusion_matrices(label_df, args.results / "confusion_matrices")
    level_sensitivity = write_level_sensitivity(label_df, args.results / "level_sensitivity.md")
    sycophancy = write_sycophancy(label_df, args.results / "sycophancy_scores.md")
    category_breakdown = write_category_breakdown(label_df, filtered_entries, args.results / "category_breakdown.md")
    reasoning_quality = write_reasoning_quality(label_df, args.results / "reasoning_quality.md")
    groundedness = write_groundedness(label_df, registry, args.results / "groundedness_table.md")
    judge_reasoning_quality = write_judge_reasoning_quality(label_df, args.results / "reasoning_quality_table.md")
    contamination_rows = load_contamination_results(args.results)
    contamination_table = write_contamination_report(label_df, contamination_rows, args.results / "contamination_table.md")
    difficulty_calibration = write_difficulty_calibration(label_df, filtered_entries, args.results / "difficulty_calibration_table.md")
    per_class_accuracy = write_per_class_accuracy(label_df, filtered_entries, args.results / "per_class_accuracy.md")
    consistency_detail = pd.DataFrame()
    consistency_summary = pd.DataFrame()
    if args.include_formats and not df.empty:
        consistency_detail, consistency_summary = compute_consistency_scores(df)
        (args.results / "consistency_scores.md").write_text(frame_to_markdown(consistency_summary) + "\n", encoding="utf-8")
    write_summary(label_df, args.results / "summary.md", consistency_summary if args.include_formats else None)
    export_scores_csv(
        args.results,
        {
            "accuracy_table": accuracy,
            "level_sensitivity": level_sensitivity,
            "sycophancy": sycophancy,
            "category_breakdown": category_breakdown,
            "reasoning_quality": reasoning_quality,
            "reasoning_quality_table": judge_reasoning_quality,
            "groundedness": groundedness,
            "contamination": contamination_table,
            "difficulty_calibration": difficulty_calibration,
            "per_class_accuracy": per_class_accuracy,
            "consistency_scores": consistency_summary,
            "consistency_detail": consistency_detail,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
