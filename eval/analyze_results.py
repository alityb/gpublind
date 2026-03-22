from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from eval.baselines import frequency_baseline, random_baseline, roofline_baseline
from registry import KernelRegistry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GPUBlind results")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--mined", type=Path, default=Path("data/mined_kernels.jsonl"))
    parser.add_argument("--kernels", type=Path, default=Path("kernels"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles"))
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


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/level_*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)
    return rows


def load_registry(mined: Path, kernels: Path, profiles: Path) -> KernelRegistry:
    registry = KernelRegistry(profile_dir=profiles, mock=True)
    registry.load_mined(mined)
    registry.load_handwritten(kernels)
    return registry


def latest_per_combo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["level"]), str(row["kernel_id"]))
        latest[key] = row
    return list(latest.values())


def accuracy_table(df: pd.DataFrame, registry: KernelRegistry) -> pd.DataFrame:
    models = sorted(df["model"].unique()) if not df.empty else []
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"Model": model}
        for level in range(1, 6):
            subset = df[(df["model"] == model) & (df["level"] == level)]
            row[f"L{level}"] = round(100.0 * subset["correct"].mean(), 2) if not subset.empty else None
        rows.append(row)
    random_row = {"Model": "Random"}
    random_stats = random_baseline(list(registry))
    for level in range(1, 6):
        random_row[f"L{level}"] = round(100.0 * random_stats["mean"], 2)
    rows.append(random_row)
    frequency = frequency_baseline(list(registry))
    frequency_row = {"Model": "Frequency"}
    for level in range(1, 6):
        frequency_row[f"L{level}"] = round(100.0 * float(frequency["accuracy"]), 2)
    rows.append(frequency_row)
    roofline = roofline_baseline(list(registry))
    roofline_row = {"Model": "Roofline"}
    for level in range(1, 6):
        roofline_row[f"L{level}"] = round(100.0 * float(roofline["accuracy"]), 2)
    rows.append(roofline_row)
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


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
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
    registry = load_registry(args.mined, args.kernels, args.profiles)
    rows = latest_per_combo(load_results(args.results))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["correct"] = df["correct"].astype(bool)
        df["level"] = df["level"].astype(int)
        if "correct_reasoning" not in df.columns:
            df["correct_reasoning"] = None
    accuracy = accuracy_table(df, registry)
    (args.results / "accuracy_table.md").write_text(frame_to_markdown(accuracy) + "\n", encoding="utf-8")
    write_confusion_matrices(df, args.results / "confusion_matrices")
    level_sensitivity = write_level_sensitivity(df, args.results / "level_sensitivity.md")
    sycophancy = write_sycophancy(df, args.results / "sycophancy_scores.md")
    category_breakdown = write_category_breakdown(df, registry, args.results / "category_breakdown.md")
    reasoning_quality = write_reasoning_quality(df, args.results / "reasoning_quality.md")
    write_summary(df, args.results / "summary.md")
    export_scores_csv(
        args.results,
        {
            "accuracy_table": accuracy,
            "level_sensitivity": level_sensitivity,
            "sycophancy": sycophancy,
            "category_breakdown": category_breakdown,
            "reasoning_quality": reasoning_quality,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
