#!/usr/bin/env python3
"""Render Moebius benchmark TSV results as PNG charts and a markdown summary."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIDELITY_COLUMNS = ["max_abs", "mean_abs", "rmse", "p99_abs"]
SPEED_COLUMNS = ["inference_seconds", "load_seconds", "total_seconds"]


def parse_float(value: str) -> float:
    value = value.strip()
    if not value or value == "-":
        return math.nan
    return float(value)


def read_tsv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {"variant": row["variant"]}
            for key, value in row.items():
                if key == "variant":
                    continue
                parsed[key] = parse_float(value)
            rows.append(parsed)
        return rows


def variants(rows: list[dict[str, float | str]]) -> list[str]:
    return [str(row["variant"]) for row in rows]


def values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row.get(key, math.nan)) for row in rows]


def save_bar_chart(
    rows: list[dict[str, float | str]],
    key: str,
    title: str,
    ylabel: str,
    output: Path,
    target: float | None = None,
) -> None:
    names = variants(rows)
    data = values(rows, key)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(names, data, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    if target is not None:
        ax.axhline(target, color="#dc2626", linestyle="--", linewidth=1.5)
        ax.text(len(names) - 0.5, target, f" {target:g} MB target", color="#dc2626", va="bottom")
    for bar, value in zip(bars, data):
        if math.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_grouped_fidelity_chart(rows: list[dict[str, float | str]], output: Path, title: str, ylabel: str) -> None:
    names = variants(rows)
    x = list(range(len(names)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6.2))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
    for index, column in enumerate(FIDELITY_COLUMNS):
        offsets = [position + (index - 1.5) * width for position in x]
        ax.bar(offsets, values(rows, column), width=width, label=column, color=colors[index])
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_speed_chart(rows: list[dict[str, float | str]], output: Path) -> None:
    names = variants(rows)
    x = list(range(len(names)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2563eb", "#f59e0b", "#64748b"]
    labels = ["inference", "load", "total"]
    for index, (column, label) in enumerate(zip(SPEED_COLUMNS, labels)):
        offsets = [position + (index - 1) * width for position in x]
        ax.bar(offsets, values(rows, column), width=width, label=label, color=colors[index])
    ax.set_title("Performance Timings")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_tradeoff_chart(rows: list[dict[str, float | str]], output: Path, x_key: str, y_key: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for row in rows:
        x = float(row.get(x_key, math.nan))
        y = float(row.get(y_key, math.nan))
        size = float(row.get("model_mb", 1.0))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        ax.scatter(x, y, s=max(45, min(300, size / 3)), alpha=0.75, color="#2563eb")
        ax.annotate(str(row["variant"]).replace("ft_places2-", ""), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE vs reference")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(rows: list[dict[str, float | str]], output_dir: Path, source: Path, title: str) -> None:
    best_speed = min(rows, key=lambda row: float(row.get("inference_seconds", math.inf)))
    best_rmse = min(rows, key=lambda row: float(row.get("rmse", math.inf)))
    smallest = min(rows, key=lambda row: float(row.get("model_mb", math.inf)))
    lines = [
        f"# {title}",
        "",
        f"Source TSV: `{source}`",
        "",
        "| variant | model MB | max abs | mean abs | RMSE | p99 abs | PSNR dB | cosine | inference s | load s | total s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {model_mb:.2f} | {max_abs:.6g} | {mean_abs:.6g} | {rmse:.6g} | {p99_abs:.6g} | {psnr_db:.3f} | {cosine:.8f} | {inference_seconds:.3f} | {load_seconds:.3f} | {total_seconds:.3f} |".format(
                variant=row["variant"],
                model_mb=float(row["model_mb"]),
                max_abs=float(row["max_abs"]),
                mean_abs=float(row["mean_abs"]),
                rmse=float(row["rmse"]),
                p99_abs=float(row["p99_abs"]),
                psnr_db=float(row["psnr_db"]),
                cosine=float(row["cosine"]),
                inference_seconds=float(row["inference_seconds"]),
                load_seconds=float(row["load_seconds"]),
                total_seconds=float(row["total_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            f"- Lowest RMSE: `{best_rmse['variant']}` at `{float(best_rmse['rmse']):.6g}`.",
            f"- Fastest inference: `{best_speed['variant']}` at `{float(best_speed['inference_seconds']):.3f}s`.",
            f"- Smallest package: `{smallest['variant']}` at `{float(smallest['model_mb']):.2f} MB`.",
            "",
            "Generated charts:",
            "",
            "- `fidelity_errors.png`",
            "- `performance_timings.png`",
            "- `model_size.png`",
            "- `speed_fidelity_tradeoff.png`",
            "- `size_fidelity_tradeoff.png`",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Benchmark TSV from Scripts/benchmark_fidelity.sh.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Moebius MLX Benchmark Report")
    parser.add_argument("--fidelity-title", default="Fidelity Error Metrics")
    parser.add_argument("--fidelity-ylabel", default="absolute error, log scale")
    args = parser.parse_args()

    rows = read_tsv(args.input)
    if not rows:
        raise SystemExit("No benchmark rows found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_grouped_fidelity_chart(rows, args.output_dir / "fidelity_errors.png", args.fidelity_title, args.fidelity_ylabel)
    save_speed_chart(rows, args.output_dir / "performance_timings.png")
    save_bar_chart(rows, "model_mb", "Model Package Size", "MB", args.output_dir / "model_size.png", target=200)
    save_tradeoff_chart(
        rows,
        args.output_dir / "speed_fidelity_tradeoff.png",
        x_key="inference_seconds",
        y_key="rmse",
        title="Speed vs Fidelity",
        xlabel="inference seconds",
    )
    save_tradeoff_chart(
        rows,
        args.output_dir / "size_fidelity_tradeoff.png",
        x_key="model_mb",
        y_key="rmse",
        title="Size vs Fidelity",
        xlabel="model package MB",
    )
    write_summary(rows, args.output_dir, args.input, args.title)
    print(f"wrote benchmark report: {args.output_dir}")


if __name__ == "__main__":
    main()
