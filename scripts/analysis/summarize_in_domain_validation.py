#!/usr/bin/env python3
"""Summarize completed 7B confirmatory XLCoST validation histories.

This is a descriptive, in-domain-only report. It selects each run's epoch with
the lowest validation target loss, matching the frozen checkpoint-selection
rule. It does not load external manifests or compute inferential statistics.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments" / "registry.csv"
OUTPUT = ROOT / "reports" / "progress" / "in_domain_validation_screen.md"
BACKBONE = "Qwen/Qwen2.5-Coder-7B"
HORIZON = "shared-2-epoch-7B-horizon"
FIELDS = (
    "validation_target_loss",
    "validation_bleu4_macro",
    "validation_codebleu_macro",
    "validation_crystalbleu_macro",
    "validation_exact_match_macro",
    "mean_recurrent_depth",
    "stop_after_first_fraction",
    "reach_rmax_fraction",
    "router_entropy",
)


def fmt(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def main() -> None:
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trained = [
        row
        for row in rows
        if row["training_status"] == "TRAINED"
        and row["backbone"] == BACKBONE
        and row["training_horizon_policy"] == HORIZON
    ]
    if not trained:
        raise RuntimeError("No trained 7B confirmatory rows found in registry")

    selected: list[dict[str, object]] = []
    for row in trained:
        metrics_path = ROOT / row["metrics_path"]
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics for {row['experiment_id']}: {metrics_path}")
        history = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not history:
            raise RuntimeError(f"Empty validation history for {row['experiment_id']}")
        if max(int(item["epoch"]) for item in history) < 2:
            raise RuntimeError(f"TRAINED row has no epoch-2 validation: {row['experiment_id']}")
        best = min(history, key=lambda item: float(item["validation_target_loss"]))
        selected.append(
            {
                "experiment_id": row["experiment_id"],
                "architecture": row["architecture"],
                "seed": row["seed"],
                "best_epoch": int(best["epoch"]),
                **{field: best.get(field) for field in FIELDS},
            }
        )

    by_arch: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in selected:
        by_arch[str(item["architecture"])].append(item)

    lines = [
        "# In-domain validation screen",
        "",
        "Generated from the canonical registry and per-run `metrics.jsonl` histories. For each trained Qwen2.5-Coder-7B confirmatory run, the displayed epoch is selected by minimum XLCoST validation target loss, following the frozen checkpoint-selection rule.",
        "",
        "**Scope:** descriptive XLCoST validation only. These values are not external benchmark results, do not establish functional correctness, and are not inferential comparisons. Runs with incomplete seed replication are shown individually and must not be treated as balanced architecture means. The report does not read external manifests or predictions.",
        "",
        f"Canonical trained runs included: **{len(selected)}**. Each selected epoch is within the fixed two-epoch horizon.",
        "",
        "## Architecture-level descriptive means",
        "",
        "Means use only currently TRAINED seeds; the `n/seeds` column makes incomplete replication visible. Do not compare these partial means as confirmatory effects.",
        "",
        "| Architecture | n / seeds | Best epoch mean | Val loss | BLEU-4 | CodeBLEU | CrystalBLEU | Exact match | Mean depth | Stop at 1 | Reach Rmax | Router entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in sorted(by_arch):
        group = by_arch[architecture]
        seeds = ",".join(sorted({str(item["seed"]) for item in group}, key=int))
        epoch_mean = mean(int(item["best_epoch"]) for item in group)
        values = [fmt(mean(float(item[field]) for item in group if item[field] is not None)) for field in FIELDS]
        lines.append(
            f"| {architecture} | {len(group)} / {seeds} | {epoch_mean:.2f} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-run selected validation rows",
            "",
            "| Experiment | Seed | Best epoch | Val loss | BLEU-4 | CodeBLEU | CrystalBLEU | Exact match | Mean depth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(selected, key=lambda x: (str(x["architecture"]), int(str(x["seed"])) )):
        lines.append(
            f"| `{item['experiment_id']}` | {item['seed']} | {item['best_epoch']} | "
            + " | ".join(fmt(item.get(field)) for field in FIELDS[:6])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Validation lexical metrics are diagnostics; validation target loss alone selects checkpoints.",
            "- External execution Pass@1, paired uncertainty estimates, and matched end-to-end compute costs are reported separately in the final research report.",
            "- The report summarizes the 40 canonical Qwen2.5-Coder-7B runs. The authoritative registry and per-run histories remain the source data.",
            "",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} from {len(selected)} trained canonical runs")


if __name__ == "__main__":
    main()
