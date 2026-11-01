#!/usr/bin/env python3
"""Build a complete, reproducible summary from frozen external score manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.dispatch_external_predictions import model_runs, output_root as prediction_root, task_list  # noqa: E402
from scripts.evaluate.dispatch_external_scoring import MANIFESTS, result_root  # noqa: E402
REGISTRY = ROOT / "experiments/registry.csv"
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
BENCHMARKS = ("xcodeeval", "transcoder_test", "humanevalx", "avatar_tc")
SPLITS = ("common_core", "full_directional")
METRICS = ("pass_at_1",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direction_rows(value: dict[str, Any], key: str) -> dict[str, Any]:
    rows = value.get("directional_metrics")
    if not isinstance(rows, dict) or len(rows) != 6:
        raise ValueError(f"Expected six directional metrics in {key}")
    for direction, metrics in rows.items():
        if not isinstance(metrics, dict) or metrics.get("tasks", 0) < 1:
            raise ValueError(f"Invalid direction row {direction} in {key}")
        for metric in METRICS:
            score = metrics.get(metric)
            if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
                raise ValueError(f"Invalid {metric} for {direction} in {key}: {score!r}")
    return rows


def summarize() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREMODEL":
        raise ValueError("External results summary requires the frozen pre-model protocol")
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        runs = model_runs(list(csv.DictReader(stream)))
    if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
        raise ValueError("External results summary requires all 55 canonical runs TRAINED")

    tasks = task_list(runs, protocol)
    if len(tasks) != 440:
        raise ValueError(f"Expected exactly 440 frozen benchmark/split runs, found {len(tasks)}")
    run_by_id = {row["experiment_id"]: row for row in runs}
    detail_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for task in tasks:
        experiment_id = task["experiment_id"]
        pred_root = prediction_root(
            task["revision"], experiment_id, task["benchmark"], task["split"]
        )
        pred_path = pred_root / "run_manifest.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction manifest: {pred_path}")
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        model_metadata = pred.get("model", {})
        if (
            pred.get("status") != "COMPLETE"
            or pred.get("tasks_total", 0) < 1
            or pred.get("tasks_completed") != pred.get("tasks_total")
            or pred.get("generation_errors") != 0
            or pred.get("benchmark") != task["benchmark"]
            or pred.get("split") != task["split"]
            or pred.get("frozen_protocol_sha256") != sha256(PROTOCOL)
            or model_metadata.get("experiment_id") != experiment_id
            or model_metadata.get("architecture") != task["architecture"]
            or Path(str(model_metadata.get("backbone_path", ""))).name != task["revision"]
        ):
            raise ValueError(f"Prediction manifest is incomplete or mismatched: {pred_path}")

        score_path = result_root(
            task["revision"], experiment_id, task["benchmark"], task["split"]
        ) / "score_manifest.json"
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing score manifest: {score_path}")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        manifest_path = ROOT / MANIFESTS[(task["benchmark"], task["split"])]
        if (
            score.get("status") != "COMPLETE"
            or score.get("benchmark") != task["benchmark"]
            or score.get("split") != task["split"]
            or score.get("tasks_total") != pred["tasks_total"]
            or score.get("tasks_completed") != score.get("tasks_total")
            or score.get("eligibility_manifest_sha256") != sha256(manifest_path)
            or score.get("prediction_run_manifest_sha256") != sha256(pred_path)
            or score.get("candidate_scoring_version") != "pass_at_1_single_sample_v1"
        ):
            raise ValueError(f"Score manifest is incomplete or mismatched: {score_path}")
        status_counts = score.get("status_counts")
        if (
            not isinstance(status_counts, dict)
            or sum(status_counts.values()) != score["tasks_total"]
            or any(name in status_counts for name in ("runner_error", "runner_or_setup_error"))
        ):
            raise ValueError(f"Score manifest has missing tasks or infrastructure errors: {score_path}")

        directions = _direction_rows(score, str(score_path))
        if sum(int(row["tasks"]) for row in directions.values()) != score["tasks_total"]:
            raise ValueError(f"Directional task counts do not sum to the run total: {score_path}")
        metrics = {metric: score.get(f"{metric}_macro") for metric in METRICS}
        for metric, value in metrics.items():
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f"Invalid macro {metric} in {score_path}: {value!r}")
        row = {
            "experiment_id": experiment_id,
            "backbone": task["backbone"],
            "backbone_revision": task["revision"],
            "architecture": task["architecture"],
            "seed": int(task["seed"]),
            "benchmark": task["benchmark"],
            "split": task["split"],
            "tasks": int(score["tasks_total"]),
            **metrics,
            "directional_metrics": directions,
            "score_manifest": str(score_path.relative_to(ROOT)),
            "score_manifest_sha256": sha256(score_path),
            "prediction_manifest_sha256": sha256(pred_path),
        }
        detail_rows.append(row)
        grouped[(task["revision"], task["architecture"], task["benchmark"], task["split"])].append(row)

    groups = []
    for (revision, architecture, benchmark, split), rows in sorted(grouped.items()):
        directions = sorted(rows[0]["directional_metrics"])
        summary: dict[str, Any] = {
            "backbone_revision": revision,
            "architecture": architecture,
            "benchmark": benchmark,
            "split": split,
            "seed_count": len(rows),
            "seeds": sorted(row["seed"] for row in rows),
            "experiment_ids": sorted(row["experiment_id"] for row in rows),
            "direction_order": directions,
            "metrics": {},
        }
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary["metrics"][metric] = {
                "mean_over_seeds": statistics.mean(values) if values else None,
                "available_seed_count": len(values),
            }
            summary["metrics"][metric]["direction_means_over_seeds"] = {
                direction: (
                    statistics.mean([
                        float(row["directional_metrics"][direction][metric])
                        for row in rows
                        if row["directional_metrics"][direction].get(metric) is not None
                    ])
                    if any(row["directional_metrics"][direction].get(metric) is not None for row in rows)
                    else None
                )
                for direction in directions
            }
        groups.append(summary)

    return {
        "status": "COMPLETE",
        "protocol_version": protocol.get("protocol_version"),
        "protocol_sha256": sha256(PROTOCOL),
        "canonical_model_runs": len(runs),
        "prediction_score_runs": len(detail_rows),
        "group_count": len(groups),
        "primary_metric_policy": "one seeded stochastic candidate per task; execution Pass@1",
        "macro_policy": "equal weight over all six translation directions; seed groups summarized as arithmetic means",
        "detail_rows": detail_rows,
        "groups": groups,
    }


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External benchmark results summary",
        "",
        f"Protocol v{report['protocol_version']} (`{report['protocol_sha256']}`); "
        f"{report['prediction_score_runs']}/440 prediction and score runs reconciled.",
        "Pass@1 uses one seeded stochastic candidate per task. "
        "Macro scores weight the six directions equally; architecture rows summarize the recorded seeds.",
        "Per-run and per-direction values, score-manifest hashes, and experiment IDs are in the companion JSON.",
        "",
    ]
    for benchmark in BENCHMARKS:
        for split in SPLITS:
            selected = [
                row for row in report["detail_rows"]
                if row["benchmark"] == benchmark and row["split"] == split
            ]
            lines.extend([
                f"## {benchmark} — {split}",
                "",
                "| Revision | Architecture | Seed | Tasks | Pass@1 macro | Pass@1 by direction |",
                "|---|---|---:|---:|---:|---|",
            ])
            for row in sorted(selected, key=lambda x: (x["backbone_revision"], x["architecture"], x["seed"])):
                p1 = row["directional_metrics"]
                p1_text = "; ".join(f"{d} {_fmt(p1[d].get('pass_at_1'))}" for d in sorted(p1))
                lines.append(
                    f"| `{row['backbone_revision']}` | {row['architecture']} | {row['seed']} | {row['tasks']} "
                    f"| {_fmt(row['pass_at_1'])} | {p1_text} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=Path("reports/analysis/external_benchmark_summary.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/analysis/external_benchmark_summary.md"))
    args = parser.parse_args()
    report = summarize()
    for path, content in (
        (args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (args.markdown_output, render_markdown(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": report["status"], "score_runs": report["prediction_score_runs"], "groups": report["group_count"]}))


if __name__ == "__main__":
    main()
