#!/usr/bin/env python3
"""Publish an honest seed-42 accuracy preview after four complete score runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.dispatch_external_predictions import (  # noqa: E402
    model_runs,
    output_root as prediction_root,
    task_list,
)
from scripts.evaluate.dispatch_external_scoring import MANIFESTS, result_root  # noqa: E402
from src.statistics.paired import build_outcome_tensor, hierarchical_bootstrap  # noqa: E402
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
REGISTRY = ROOT / "experiments/registry.csv"
JSON_OUTPUT = ROOT / "reports/analysis/xcodeeval_common_core_initial_seed42.json"
MARKDOWN_OUTPUT = ROOT / "reports/analysis/xcodeeval_common_core_initial_seed42.md"
PRIMARY_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
ARCHITECTURES = ("full_sart", "dense", "mor_style", "uniform_token_loop")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize() -> dict[str, Any] | None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREMODEL" or protocol.get("protocol_version") != 3:
        raise ValueError("Preliminary summary requires frozen Pass@1 Protocol v3")
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        runs = model_runs(list(csv.DictReader(stream)))
    if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
        raise ValueError("Preliminary summary requires the complete 55-model registry")
    selected = [task for task in task_list(runs, protocol)
                if task["revision"] == PRIMARY_REVISION
                and task["benchmark"] == "xcodeeval"
                and task["split"] == "common_core"
                and task["seed"] == 42
                and task["architecture"] in ARCHITECTURES]
    if len(selected) != 4 or {task["architecture"] for task in selected} != set(ARCHITECTURES):
        raise ValueError("Expected the four prioritized seed-42 model runs")
    task_rows = []
    manifests = {}
    for task in selected:
        root = result_root(task["revision"], task["experiment_id"],
                           task["benchmark"], task["split"])
        manifest_path = root / "score_manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE":
            return None
        pred_path = prediction_root(
            task["revision"], task["experiment_id"], task["benchmark"], task["split"]
        ) / "run_manifest.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Initial prediction manifest is absent: {pred_path}")
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        eligibility = ROOT / MANIFESTS[(task["benchmark"], task["split"])]
        if (
            manifest.get("tasks_total") != 1992
            or manifest.get("tasks_completed") != 1992
            or manifest.get("candidate_scoring_version") != "pass_at_1_single_sample_v1"
            or len(manifest.get("directional_metrics", {})) != 6
            or sum(int(value.get("tasks", 0)) for value in manifest["directional_metrics"].values()) != 1992
            or manifest.get("prediction_run_manifest_sha256") != sha256(pred_path)
            or manifest.get("eligibility_manifest_sha256") != sha256(eligibility)
            or pred.get("status") != "COMPLETE"
            or pred.get("frozen_protocol_sha256") != sha256(PROTOCOL)
            or pred.get("tasks_total") != 1992
            or pred.get("tasks_completed") != 1992
            or pred.get("generation_errors", 0) != 0
            or manifest.get("status_counts", {}).get("runner_error", 0) != 0
            or manifest.get("status_counts", {}).get("runner_or_setup_error", 0) != 0
        ):
            raise ValueError(f"Incomplete or mismatched initial score: {manifest_path}")
        rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in root.rglob("*.json") if path.name != "score_manifest.json"]
        if len(rows) != 1992 or len({row["task_id"] for row in rows}) != 1992:
            raise ValueError(f"Initial task-score identities are incomplete: {root}")
        if any(row.get("experiment_id") != task["experiment_id"]
               or row.get("benchmark") != "xcodeeval"
               or row.get("split") != "common_core" for row in rows):
            raise ValueError(f"Initial task-score metadata mismatch: {root}")
        task_rows.extend(rows)
        manifests[task["architecture"]] = {
            "experiment_id": task["experiment_id"],
            "score_manifest": str(manifest_path.relative_to(ROOT)),
            "score_manifest_sha256": sha256(manifest_path),
            "pass_at_1_macro": manifest["pass_at_1_macro"],
            "directional_metrics": manifest["directional_metrics"],
            "status_counts": manifest.get("status_counts", {}),
        }
    comparisons = {}
    for baseline in ARCHITECTURES[1:]:
        tensor, problems, directions, seeds = build_outcome_tensor(
            task_rows, "full_sart", baseline,
            benchmark="xcodeeval", split="common_core", metric="pass_at_1",
        )
        if seeds != [42] or len(problems) != 332 or len(directions) != 6:
            raise ValueError("Initial paired comparison lacks the frozen seed/problem matrix")
        bootstrap = hierarchical_bootstrap(tensor, replicates=10_000)
        comparisons[f"full_sart_vs_{baseline}"] = {
            "baseline": baseline,
            "problem_count": len(problems),
            "seed": 42,
            "pass_at_1_difference_percentage_points": bootstrap["difference_percentage_points"],
            "exploratory_problem_bootstrap_95_percent_interval_percentage_points":
                bootstrap["ci95_percentage_points"],
        }
    return {
        "status": "preliminary_seed42_only",
        "protocol_sha256": sha256(PROTOCOL),
        "benchmark": "xcodeeval", "split": "common_core",
        "backbone_revision": PRIMARY_REVISION,
        "models": manifests,
        "comparisons": comparisons,
        "interpretation_limit": "Single seed and no matched end-to-end compute profile yet; this preview cannot confirm H1 or H2. Three-seed paired inference, all planned controls, and efficiency profiles remain required.",
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Initial xCodeEval Common-Core results — seed 42",
        "",
        "**Preliminary accuracy preview.** One 7B seed across 332 problems and six translation directions. Pass@1 uses one seeded stochastic candidate per task under frozen Protocol v3. These results cannot confirm the matched-compute hypothesis; the three-seed analysis and inference profiles are still required.",
        "",
        "| Architecture | Pass@1 macro | Runner/setup errors |",
        "|---|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        model = report["models"][architecture]
        lines.append(f"| {architecture} | {model['pass_at_1_macro']:.4f} | "
                     f"{model['status_counts'].get('runner_or_setup_error', 0)} |")
    lines.extend(["", "| Full-SART contrast | Difference (percentage points) | Exploratory 95% problem bootstrap interval |",
                  "|---|---:|---:|"])
    for name, comparison in report["comparisons"].items():
        low, high = comparison["exploratory_problem_bootstrap_95_percent_interval_percentage_points"]
        lines.append(f"| {name} | {comparison['pass_at_1_difference_percentage_points']:+.2f} | "
                     f"[{low:+.2f}, {high:+.2f}] |")
    lines.extend(["", report["interpretation_limit"],
                  "", "Full per-direction rates, score manifest hashes, and experiment IDs are in the companion JSON.", ""])
    return "\n".join(lines)


def main() -> None:
    if JSON_OUTPUT.is_file() and MARKDOWN_OUTPUT.is_file():
        try:
            existing = json.loads(JSON_OUTPUT.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "preliminary_seed42_only"
                and existing.get("protocol_sha256") == sha256(PROTOCOL)
                and all(
                    sha256(ROOT / model["score_manifest"]) == model["score_manifest_sha256"]
                    for model in existing.get("models", {}).values()
                )
                and len(existing.get("models", {})) == 4
            ):
                return
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    result = summarize()
    if result is None:
        print("Initial four score runs are not complete")
        return
    atomic_text(JSON_OUTPUT, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(MARKDOWN_OUTPUT, render(result))
    print(f"Wrote {MARKDOWN_OUTPUT}")


if __name__ == "__main__":
    main()
