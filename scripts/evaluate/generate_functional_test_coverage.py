#!/usr/bin/env python3
"""Generate functional test coverage counts from frozen pre-model manifests."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data/manifests"
OUTPUT = ROOT / "reports/data/functional_test_coverage.md"

SOURCES = {
    "xCodeEval": [("Common-Core", "xcodeeval_common_core_manifest.json", None),
                  ("Full-Directional", "xcodeeval_full_direction_manifest.json", None)],
    "TransCoder-Test": [("Common-Core", "transcoder_test_manifest.json", "common_core"),
                        ("Full-Directional", "transcoder_test_manifest.json", "full_directional")],
    "HumanEval-X": [("Common-Core", "humanevalx_manifest.json", "common_core"),
                    ("Full-Directional", "humanevalx_manifest.json", "full_directional")],
    "AVATAR-TC": [("Common-Core", "avatar_tc_manifest.json", "common_core"),
                  ("Full-Directional", "avatar_tc_manifest.json", "full_directional")],
}


def _direction_data(data: dict[str, Any], split: str | None) -> dict[str, Any]:
    section = data if split is None else data[split]
    return section.get("directions", section)


def _task_problem(task: dict[str, Any]) -> str:
    return str(task.get("src_uid", task.get("problem_id", "UNKNOWN")))


def _task_tests(task: dict[str, Any]) -> int:
    for key in ("test_count", "target_test_count"):
        if key in task:
            return int(task[key])
    raise KeyError(f"No test count in task {task.keys()}")


def _render() -> str:
    lines = [
        "# Functional test coverage",
        "",
        "Counts below are computed from the frozen pre-model eligibility manifests. A translation task is one selected source implementation in one directed language pair. `Test inputs across tasks` sums executable inputs over those translation tasks; when one problem is translated in multiple directions, its tests are counted once for each translation task. `Unique problems` is deduplicated by benchmark problem ID. The common-core and full-directional sets are reported separately.",
        "",
        "## Aggregate coverage",
        "",
        "| Benchmark | Set | Unique problems | Translation tasks | Directions | Test inputs across tasks | Mean / task | Median / task | Min / task | Max / task |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details: list[str] = []
    for benchmark, specs in SOURCES.items():
        for set_name, filename, split in specs:
            path = MANIFESTS / filename
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            directions = _direction_data(data, split)
            direction_rows = []
            task_counts: list[int] = []
            problem_ids: set[str] = set()
            task_total = 0
            for direction, record in directions.items():
                if not isinstance(record, dict) or "tasks" not in record:
                    continue
                tasks = record["tasks"]
                counts = [_task_tests(task) for task in tasks]
                task_counts.extend(counts)
                task_total += len(tasks)
                problem_ids.update(_task_problem(task) for task in tasks)
                direction_rows.append((direction, len(tasks), sum(counts), counts))
            if not task_counts:
                continue
            lines.append(
                f"| {benchmark} | {set_name} | {len(problem_ids)} | {task_total} | {len(direction_rows)} | {sum(task_counts)} | {statistics.mean(task_counts):.2f} | {statistics.median(task_counts):g} | {min(task_counts)} | {max(task_counts)} |"
            )
            details.extend([
                f"## {benchmark} — {set_name}",
                "",
                f"Manifest: `{filename}`; status: `{data.get('status', 'not recorded')}`.",
                "",
                "| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for direction, n, total, counts in direction_rows:
                details.append(
                    f"| {direction} | {n} | {total} | {statistics.mean(counts):.2f} | {statistics.median(counts):g} | {min(counts)} | {max(counts)} |"
                )
            details.extend(["", f"Test inputs summed across tasks: **{sum(task_counts):,}**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.", ""])
    lines.extend(["", "## Per-direction counts", ""])
    lines.extend(details)
    lines.extend([
        "## Pre-model statistical precision",
        "",
        "These conservative planning estimates were frozen from eligible problem counts before model outcomes were inspected. They treat problem IDs as the sampling clusters and retain related translation directions within a cluster. The 80% power MDE and 95% CI half-width are worst-case bounded paired-difference estimates; actual precision depends on paired model discordance and cross-direction covariance.",
        "",
        "| Benchmark | Common-Core problems | Approx. 80% power MDE (pp) | Worst-case 95% CI half-width (pp) |",
        "|---|---:|---:|---:|",
    ])
    for benchmark, filename, split in (
        ("xCodeEval", "xcodeeval_common_core_manifest.json", None),
        ("TransCoder-Test", "transcoder_test_manifest.json", "common_core"),
        ("HumanEval-X", "humanevalx_manifest.json", "common_core"),
        ("AVATAR-TC", "avatar_tc_manifest.json", "common_core"),
    ):
        data = json.loads((MANIFESTS / filename).read_text(encoding="utf-8"))
        if benchmark == "xCodeEval":
            section = data.get("common_core_coverage", {})
        else:
            section = data.get(split, {})
        precision = section.get("precision", data.get("precision", {}))
        if not precision and benchmark == "HumanEval-X":
            # The HumanEval-X file stores the common-core precision at its root.
            precision = data.get("precision", {})
        problem_count = section.get("unique_problems", len(data.get("common_core_problem_ids", [])))
        mde = precision.get("approximate_80_percent_power_mde_pp", precision.get("worst_case_80_percent_power_mde_pp", "n/r"))
        ci = precision.get("worst_case_95_percent_ci_half_width_pp", "n/r")
        lines.append(f"| {benchmark} | {problem_count} | {mde} | {ci} |")
    lines.extend([
        "## Interpretation",
        "",
        "Pass@1 is scored per translation task from one seeded stochastic candidate: the generated program must pass every eligible test for that task. Individual test assertions do not increase the number of independent problems; statistical analyses cluster related directions by problem ID.",
        "",
        "AVATAR-TC provides one recorded test input per task in its frozen manifest, so its large task count should not be mistaken for deep per-problem test coverage. Benchmark results must be interpreted with this coverage limitation and the benchmark-specific test-soundness limitations documented in the final report.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content = _render()
    temp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
