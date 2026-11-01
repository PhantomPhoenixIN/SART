#!/usr/bin/env python3
"""Assemble the final artifact-backed research report after all analyses finish."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
REGISTRY = ROOT / "experiments/registry.csv"
DRAFT = ROOT / "reports/SART_CT_research_report_draft.md"
OUTPUT = ROOT / "reports/SART_CT_research_report.md"
ARTIFACTS = {
    "external": ROOT / "reports/analysis/external_benchmark_summary.json",
    "primary": ROOT / "reports/statistics/xcodeeval_common_core_primary.json",
    "efficiency": ROOT / "reports/analysis/xcodeeval_common_core_efficiency_frontier.json",
    "h3": ROOT / "reports/analysis/xcodeeval_common_core_h3_difficulty_interaction.json",
    "h4": ROOT / "reports/analysis/xcodeeval_common_core_h4_routing_difficulty.json",
    "h5": ROOT / "reports/analysis/xcodeeval_common_core_h5_structure_controls.json",
}
COMPARISONS = (
    "full_sart_vs_dense", "full_sart_vs_uniform_token_loop", "full_sart_vs_mor_style",
    "full_sart_vs_random_region", "full_sart_vs_structure_only",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_complete() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREMODEL" or protocol.get("protocol_version") != 3:
        raise ValueError("Final report requires frozen one-candidate Protocol v3")
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    canonical = [row for row in rows
                 if row.get("training_horizon_policy") != "up-to-5-epoch-validation-pilot"
                 and row.get("training_status") != "CANCELLED_SUPERSEDED"]
    if len(canonical) != 55 or any(row.get("training_status") != "TRAINED" for row in canonical):
        raise ValueError("Final report requires all 55 canonical model runs trained")
    values = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in ARTIFACTS.items()}
    external = values["external"]
    if (
        external.get("status") != "COMPLETE"
        or external.get("protocol_sha256") != sha256(PROTOCOL)
        or external.get("canonical_model_runs") != 55
        or external.get("prediction_score_runs") != 440
        or len(external.get("detail_rows", [])) != 440
    ):
        raise ValueError("External summary does not reconcile the complete frozen matrix")
    if set(values["primary"].get("comparisons", {})) != {
        f"pass_at_1:{name}" for name in COMPARISONS
    }:
        raise ValueError("Primary statistics lack the five Pass@1 contrasts")
    if (values["efficiency"].get("status") != "analyzed"
            or len(values["efficiency"].get("model_groups", [])) != 55):
        raise ValueError("Efficiency frontier is incomplete")
    if any(values[name].get("status") != "analyzed" for name in ("h3", "h4", "h5")):
        raise ValueError("H3/H4/H5 mechanisms are incomplete")
    markdown = ROOT / "reports/analysis/external_benchmark_summary.md"
    if not markdown.is_file() or markdown.read_text(encoding="utf-8").count("## ") != 8:
        raise ValueError("Eight complete benchmark/split result tables are required")
    validation_screen = ROOT / "reports/progress/in_domain_validation_screen.md"
    if not validation_screen.is_file():
        raise ValueError("The final report requires the XLCoST validation metric screen")
    validation_text = validation_screen.read_text(encoding="utf-8")
    if (
        "Canonical trained runs included: **40**" not in validation_text
        or "BLEU-4 | CodeBLEU | CrystalBLEU | Exact match" not in validation_text
        or "still required" in validation_text
    ):
        raise ValueError("The XLCoST validation screen is incomplete or stale")
    return protocol, values


def _fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def render(protocol: dict[str, Any], values: dict[str, dict[str, Any]]) -> str:
    draft = DRAFT.read_text(encoding="utf-8")
    methods = draft.split("## Results and discussion — pending", 1)[0]
    method_lines = methods.splitlines()
    status_indexes = [index for index, line in enumerate(method_lines)
                      if line.startswith("**Status:**")]
    if len(status_indexes) != 1:
        raise ValueError("Draft must contain exactly one status line")
    method_lines[status_indexes[0]] = (
        "**Status:** complete artifact-backed report generated "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} from frozen Protocol v3."
    )
    methods = "\n".join(method_lines)
    primary = values["primary"]
    external = values["external"]
    efficiency = values["efficiency"]
    h3 = values["h3"]
    h4 = values["h4"]
    h5 = values["h5"]
    lines = [methods.rstrip(), "", "## Results and discussion", "",
             "All 55 canonical model runs and 440 benchmark/split prediction and score runs are complete. "
             "The [full external results tables](analysis/external_benchmark_summary.md) list Pass@1 "
             "for every run and each of the six directions. The "
             "[machine-readable summary](analysis/external_benchmark_summary.json) retains "
             "experiment IDs, score-manifest hashes, and seed groupings.", "",
             "### Primary 7B xCodeEval Common-Core contrasts", "",
             "Effects are Full-SART minus the named baseline in Pass@1 percentage points. "
             "Intervals resample seeds and problem clusters; p values use paired problem-cluster "
             "randomization and Holm adjustment across the five planned contrasts.", "",
             "| Baseline | Effect (pp) | 95% CI (pp) | Raw p | Holm p | Matched FLOPs (±5%) | Assessment |",
             "|---|---:|---:|---:|---:|---|---|" ]
    main_backbone = next((v for k, v in efficiency["backbone_results"].items()
                          if k.endswith("0396a76181e127dfc13e5c5ec48a8cee09938b02")), None)
    if main_backbone is None:
        raise ValueError("Primary backbone has no efficiency summary")
    for name in COMPARISONS:
        result = primary["comparisons"][f"pass_at_1:{name}"]
        bootstrap = result["bootstrap"]
        low, high = bootstrap["ci95_percentage_points"]
        raw_p = result["paired_cluster_randomization"]["p_value_two_sided"]
        adjusted = primary["holm_adjusted_p_values"][f"pass_at_1:{name}"]
        compute = main_backbone["full_sart_comparisons"].get(name)
        matched = compute is not None and compute["within_five_percent_flops"]
        effect = bootstrap["difference_percentage_points"]
        if matched and effect > 0 and low > 0 and adjusted < 0.05:
            assessment = "supports a matched-compute Pass@1 improvement"
        elif matched and effect < 0 and high < 0 and adjusted < 0.05:
            assessment = "Full-SART performs worse at matched compute"
        elif not matched:
            assessment = "compute tolerance unmet; accuracy contrast reported"
        else:
            assessment = "improvement uncertain under the planned test"
        baseline = name.removeprefix("full_sart_vs_")
        lines.append(
            f"| {baseline} | {effect:+.2f} | [{low:+.2f}, {high:+.2f}] | "
            f"{_fmt(raw_p, 4)} | {_fmt(adjusted, 4)} | "
            f"{'yes' if matched else 'no'} | {assessment} |"
        )
    lines.extend(["", "Relative error reduction, per-direction effects, McNemar summaries, "
                  "realized problem counts, and full resampling metadata are in the "
                  "[primary statistics artifact](statistics/xcodeeval_common_core_primary.json).", "",
                  "### Accuracy and inference cost", "",
                  "The [efficiency frontier](analysis/xcodeeval_common_core_efficiency_frontier.json) "
                  "contains 55 model-level profiles and reports total end-to-end FLOPs, "
                  "prefill/recurrent/decode components, latency, memory, and parameters. "
                  "The profile uses 24 frozen in-domain validation examples; external accuracy "
                  "comes from xCodeEval Common-Core. Its Pareto frontier is descriptive.", "",
                  "The [XLCoST validation screen](progress/in_domain_validation_screen.md) reports "
                  "BLEU-4, CodeBLEU components, CrystalBLEU, exact match, and routing diagnostics "
                  "for all 40 canonical 7B runs. These are descriptive diagnostics; checkpoint "
                  "selection uses validation target loss only.", "",
                  "| Primary 7B architecture | Mean Pass@1 over seeds | Median total FLOPs | Median latency (ms) |",
                  "|---|---:|---:|---:|"])
    for architecture, row in sorted(main_backbone["architectures"].items()):
        lines.append(
            f"| {architecture} | {_fmt(row['pass_at_1_macro_mean_over_seeds'], 4)} | "
            f"{_fmt(row['median_estimated_total_flops'], 2)} | "
            f"{_fmt(row['median_end_to_end_fixed_device_ms'], 2)} |"
        )
    lines.extend(["", "### Structural difficulty and routing", "",
                  f"H3 Full-SART minus MoR-style Q4−Q1 interaction: "
                  f"{_fmt(h3['macro_equal_direction_q4_minus_q1'], 4)} "
                  f"(95% clustered bootstrap interval "
                  f"{h3['hierarchical_seed_problem_bootstrap_95_percent_interval']}; "
                  f"{h3['matched_problem_count']} matched problems). "
                  "This is a retrospective structural-difficulty analysis. "
                  "[Full H3 output](analysis/xcodeeval_common_core_h3_difficulty_interaction.json).", "",
                  f"H4 source-region expected depth versus structural difficulty: "
                  f"Spearman ρ={_fmt(h4['spearman_rho'], 4)} "
                  f"(95% clustered bootstrap interval "
                  f"{h4['hierarchical_seed_problem_bootstrap_95_percent_interval']}; "
                  f"{h4['per_region_observations']} region observations). "
                  "This association is non-causal and uses a structural correspondence proxy. "
                  "[Full H4 output](analysis/xcodeeval_common_core_h4_routing_difficulty.json).", "",
                  "### True-structure controls", "",
                  "The H5 controls compare the seed-42 Full-SART run with graph-free region "
                  "routing, random regions, and shuffled structure on identical Common-Core "
                  "problems. These secondary single-seed intervals resample problems and do "
                  "not capture seed variation. [Full H5 output]"
                  "(analysis/xcodeeval_common_core_h5_structure_controls.json).", "",
                  "| H5 baseline | Pass@1 effect (pp) | 95% problem interval (pp) | Matched FLOPs (±5%) |",
                  "|---|---:|---:|---|"])
    for name, result in h5["comparisons"].items():
        low, high = result["hierarchical_problem_bootstrap_95_percent_interval_percentage_points"]
        lines.append(
            f"| {name.removeprefix('full_sart_vs_')} | "
            f"{result['pass_at_1_difference_percentage_points']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{'yes' if result['within_five_percent_flops'] else 'no'} |"
        )
    lines.extend(["", "### Robustness across benchmarks, size, and model family", "",
                  "The following Common-Core values are descriptive means over each listed "
                  "seed set. Full-Directional values and every directional result remain in "
                  "the complete external tables. The 14B and StarCoder2 checks have one seed "
                  "each and do not provide seed uncertainty.", "",
                  "| Backbone revision | Benchmark | Architecture | Seeds | Mean Pass@1 macro |",
                  "|---|---|---|---:|---:|"])
    robustness_groups = [
        group for group in external["groups"]
        if group["split"] == "common_core"
        and group["architecture"] in {"full_sart", "dense", "mor_style"}
        and (group["backbone_revision"] == "0396a76181e127dfc13e5c5ec48a8cee09938b02"
             or group["benchmark"] == "xcodeeval")
    ]
    for group in sorted(robustness_groups, key=lambda row: (
        row["backbone_revision"], row["benchmark"], row["architecture"]
    )):
        lines.append(
            f"| `{group['backbone_revision']}` | {group['benchmark']} | "
            f"{group['architecture']} | {group['seed_count']} | "
            f"{_fmt(group['metrics']['pass_at_1']['mean_over_seeds'], 4)} |"
        )
    lines.extend(["", "All null and negative outcomes remain in the linked per-run tables. "
                  "A positive routing association alone does not establish a causal mechanism.", "",
                  "## Reproducibility and limitations", "",
                  f"- Frozen external protocol: [Protocol v3](evaluation/frozen_external_protocol.json), "
                  f"SHA-256 `{sha256(PROTOCOL)}`. The archived "
                  "[v2 protocol](evaluation/frozen_external_protocol_v2_archived.json) "
                  "and its partial predictions are excluded from every v3 analysis.",
                  "- Canonical training registry and 55 selected checkpoints: "
                  "[registry](../experiments/registry.csv); exact checkpoint paths, seeds, "
                  "and training configs are in its rows.",
                  "- Frozen task identities, references, tests, and structure caches: "
                  "[manifest directory](../data/manifests/) and "
                  "[external structural caches](../data/structural/external/).",
                  "- Attempt and environment provenance: "
                  "[prediction ledger](progress/external_prediction_jobs_v3.csv), "
                  "[scoring ledger](progress/external_scoring_jobs_v3.csv), "
                  "[profile ledger](progress/efficiency_profile_jobs.csv), and "
                  "per-attempt environment directories under `experiments/`.",
                  "- Reproduction code: "
                  "[generation](../scripts/evaluate/generate_external_predictions.py), "
                  "[scoring](../scripts/evaluate/score_external_predictions.py), "
                  "[paired statistics](../scripts/analysis/run_primary_statistics.py), "
                  "[efficiency profiling](../scripts/evaluate/profile_inference_cost.py), "
                  "and [result summary](../scripts/analysis/summarize_external_results.py).",
                  "- Benchmark executions use the project's isolated compatibility harness, "
                  "not the official ExecEval service. Pretrained-model contamination cannot "
                  "be excluded. Parser/graph coverage and structural region matching are "
                  "imperfect. Family checks with one seed have limited uncertainty coverage. "
                  "The user-directed Pass@1-only amendment removes multi-candidate endpoint evidence.", ""])
    return "\n".join(lines)


def main() -> None:
    protocol, values = _read_complete()
    body = render(protocol, values)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
