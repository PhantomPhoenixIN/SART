# Frozen research hypotheses and analysis plan

Status: **Protocol v3 amended by the user and frozen before external scoring (2026-10-12).** The archived v2 record preserves the earlier five-candidate plan. These hypotheses and primary analyses are frozen before any model prediction is scored on external tasks. Validation-only implementation choices may be made under the budgets in `configs/training/search.yaml`; test eligibility, hypotheses, metrics, and contrasts cannot be changed after external scores are observed.

## Primary scientific question

Does explicit program structure help a translation model decide where to spend recurrent neural computation, beyond effects from parameter count, recurrence, token/region grouping, or adaptive routing alone?

## Hypotheses

**H1 — Functional correctness at matched compute.** On the xCodeEval Common-Core set, Full-SART improves macro execution Pass@1 over Dense, Uniform Token Loop, and MoR-style Token Adaptive at approximately matched end-to-end inference FLOPs (target tolerance ±5% where feasible). Dense-ParamMatch tests whether capacity explains the result. TransCoder-Test and HumanEval-X provide independent benchmark replication; AVATAR-TC is auxiliary.

**H2 — Accuracy/compute frontier.** Full-SART has a better or competitive Pass@1 versus end-to-end-FLOPs Pareto frontier than the fixed-depth and token-adaptive baselines. If accuracy is similar, the relevant outcome may be lower total compute or latency rather than higher Pass@1.

**H3 — Structural-difficulty interaction.** The Full-SART minus MoR-style Pass@1 difference grows from the easiest to hardest frozen quartile of reference-derived structural transformation difficulty on common problem IDs.

**H4 — Routing validity.** Across source regions, expected recurrent depth is positively associated with independent source-to-reference structural difficulty (Spearman correlation). The reference target is used only for retrospective analysis, never as model input or routing signal.

**H5 — True structure contributes.** Full-SART outperforms RegionAdaptive-NoGraph, Random-Region Adaptive, and Shuffled-Structure SART under matched data and compute. These contrasts separate region content/grouping, arbitrary grouping, and explicit structural relations/types.

**H6 — Robustness.** Any supported architecture effect is not entirely dependent on one direction, one seed, one benchmark, or the Qwen family. Report all six directions, the three confirmatory seeds, xCodeEval/TransCoder-Test/HumanEval-X, and the 14B/StarCoder2 critical checks whether positive, null, or negative.

## Primary endpoint and aggregation

- Primary functional metric under Protocol v3: sampled execution Pass@1 from one seeded stochastic candidate per task. Tasks remain clustered under their programming problem. Do not pool archived v1 greedy or v2 five-candidate partial outputs with v3.
- When executable correctness is unavailable (XLCoST validation), report BLEU-4, CodeBLEU, exact match, and CrystalBLEU together. These lexical metrics never substitute for Pass@k on executable benchmarks.
- Primary benchmark: xCodeEval Common-Core; the unit of sampling is `src_uid`/problem ID, and all six directions from one problem stay in the same cluster.
- Primary aggregate: equal-weight macro average over the six directions. Show every directional score, micro average, and each external benchmark separately. Never pool Common-Core with Full-Directional.
- External model-checkpoint selection is forbidden. Checkpoint selection uses only XLCoST in-domain validation loss within the common frozen training horizon.

## Confirmatory contrasts and statistics

Primary comparison family: Full-SART versus Dense, Uniform Token Loop, MoR-style Token Adaptive, Random-Region Adaptive, and Structure-Only, on Pass@1. Use three seeds {13, 42, 87}; paired hierarchical bootstrap over seeds and problem IDs with at least 10,000 replicates; paired problem-cluster randomization; Holm correction across both endpoints and this comparison family. Report absolute percentage-point differences, relative error reduction, and 95% intervals. Per-direction McNemar tests are secondary.

H3 uses frozen within-direction equal-weight percentile-rank composites and quartiles defined before model scoring. H4 uses per-region expected depth and records direction/problem clusters; correlations are associational. H2 uses total end-to-end FLOPs plus prefill/recurrent/decode decomposition and measured latency; recurrent-module-only cost is not an efficiency endpoint.

The pre-model H3 artifact is `reports/analysis/xcodeeval_common_core_structural_difficulty.json` (status `FROZEN_PREMODEL`). It joins each frozen Common-Core task to its exact frozen source and target-reference hashes, derives AST node-type profile distance, region-type divergence, CFG and DFG edge-feature distances, API-name divergence, and operator divergence, then ranks available components within direction and averages them equally. Quartiles are direction-specific. AST and region features are unavailable when parsing fails; task quartiles require at least four of six dimensions. The AST component is a normalized node-type histogram/profile distance, not ordered tree-edit distance. H3's outcome analysis verifies task/code hashes and complete three-seed coverage, then hierarchically resamples seed IDs and problem clusters while retaining each cluster's direction cells. The artifact is retrospective-analysis metadata only and never enters model input, routing, or checkpoint selection.

The pre-model H4 artifact is `reports/analysis/xcodeeval_common_core_region_difficulty.json` (status `FROZEN_PREMODEL`). It uses a minimum-cost one-to-one assignment between language-normalized source and target-reference regions. Pair cost is the equal-weight mean of region-type mismatch, normalized log AST-depth difference, log child-count difference, and log byte-span-length difference; unmatched source regions receive cost 1. The external region cache uses semantic AST regions; CFG/DFG edges are not available in this matching artifact. This is a structural correspondence proxy, not semantic translation alignment. Both source and target parse for 1,852/1,992 tasks; the artifact stores all 148,694 source-region records where target parsing permitted comparison, with code hashes, region features, and matched IDs. External Full-SART prediction records preserve each source region's expected depth from the primary sample's prefill, region ID, byte span, type, and extracted structural features. `scripts/analysis/analyze_routing_difficulty.py` verifies task/code hashes and region spans before computing retrospective routing associations with hierarchical seed/problem bootstrap intervals. No benchmark outcomes influence the difficulty artifact, model input, routing, or checkpoint selection.

## Interpretation rules

- Do not infer causal mechanism from H4 correlation alone.
- A result at or below MoR is retained as null/negative evidence; no post-hoc test or benchmark changes.
- A small but significant effect is described as small. An uncertain estimate is not described as an improvement.
- Published third-party scores with different models/training are reported separately from controlled same-backbone results.
