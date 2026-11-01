# Frozen xCodeEval Common-Core structural difficulty

Status: `FROZEN_PREMODEL`. Features are computed only from frozen selected source programs,
accepted canonical target references, and the frozen Joern cache. No model predictions,
outcomes, or scores are read. Quartiles must not be recomputed after outcomes are observed.

Tasks: 1992 across six directions and 332 common problems.

## Feature definition

The composite is the equal-weight mean of within-direction percentile ranks over available
dimensions (at least four of six per task): named AST node-type distribution distance,
semantic-region type Jensen-Shannon distance, CFG edge-type density distance, DFG edge-type
density distance, call-name Jaccard divergence, and operator-histogram distance. AST node
distance is a normalized feature-profile distance, not ordered tree-edit distance. If either
program fails the pinned Tree-sitter parse, AST and region dimensions are masked; available
CFG/DFG/API/operator features remain, and parse status is retained per side. Python 2 sources
are normalized with lib2to3 only when required for parsing, with that operation recorded.

## Feature coverage

| Feature | Tasks with value | Total |
|---|---:|---:|
| ast_node_distance | 1852 | 1992 |
| region_type_divergence | 1852 | 1992 |
| cfg_feature_distance | 1992 | 1992 |
| dfg_feature_distance | 1992 | 1992 |
| api_divergence | 1992 | 1992 |
| operator_divergence | 1992 | 1992 |

Every task received a frozen quartile label. CFG and DFG feature coverage is required on
both source and reference; where those are available, parser failures leave four other
dimensions available. Coverage flags and exact task-level features are in the JSON artifact.

## Direction-specific frozen thresholds

| Direction | Q1/Q2 edge | Q2/Q3 edge | Q3/Q4 edge | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++-to-Java | 0.369733 | 0.503422 | 0.651643 | 83 | 83 | 83 | 83 |
| C++-to-Python | 0.392944 | 0.492912 | 0.631370 | 83 | 83 | 83 | 83 |
| Java-to-C++ | 0.369733 | 0.503422 | 0.651643 | 83 | 83 | 83 | 83 |
| Java-to-Python | 0.384315 | 0.498741 | 0.606747 | 83 | 82 | 84 | 83 |
| Python-to-C++ | 0.392944 | 0.492912 | 0.631370 | 83 | 83 | 83 | 83 |
| Python-to-Java | 0.384315 | 0.498741 | 0.606747 | 83 | 82 | 84 | 83 |

Quartiles are ranked independently within each direction so language-pair baselines do not
set a shared threshold. The primary H3 comparison should join outcomes by problem ID and
direction using these frozen labels; keep any missing outcome cells explicit.

## Provenance hashes

- xCodeEval Common-Core task manifest: `7b58147227e21e26575b7c94635f9a174148ef7ee3c956dd963947ed8d8c912b`
- xCodeEval reference-selection manifest: `adf3970efe8a6510a7c58a3b4eab01b140c6f726de9429bd7afe74d7f22af287`
- External Joern edge manifest: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`

Machine-readable task-level features and source/reference hashes: `xcodeeval_common_core_structural_difficulty.json`.
