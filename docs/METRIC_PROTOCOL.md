# Metric protocol

Status: Protocol v3 was frozen on 2026-10-12 before external scoring. The
published compact score export contains completed Protocol v3 runs; its index
lists any pending runs. The v2 five-candidate plan and partial predictions
are archived and excluded from v3 analyses.

## Executable benchmarks

- **Pass@1** uses one seeded stochastic candidate per frozen translation task
  (temperature 0.8, top-p 0.95). The candidate must pass every frozen eligible test.
  Its seed is derived from the experiment ID, task ID, and sample index 0.
- The six directions are reported individually and as an equal-weight macro. Related
  directions from the same problem remain one statistical cluster.

## XLCoST validation, without functional tests

Report the bundle **BLEU-4, CodeBLEU, exact match, and CrystalBLEU** for each of the six
directions, followed by an equal-weight direction macro. These are diagnostics and do not
replace execution-based metrics on external benchmarks. Validation target loss alone
selects checkpoints.

All four metrics use Pygments Python, Java, or C++ lexers with comments and whitespace
removed. BLEU-4 is corpus BLEU with four equal n-gram weights. CodeBLEU uses the pinned
`k4black/codebleu` commit
[`b0edb622f6a52fe9d1edc407be5061d3e1462a7f`](https://github.com/k4black/codebleu/tree/b0edb622f6a52fe9d1edc407be5061d3e1462a7f),
with uniform component weights; retain its n-gram, weighted n-gram, syntax, and data-flow
components. Its bundled implementation routes C++ data-flow extraction through its C#
DFG function, so that component is an approximation and must not be interpreted as a
validated C++ static-analysis result. If no reference data-flow edges are extracted for a
direction, the package returns a zero data-flow component but substitutes 1 for that
component in the composite; retain the component and report this package behavior. CrystalBLEU uses `k=500` most frequent 1–4-grams
computed from all target references within each frozen validation direction; metric scores
are computed on the balanced generated validation subset.
Exact match normalizes line endings and otherwise compares extracted code exactly.

The CrystalBLEU implementation follows the package API documented at
[`sola-st/crystalbleu`](https://github.com/sola-st/crystalbleu) and pinned locally as
`crystalbleu==0.2`. NLTK is pinned to `3.9.2`; Pygments is pinned to `2.19.1`.
