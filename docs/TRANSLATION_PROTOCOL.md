# Translation protocol

Status: **Protocol v2 amendment frozen before external scoring on 2026-10-12.** Protocol v1 is archived at `docs/archive/TRANSLATION_PROTOCOL_v1.md`. Serialization, training data, checkpoints, benchmarks, tests, and scoring remain unchanged. The resolved training configuration remains in `configs/data/sequence.yaml`.

All architecture variants on a backbone use the exact same serialized prompt, data examples, target text, generation config and output extraction. Do not add model-specific prompts or repair generated code in the primary evaluation.

## Canonical serialization

Plain UTF-8 text; markers are literal text and are not newly registered tokenizer special tokens:

```text
Translate the complete source program from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}.
Preserve its behavior. Return only the translated program.

### TARGET INTERFACE ({TARGET_LANGUAGE})
{target_interface_or_empty}

### SOURCE PROGRAM ({SOURCE_LANGUAGE})
{source_code}
### END SOURCE PROGRAM

### TARGET PROGRAM ({TARGET_LANGUAGE})
```

The model target is only the reference target program with line endings normalized to LF. Source and target languages are the literal values `Python`, `Java`, or `C++`. `target_interface_or_empty` contains a benchmark-provided target declaration when the task defines one; otherwise it is the empty string. In HumanEval-X, use the target-language `declaration` exactly as the interface and omit its docstring/description, consistent with the benchmark's translation task. Do not synthesize interfaces. This field is provided identically to every model on that task. Use this protocol for Dense, Dense-ParamMatch, all recurrent methods, graph ablations, seeds and both backbones. A format change after the validation pilot requires a new documented protocol version and rerun of affected experiments.

## Output extraction and cleanup

1. Use only newly generated tokens; do not parse hidden chain-of-thought or call another model.
2. Normalize `\r\n` and `\r` to `\n`.
3. If and only if the entire output consists of one Markdown fenced block (opening fence, optional language label, content, closing fence), remove that one pair of fences. Apply the same deterministic parser to every architecture.
4. Do not trim/repair code beyond this; do not add imports, braces, function definitions, symbols, or inferred boilerplate. Raw execution Pass@1 and empirical Pass@5 are the primary functional metrics.
5. An empty, partial, truncated, syntactically invalid or un-compilable extracted result is scored as generated; it receives no model-specific correction.

## Decoding and length

- Pass@1 and Pass@5 use the same set of exactly five sampled candidates (`do_sample=true`, `num_beams=1`). Pass@1 is the correctness of sample index 0; Pass@5 is whether any of the five candidates passes every test. Seeds derive deterministically from experiment ID, task ID, and sample index. This user-directed v2 amendment removes the separate greedy generation to reduce decoding work. Pass-Ratio@5 averages the squared ratio of distinct tests passed for each of those five candidates; also report the unsquared average as a diagnostic. Store each candidate separately and aggregate at the programming-problem cluster. Sampling parameters are frozen in `configs/evaluation/prediction.yaml`; never tune them on external tests. V2 Pass@1 is stochastic and must not be pooled with or described as the v1 greedy Pass@1 metric.
- For lexical evaluation, always report BLEU-4, CodeBLEU, exact match, and CrystalBLEU as a bundle. They are descriptive and do not replace functional correctness.
- On the non-executable XLCoST validation split, compute corpus BLEU-4, CodeBLEU, exact match, and CrystalBLEU for each direction, then report an equal-weight direction macro. Use the Pygments language lexer with whitespace and comments removed for BLEU-family tokenization. CrystalBLEU uses `k=500`; its ignored common n-grams are estimated from all target references in that direction's frozen validation split, while scores use the balanced generated subset. CodeBLEU uses the pinned k4black implementation at commit `b0edb622f6a52fe9d1edc407be5061d3e1462a7f`, with the same lexer tokens. Report CodeBLEU's n-gram, weighted n-gram, syntax, and data-flow components alongside the composite. Exact match normalizes line endings only. These metrics do not select checkpoints; validation target loss remains the sole selection criterion.
- Full definitions, package pins, benchmark availability rules, and the CodeBLEU C++ data-flow limitation are recorded in `docs/METRIC_PROTOCOL.md`.
- For Qwen2.5-Coder-7B, use `max_new_tokens=768`. The measured train-plus-validation target 99.5th percentile, including EOS, is 671 tokens and was rounded upward to the next 256-token boundary. Use a 2,048-token training context and exclude rather than truncate any example whose full protocol prompt plus target and EOS exceeds it. This removes 18/47,134 clean training directions and 8/2,406 validation directions; no clean in-domain test direction exceeds the limit. Full distributions are in `reports/data/qwen_token_length_audit.json`.
- Report whether max length was reached without EOS (truncation), output token length, and latency for every attempt. The generated code at the fixed cap is evaluated as-is.
- For any additional backbone, choose its corresponding token cap using the same training/validation-only percentile rule; freeze before external inference.

## Controls before final use

Before starting the convergence pilot, validate serialization round-trips for source content, target masking, source-only recurrent prefix span, and output extraction with synthetic strings containing code fences and line-ending variants. Record final resolved prompt and generation configuration in each experiment manifest.

## User-directed protocol amendment — 2026-10-12

The user authorized stochastic generations instead of a separate greedy decode for Pass@1 and Pass@5 to reduce evaluation time. Before external scores existed, Protocol v2 was frozen with Pass@1 defined as sample 0 of the exact five seeded Pass@5 candidates. All five candidates and decoding settings remain shared across every model and benchmark. The four partial v1 prediction runs were cancelled before scoring; their outputs and protocol are preserved under `reports/predictions_v1/` and `reports/evaluation/archive/`. V2 predictions and scores use separate `protocol_v2` output trees.

## External-evaluation freeze gate

The external prediction driver requires a JSON freeze record passed as
`--frozen-protocol`. Do not create this record until the convergence pilot has frozen the
shared horizon, the best-within-horizon checkpoints have been selected using only the
in-domain validation split, and each adaptive method's inference threshold or top-k
fraction has been calibrated on that validation split. The record must use status
`FROZEN_PREMODEL` and include:

- `selection_metric: validation_target_loss` and
  `external_metrics_used_for_selection: false`;
- `training_horizon_epochs`, keyed by the pinned backbone revision directory name;
- `translation_protocol_sha256`, hashing this document;
- `external_task_manifests`, keyed as `<benchmark>:<split>` and containing the
  corresponding frozen task-manifest SHA256 for every benchmark/split to be evaluated;
- the exact deterministic generation fields from `configs/evaluation/prediction.yaml`,
  including per-backbone EOS ID and output cleanup version;
- `routing_policies`, keyed first by pinned backbone revision directory and then by
  architecture, containing its validation-selected `hard_threshold` or
  `top_k_fraction`, `max_recurrence`, and `insertion_fraction`. Each backbone can have
  separate validation-calibrated thresholds so matched inference budgets are not assumed
  to transfer across model sizes or families.

For each adaptive architecture and pinned backbone revision, calibrate using only the
seed-42 validation-selected `best.pt` checkpoint and the deterministic validation subset.
Apply that single selected policy to every confirmatory seed for the same architecture and
revision; do not calibrate thresholds separately by seed. This keeps the routing policy
fixed across seed replicates. Report realized recurrent depth and measured inference cost
for every seed. Non-adaptive architectures use the frozen policy values `hard_threshold: null`
and `top_k_fraction: null`.

Before loading external tasks for generation, the driver checks the checkpoint's epoch
horizon, validation-selected `best.pt` status, backbone revision, translation-protocol
hash, frozen task-manifest hash, generation settings and architecture routing settings.
Missing or mismatched fields stop inference. The record itself is copied into the output
run manifest so the result can be traced to the frozen policy. Validation profiling and
calibration do not use this external-generation driver.

## Benchmark-specific input availability

The canonical format has fixed fields for target interface, source program, and target program prefix. For XLCoST and xCodeEval, the interface field is empty unless an official task artifact supplies a target signature. For HumanEval-X, it contains the official target declaration while the description is excluded. TransCoder-Test and AVATAR-TC follow their official task interface; include only interface material available before generation and record it in the frozen task manifest. The architecture prompt remains unchanged across benchmarks and methods.

## TransCoder-Test task adapter (Protocol v1)

- The input is the single official source-language `f_gold` function for the problem, extracted from the pinned benchmark's corresponding test script. The target interface is the official target `f_gold` signature with only the function symbol changed to `f_filled`, matching the benchmark's `TOFILL` harness contract. This exposes the callable type and name but no target implementation.
- The target text is target-language function code named `f_filled`. The evaluator places it verbatim at the official `TOFILL` marker. It does not rename or repair generated code. The official test harness compares `f_filled` with `f_gold` for every official input.
- Before inference, canonical preflight duplicates the trusted `f_gold` implementation as `f_filled`, then checks the official reported pass count against the audited test-input count in three runs. Preflight references and failures are recorded by task ID.
- Some official Java harnesses import `javafx.util.Pair`, which is not shipped in the pinned JDK. Those tasks use a small source-compatible `Pair` class compiled from the project sandbox only; this adjustment applies to reference and candidate compilation identically and is listed in each manifest/runtime report.

## AVATAR-TC task adapter (Protocol v1)

- The source prompt contains the one pinned official test-split program in its original language. The target interface is empty because AVATAR-TC supplies complete Java/Python console programs rather than callable-function declarations.
- The generated target is a complete target-language program. It receives the official input bytes on standard input and is scored against the frozen output alternatives. The source and target references were each required to parse, pass the published case and do so consistently in three executions before the problem entered the pre-model manifest.
- Matching is case-insensitive and ignores whitespace, retains punctuation, and uses the shared numeric-token tolerance. Multiple published acceptable output strings are passed as alternatives. No generated output repair or language-specific postprocessing is permitted beyond the global protocol.
- AVATAR-TC remains auxiliary because it contains one test input per problem and its artifact license has not been independently verified. Exact common task IDs and test/code hashes are in `data/manifests/avatar_tc_manifest.json`.
