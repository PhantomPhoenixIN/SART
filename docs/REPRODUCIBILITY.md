# Reproducibility

## Scope and versions

The implementation and experiment settings are under `src/`, `scripts/`, and `configs/`. `requirements.lock` pins the Python environment, including PyTorch 2.5.1+cu118 and Transformers 4.48.3. The primary Qwen2.5-Coder-7B revision is `0396a76181e127dfc13e5c5ec48a8cee09938b02`. Other pinned model and dataset revisions are in `configs/data/artifacts.yaml` and the `content/**/artifact_manifest.json` files. Each canonical model run, seed, config, and checkpoint path is recorded in `experiments/registry.csv`.

The original environment used Python 3.10, NVIDIA A100 GPUs, driver 520.61.05, and CUDA 11.8 compatible wheels. A fresh environment can be created with:

```bash
python3.10 -m venv env
env/bin/python -m pip install --upgrade pip
env/bin/python -m pip install -r requirements.lock
```

The CUDA wheel index is already specified in `requirements.lock`. Use a PyTorch build appropriate for another host's driver if needed, and record the substitution. The benchmark execution harness also requires C++, Java, and Python runtimes documented in `docs/EXECUTION_SANDBOX.md` and `configs/evaluation/benchmarks.yaml`.

## Public data and models

Retrieve the pinned upstream XLCoST, xCodeEval, TransCoder-Test, HumanEval-X, and AVATAR-TC snapshots with the acquisition scripts in `scripts/download/` and the revisions in `configs/data/artifacts.yaml`. Fetch the pretrained models from their pinned Hugging Face revisions. `content/**/artifact_manifest.json` records local source hashes. The published XLCoST `manifest.json` files preserve the exact training-sample selection and structural-shard checksums; the external structural-cache manifests preserve task/cache hashes. Shard payloads are rebuilt from the pinned sources. No trained checkpoint is bundled with this repository; retraining from the frozen configs or obtaining matching checkpoints is required to regenerate predictions.

The three benchmark manifests containing upstream source programs are compressed in `data/manifests/`. Expand them in place before evaluating:

```bash
gzip -dk data/manifests/xcodeeval_reference_selection.json.gz
gzip -dk data/manifests/transcoder_test_manifest.json.gz
gzip -dk data/manifests/avatar_tc_manifest.json.gz
gzip -dk reports/analysis/xcodeeval_common_core_region_difficulty.json.gz
```

The exact selection rules and hashes are preserved in the manifests. Raw XLCoST training examples, full upstream repositories, and model weights are obtained separately. The methods and attribution obligations for benchmark material are summarized in `docs/DATA_AND_LICENSES.md`.

`docs/TRANSLATION_PROTOCOL.md` is retained byte for byte because the frozen protocol records its SHA-256; it includes historical Protocol v2 decoding text. For the published Protocol v3 score rows, the operative one-candidate decoding settings are `reports/evaluation/frozen_external_protocol.json` and the Pass@1 definition in `docs/METRIC_PROTOCOL.md`. Do not use the archived five-candidate text to reinterpret Protocol v3 results.

## Rebuild the experiment

1. Verify the acquired files against their artifact manifests. Prepare XLCoST training pairs and structures with `scripts/preprocess/` and the frozen settings in `configs/data/`. The training and validation sample policy is in `configs/data/confirmatory_training.yaml` and `configs/training/horizon.yaml`.
2. Generate or inspect the 55 canonical run configs under `configs/generated/` and `experiments/registry.csv`. Run training with `src/training/run_translation.py` using the corresponding YAML config. The OpenPBS wrappers under `run_on_astra/` show the cluster resource requests; replace the hard-coded installation root and queue names on another system.
3. Recreate benchmark structural caches with `scripts/preprocess/prepare_external_structures.py`, then use the frozen task manifests, `docs/TRANSLATION_PROTOCOL.md`, and `reports/evaluation/frozen_external_protocol.json` for prediction. The entry point is `scripts/evaluate/generate_external_predictions.py`.
4. Execute frozen tests with `scripts/evaluate/score_external_predictions.py`. The rootless local harness is described in `docs/EXECUTION_SANDBOX.md`. It is a compatibility fallback rather than the official ExecEval service; do not pool its scores with official ExecEval scores.

Model checkpoints are selected using in-domain XLCoST validation loss only. The external xCodeEval Common-Core and Full-Directional sets remain separate. The primary statistical comparisons and correction family are specified in `configs/evaluation/statistics.yaml` and `docs/METRIC_PROTOCOL.md`.

The 40-run primary-family XLCoST validation screen, including BLEU-4, CodeBLEU, exact match, and CrystalBLEU, is preserved in `reports/progress/in_domain_validation_screen.md`; it is descriptive and does not replace executable external evaluation.

## Recompute analyses from published outcomes

`reports/results/compact/` contains one compressed JSON Lines file per benchmark plus `score_export_index.json`. The index lists the complete run manifests, their checksums, task counts, and checksums of the exported files. It is a snapshot of completed Protocol v3 score runs; pending runs are listed explicitly. Each row retains the task/problem identity, seed, direction, Pass@1, execution status, test counts, and source/prediction/test hashes. Generated programs and full harness diagnostics are not included in this compact export.

Restore the tree shape expected by the analysis scripts, then run the primary analysis:

```bash
env/bin/python scripts/reproduce/materialize_compact_scores.py \
  --compact-root reports/results/compact \
  --output reports/results/protocol_v3
env/bin/python scripts/analysis/run_primary_statistics.py \
  --results-root reports/results/protocol_v3 \
  --output reports/statistics/xcodeeval_common_core_primary_recomputed.json
env/bin/python scripts/analysis/verify_region_loop_direct.py \
  --results-root reports/results/protocol_v3 \
  --output reports/statistics/xcodeeval_common_core_region_loop_direct_recomputed.json
```

The direct paired Region Loop comparison and its seed/problem bootstrap output are in `reports/statistics/xcodeeval_common_core_region_loop_direct_verification.json`. Profiling source and the same-example measurements are in `scripts/evaluate/profile_inference_cost.py` and `reports/profiling/verification/`. The one-example profile is a feasibility measurement; it is not the full 24-program efficiency study.

The six examples discussed in `reports/analysis/qualitative_analysis.txt` have their prediction and score records in `reports/qualitative_cases/` for all seven compared architectures. These records allow the illustrative cases to be checked against the released analysis; they are a selected sample, not a full prediction archive.

## Integrity checks

```bash
env/bin/python -m pytest -q tests/test_sparse_adaptive_recurrence.py tests/test_paired_statistics.py
env/bin/python -m py_compile scripts/evaluate/profile_inference_cost.py
```

For exact score provenance, compare each compact archive SHA-256 with `score_export_index.json`. Per-task generated program hashes and prediction-record hashes can be matched against independently regenerated prediction files. The compact score export can be recreated from the full score tree with `scripts/reproduce/export_compact_scores.py`.
