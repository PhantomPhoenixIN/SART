# SART-CT

Research artifact for **Structure-Adaptive Recurrent Computation for Semantics-Preserving Code Translation**. It contains the model and routing implementations, training and evaluation code, frozen experiment configurations, benchmark selection manifests, tests, and compact per-task execution outcomes.

The study compares a standard transformer with uniform token and region recurrence, token-adaptive recurrence, and structure-adaptive region recurrence across Python, Java, and C++ translation. The primary backbone is Qwen2.5-Coder-7B. Results from Qwen2.5-Coder-1.5B, Qwen2.5-Coder-14B, and StarCoder2-7B serve as scale and model-family checks.

## Reproduce

Read [Reproducibility](docs/REPRODUCIBILITY.md) for the pinned environment, dataset/model provenance, preprocessing, training, evaluation, and analysis commands. [Data and license notices](docs/DATA_AND_LICENSES.md) identify upstream sources and redistribution terms. The release includes compact score rows and a script to restore the directory layout expected by the statistical analyses.

```bash
python3.10 -m venv env
env/bin/python -m pip install -r requirements.lock
env/bin/python -m pytest -q tests/test_sparse_adaptive_recurrence.py tests/test_paired_statistics.py
```

GPU experiments were run on NVIDIA A100 hardware with CUDA 11.8 compatible PyTorch. The `run_on_astra/` job files target OpenPBS and contain site-specific paths; adapt their project-root and queue settings on another cluster.

## Repository map

- `src/`: dataset, model, recurrent router, evaluation, and statistics implementations.
- `scripts/`: acquisition, preprocessing, training, evaluation, analysis, and score-export entry points.
- `configs/`: frozen model, dataset, decoding, and analysis settings.
- `data/manifests/`: frozen benchmark task selections. Three manifests containing benchmark source code are compressed; unpack them before rerunning evaluation.
- `reports/evaluation/frozen_external_protocol.json`: frozen decoding and routing policy.
- `reports/results/compact/`: per-task execution outcome rows with checksums and a run index.
- `reports/statistics/`, `reports/analysis/`, `reports/profiling/`: reproducible analysis outputs and limited profiling records.
- `experiments/registry.csv`: canonical training-run registry.

Raw benchmark corpora, pretrained model weights, trained checkpoints, local environments, and per-task generated programs are not in Git. The pinned upstream sources, local reconstruction steps, and the scope of compact published outcomes are documented in [Reproducibility](docs/REPRODUCIBILITY.md). The local execution runner is an ExecEval-compatible fallback; it is not the official ExecEval service.
