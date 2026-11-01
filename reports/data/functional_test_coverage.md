# Functional test coverage

Counts below are computed from the frozen pre-model eligibility manifests. A translation task is one selected source implementation in one directed language pair. `Test inputs across tasks` sums executable inputs over those translation tasks; when one problem is translated in multiple directions, its tests are counted once for each translation task. `Unique problems` is deduplicated by benchmark problem ID. The common-core and full-directional sets are reported separately.

## Aggregate coverage

| Benchmark | Set | Unique problems | Translation tasks | Directions | Test inputs across tasks | Mean / task | Median / task | Min / task | Max / task |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xCodeEval | Common-Core | 332 | 1992 | 6 | 102960 | 51.69 | 43.5 | 2 | 239 |
| xCodeEval | Full-Directional | 848 | 3024 | 6 | 165388 | 54.69 | 47 | 2 | 279 |
| TransCoder-Test | Common-Core | 321 | 1926 | 6 | 19260 | 10.00 | 10 | 10 | 10 |
| TransCoder-Test | Full-Directional | 326 | 1936 | 6 | 19360 | 10.00 | 10 | 10 | 10 |
| HumanEval-X | Common-Core | 157 | 942 | 6 | 7354 | 7.81 | 7 | 1 | 105 |
| HumanEval-X | Full-Directional | 164 | 956 | 6 | 7996 | 8.36 | 7 | 1 | 105 |
| AVATAR-TC | Common-Core | 1649 | 3298 | 2 | 3298 | 1.00 | 1 | 1 | 1 |
| AVATAR-TC | Full-Directional | 1649 | 3298 | 2 | 3298 | 1.00 | 1 | 1 | 1 |

## Per-direction counts

## xCodeEval — Common-Core

Manifest: `xcodeeval_common_core_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Python-to-Java | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |
| Java-to-Python | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |
| Python-to-C++ | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |
| C++-to-Python | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |
| Java-to-C++ | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |
| C++-to-Java | 332 | 17160 | 51.69 | 43.5 | 2 | 239 |

Test inputs summed across tasks: **102,960**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## xCodeEval — Full-Directional

Manifest: `xcodeeval_full_direction_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Python-to-Java | 349 | 18159 | 52.03 | 43 | 2 | 239 |
| Java-to-Python | 349 | 18159 | 52.03 | 43 | 2 | 239 |
| Python-to-C++ | 776 | 43984 | 56.68 | 48 | 2 | 279 |
| C++-to-Python | 776 | 43984 | 56.68 | 48 | 2 | 279 |
| Java-to-C++ | 387 | 20551 | 53.10 | 46 | 2 | 239 |
| C++-to-Java | 387 | 20551 | 53.10 | 46 | 2 | 239 |

Test inputs summed across tasks: **165,388**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## TransCoder-Test — Common-Core

Manifest: `transcoder_test_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| C++-to-Java | 321 | 3210 | 10.00 | 10 | 10 | 10 |
| C++-to-Python | 321 | 3210 | 10.00 | 10 | 10 | 10 |
| Java-to-C++ | 321 | 3210 | 10.00 | 10 | 10 | 10 |
| Java-to-Python | 321 | 3210 | 10.00 | 10 | 10 | 10 |
| Python-to-C++ | 321 | 3210 | 10.00 | 10 | 10 | 10 |
| Python-to-Java | 321 | 3210 | 10.00 | 10 | 10 | 10 |

Test inputs summed across tasks: **19,260**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## TransCoder-Test — Full-Directional

Manifest: `transcoder_test_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| C++-to-Java | 323 | 3230 | 10.00 | 10 | 10 | 10 |
| C++-to-Python | 323 | 3230 | 10.00 | 10 | 10 | 10 |
| Java-to-C++ | 323 | 3230 | 10.00 | 10 | 10 | 10 |
| Java-to-Python | 322 | 3220 | 10.00 | 10 | 10 | 10 |
| Python-to-C++ | 323 | 3230 | 10.00 | 10 | 10 | 10 |
| Python-to-Java | 322 | 3220 | 10.00 | 10 | 10 | 10 |

Test inputs summed across tasks: **19,360**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## HumanEval-X — Common-Core

Manifest: `humanevalx_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| C++-to-Java | 157 | 1105 | 7.04 | 7 | 1 | 25 |
| C++-to-Python | 157 | 1268 | 8.08 | 7 | 1 | 105 |
| Java-to-C++ | 157 | 1304 | 8.31 | 7 | 1 | 100 |
| Java-to-Python | 157 | 1268 | 8.08 | 7 | 1 | 105 |
| Python-to-C++ | 157 | 1304 | 8.31 | 7 | 1 | 100 |
| Python-to-Java | 157 | 1105 | 7.04 | 7 | 1 | 25 |

Test inputs summed across tasks: **7,354**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## HumanEval-X — Full-Directional

Manifest: `humanevalx_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| C++-to-Java | 161 | 1410 | 8.76 | 7 | 1 | 100 |
| C++-to-Python | 157 | 1268 | 8.08 | 7 | 1 | 105 |
| Java-to-C++ | 161 | 1609 | 9.99 | 7 | 1 | 100 |
| Java-to-Python | 160 | 1285 | 8.03 | 7 | 1 | 105 |
| Python-to-C++ | 157 | 1304 | 8.31 | 7 | 1 | 100 |
| Python-to-Java | 160 | 1120 | 7.00 | 7 | 1 | 25 |

Test inputs summed across tasks: **7,996**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## AVATAR-TC — Common-Core

Manifest: `avatar_tc_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Java-to-Python | 1649 | 1649 | 1.00 | 1 | 1 | 1 |
| Python-to-Java | 1649 | 1649 | 1.00 | 1 | 1 | 1 |

Test inputs summed across tasks: **3,298**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## AVATAR-TC — Full-Directional

Manifest: `avatar_tc_manifest.json`; status: `FROZEN_PREMODEL`.

| Direction | Translation tasks | Test inputs across tasks | Mean tests / task | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Java-to-Python | 1649 | 1649 | 1.00 | 1 | 1 | 1 |
| Python-to-Java | 1649 | 1649 | 1.00 | 1 | 1 | 1 |

Test inputs summed across tasks: **3,298**. This is the number of input executions if each task is evaluated once; repeated Pass@k samples multiply execution load but do not increase the statistical problem count.

## Pre-model statistical precision

These conservative planning estimates were frozen from eligible problem counts before model outcomes were inspected. They treat problem IDs as the sampling clusters and retain related translation directions within a cluster. The 80% power MDE and 95% CI half-width are worst-case bounded paired-difference estimates; actual precision depends on paired model discordance and cross-direction covariance.

| Benchmark | Common-Core problems | Approx. 80% power MDE (pp) | Worst-case 95% CI half-width (pp) |
|---|---:|---:|---:|
| xCodeEval | 332 | 15.37 | 10.76 |
| TransCoder-Test | 321 | 15.63 | 10.94 |
| HumanEval-X | 157 | 22.35 | 15.64 |
| AVATAR-TC | 1649 | 6.9 | 4.83 |
## Interpretation

Pass@1 is scored per translation task: the generated program must pass every eligible test for that task. Pass@5 uses five candidates per task, and Pass-Ratio@5 reports partial test coverage using the same candidates. Candidate samples and individual test assertions do not increase the number of independent problems; statistical analyses cluster related directions by problem ID.

AVATAR-TC provides one recorded test input per task in its frozen manifest, so its large task count should not be mistaken for deep per-problem test coverage. Benchmark results must be interpreted with this coverage limitation and the benchmark-specific test-soundness limitations documented in the final report.
