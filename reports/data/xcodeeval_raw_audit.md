# xCodeEval raw test split audit

Status: **raw source/test-database audit plus deterministic one-source selection; not an eligibility manifest.** Parser checks and source selection are complete; canonical compile/run/test preflight and repeatability screening remain required before freezing evaluation tasks. See `xcodeeval_reference_selection.md`.

Dataset revision: `aa09386604075a3e80bcdb315dd68bcc117dedc0`  
Dataset: `NTU-NLP-sg/xCodeEval`, `code_translation/test`  
Training shards were not downloaded.

## Raw rows and source IDs

| Language | Raw accepted-status rows | Unique src_uid | Problems with multiple source rows | Distinct-problem test inputs (sum; mean; median; min; max) |
|---|---:|---:|---:|---:|
| Python | 1942 | 914 | 886 | 53486; mean 58.52; median 49; min 2; max 398 |
| Java | 1849 | 1022 | 93 | 60522; mean 59.22; median 50; min 2; max 398 |
| C++ | 1981 | 1054 | 26 | 62696; mean 59.48; median 50; min 2; max 398 |

## Cross-language candidate sets

- Raw union: 1054 distinct problem IDs; `62696; mean 59.48; median 50; min 2; max 398` hidden tests.
- Raw three-language intersection: 907 distinct IDs; `53168; mean 58.62; median 49; min 2; max 398` hidden tests.
- Common-core directional tasks before eligibility filtering: 5,442 (six directions per problem).
- Common-core target-test executions before eligibility filtering: 319,008 across six translations per problem.
- Pairwise intersections are reported as raw candidate counts below; they are not frozen eligibility counts.

| Direction | Shared problem IDs | Translation tasks | Distinct target test inputs | Mean/problem | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python → Java | 907 | 907 | 53168 | 58.62 | 49 | 2 | 398 |
| Python → C++ | 914 | 914 | 53486 | 58.52 | 49 | 2 | 398 |
| Java → Python | 907 | 907 | 53168 | 58.62 | 49 | 2 | 398 |
| Java → C++ | 1022 | 1022 | 60522 | 59.22 | 50 | 2 | 398 |
| C++ → Python | 914 | 914 | 53486 | 58.52 | 49 | 2 | 398 |
| C++ → Java | 1022 | 1022 | 60522 | 59.22 | 50 | 2 | 398 |

`exec_outcome=PASSED` is the upstream recorded outcome, not a substitute for our pinned-runtime repeated canonical test execution. The mean/median/min/max values above count unit-test inputs in the official `unittest_db.json`; they do not yet account for tasks excluded after preflight.

Raw project file hashes are stored in `artifact_manifest.json` beside the downloaded data.
