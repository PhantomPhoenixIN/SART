# TransCoder-Test raw executable-coverage audit

Status: **candidate audit only**. Reference implementations have not been canonical-executed or repeated; no task eligibility manifest is frozen and no model predictions exist. Candidate counts below are bounded by the presence/readability of official target-language reference harnesses, not model results.

The pinned official source release provides 868 test-split function IDs per language. A paired translation task is executable only when the target-language `TO_FILL` harness is available and its tests are readable. The official README describes matching function IDs to these harnesses.

## Common-Core candidates

Common-Core candidates have a readable test harness in all three target languages, so the same problem IDs can support all six directions.

- Problems: 327
- Directional tasks: 1962
- Test inputs across each distinct target-language harness (counted once per language): 9810
- Test executions across six translation tasks (each target harness used for both inbound directions): 19620
- Tests per problem across three language harnesses: mean 30.0, median 30, min 30, max 30

## Full-Directional candidates

| Direction | Candidate problems/tasks | Target harness test inputs | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| C++ → Java | 433 | 4330 | 10.00 | 10 | 10 | 10 |
| C++ → Python | 429 | 4292 | 10.00 | 10 | 10 | 11 |
| Java → C++ | 415 | 4150 | 10.00 | 10 | 10 | 10 |
| Java → Python | 429 | 4292 | 10.00 | 10 | 10 | 11 |
| Python → C++ | 415 | 4150 | 10.00 | 10 | 10 | 10 |
| Python → Java | 433 | 4330 | 10.00 | 10 | 10 | 10 |

## Exclusions and limitations

- Raw rows: {'C++': 868, 'Java': 868, 'Python': 868}; shared problem IDs before harness availability: 868.
- Harness files available: {'C++': {'available_harnesses': 415, 'missing_harnesses': 453}, 'Java': {'available_harnesses': 433, 'missing_harnesses': 435}, 'Python': {'available_harnesses': 430, 'missing_harnesses': 438}}.
- Test-count extraction failures: C++ 2, Java 1, Python 0; IDs and reasons are in `transcoder_test_raw_audit.json`.
- Python harness syntax errors: 1; the reported malformed harness is excluded from Python-target candidates.
- Candidate source/target functions still require syntax checks and reference compile/run against all harness tests at least three times under the frozen runtime. Freeze task/test hashes only after that preflight.
- The official evaluator notes test scripts are missing for some functions; missing harnesses are excluded before model inference, consistent with its published eligibility guidance.
