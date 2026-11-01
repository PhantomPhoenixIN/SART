# HumanEval-X raw functional coverage audit

Status: **raw audit only**. Canonical solutions have not yet been executed repeatedly; no eligible manifest is frozen and no model predictions exist.

- Dataset revision: `62c78627f3072a1454fa0cb0184737cafe5e4198`.
- Unique aligned problems: 164.
- Translation tasks: 984 across six directions.
- Distinct test assertions across the three language harnesses: 4639.
- Target-harness test executions across all six directions: 9278.

| Direction | Problems/tasks | Total target test inputs | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Python → Java | 164 | 1425 | 8.69 | 7.0 | 1 | 100 |
| Python → C++ | 164 | 1624 | 9.90 | 7.0 | 1 | 100 |
| Java → Python | 164 | 1590 | 9.70 | 7.0 | 1 | 105 |
| Java → C++ | 164 | 1624 | 9.90 | 7.0 | 1 | 100 |
| C++ → Python | 164 | 1590 | 9.70 | 7.0 | 1 | 105 |
| C++ → Java | 164 | 1425 | 8.69 | 7.0 | 1 | 100 |

Test inputs are counted from each language's official executable harness; Python assertions, Java expected-result vector entries and C++ assertion calls are the respective test-case units. The pinned CodeGeeX evaluator and runtime must be used or matched. Repeat canonical execution at least three times, freeze exclusions and test hashes before any model inference, and cluster comparisons by underlying task ID.
