# AVATAR-TC raw coverage audit

Status: **raw audit only**. Reference programs have not been compiled/run; no eligibility manifest is frozen and no model predictions exist.

- Pinned source revision: `c02d3b4ef856c676c3924dd720824e581a30f1a5`.
- The pinned CoTran README documents AVATAR-TC test-case tables but does not state a separate AVATAR-TC license. Treat the license as unknown pending confirmation; retain provenance and do not redistribute.
- Split rows (problem IDs, Java code rows, Python code rows): `{"test": {"java_programs": 1746, "problems": 1746, "python_programs": 1746}, "train": {"java_programs": 55179, "problems": 55179, "python_programs": 55179}, "valid": {"java_programs": 443, "problems": 443, "python_programs": 443}}`.
- Test split unique problems: 1746; IDs joined to documented input/output tables: 1746; unmatched IDs: 0.
- Test inputs: 1746 total; mean 1.00, median 1.0, min 1, max 1 per problem.
- Each documented test-table key has one input/output pair in this release. Each of the two directions has the same candidate problem IDs and target-test count; six-direction common-core is not applicable.

| Direction | Problems/tasks | Target test inputs | Mean/problem | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Java → Python | 1746 | 1746 | 1.00 | 1 | 1 | 1 |
| Python → Java | 1746 | 1746 | 1.00 | 1 | 1 | 1 |

All counts are preflight candidates. Compile/execute canonical Java and Python implementations against the documented test tables at least three times, apply the official output comparison semantics, hash the chosen tests, and freeze the eligible manifest before evaluating models.
