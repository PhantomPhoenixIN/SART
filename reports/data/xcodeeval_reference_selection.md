# xCodeEval pre-model reference selection

Status: **deterministic accepted-source selection complete; canonical compile/run/test preflight and eligibility freeze pending**. No trained-model output was inspected. The selected candidates are not yet the final evaluation denominator.

- Dataset: `NTU-NLP-sg/xCodeEval`, official pinned test revision `aa09386604075a3e80bcdb315dd68bcc117dedc0`.
- Selector: `scripts/evaluate/select_xcodeeval_references.py`.
- Manifest: `data/manifests/xcodeeval_reference_selection.json`.
- Manifest SHA256: `adf3970efe8a6510a7c58a3b4eab01b140c6f726de9429bd7afe74d7f22af287`.
- Per source language, the selector kept only upstream `exec_outcome=PASSED` submissions, required parser acceptance, then selected one submission per problem closest to the lower-median Qwen2.5-Coder-7B source token length. Stable `code_uid` is the tie breaker. This prevents multiple accepted submissions from inflating the problem count.
- Tree-sitter is primary. Python 2 uses the Python-2-compatible grammar, with private CPython 2.7 compile fallback. C++ Tree-sitter rejects are checked after the pinned ExecEval sanitizer with GCC 12.3 compile-only in the namespace sandbox.

## Candidate counts and parser paths

| Language | Accepted rows | Parse-valid rows | Selected problems | Parse path counts |
|---|---:|---:|---:|---|
| Python | 1,942 | 1,942 | 914 | Tree-sitter 1,261; Python-2 grammar 671; CPython2 fallback 10 |
| Java | 1,849 | 1,849 | 1,022 | Tree-sitter 1,849 |
| C++ | 1,981 | 1,979 | 1,054 | Tree-sitter 1,633; ExecEval-sanitized GCC12 fallback 346; 2 rejected |

The manifest retains every accepted, parser-valid candidate in deterministic order by distance to the lower-median token count, then `code_uid`. Preflight tests candidates in that order and selects the first reference that compiles and passes every hidden input across three stable repetitions. The two C++ rejects are explicitly retained in the manifest with compiler diagnostics. The selected target runtime-label distribution is Python: Python 2 273, PyPy 2 46, Python 3 351, PyPy 3 231, PyPy 3-64 13; Java: Java 6 143, Java 7 169, Java 8 594, Java 11 114, Java 17 2; C++ labels are recorded per selected `src_uid` in the manifest.

## Common-Core and Full-Directional provisional coverage

Common-Core currently contains 907 shared problem IDs and 5,442 directional tasks before reference execution. It represents 53,168 distinct hidden inputs per direction; across all six directions, this is 319,008 task-level test executions before repeated reliability runs.

| Direction | Candidate problem IDs | Tasks | Hidden inputs across tasks | Mean/problem | Median/problem | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python → Java | 907 | 907 | 53,168 | 58.62 | 49 | 2 | 398 |
| Python → C++ | 914 | 914 | 53,486 | 58.52 | 49 | 2 | 398 |
| Java → Python | 907 | 907 | 53,168 | 58.62 | 49 | 2 | 398 |
| Java → C++ | 1,022 | 1,022 | 60,522 | 59.22 | 50 | 2 | 398 |
| C++ → Python | 914 | 914 | 53,486 | 58.52 | 49 | 2 | 398 |
| C++ → Java | 1,022 | 1,022 | 60,522 | 59.22 | 50 | 2 | 398 |

The Full-Directional set above is provisional and must remain separate from Common-Core. Canonical target references must compile/run and pass every hidden input, with at least three deterministic executions where practical, before frozen Common-Core and Full-Directional manifests and pre-result power/precision estimates are generated.
