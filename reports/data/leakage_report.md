# Data leakage audit

Date: 2026-10-02
Status: **Internal XLCoST split audit and external benchmark source-code audit complete.**
No model predictions or external benchmark scores were inspected.

## Frozen fingerprint rules

| Level | Rule |
|---|---|
| Exact | Normalize CRLF/CR to LF; SHA256 remaining UTF-8 source unchanged. |
| Lexical | Tree-sitter leaf-token sequence, comments excluded and literals preserved. |
| Identifier normalized | Deterministically rename local-like identifiers by first occurrence. |
| Structural | Hash Tree-sitter AST node-type preorder with comments and leaf values removed. |
| Near duplicate | Token 5-gram MinHash (256 permutations, 64×4 bands), estimated Jaccard ≥0.80 and exact Jaccard ≥0.85; minimum 40 tokens. |

Any same-language match drops the entire XLCoST training `problem_id` across all languages and directions. Validation, test, and external tasks are retained unchanged. The near-duplicate implementation first generates candidates with MinHash bands, then applies the frozen exact 5-gram threshold.

## XLCoST internal split protection

- Parsed implementations: `{"test": 2401, "train": 25122, "validation": 1266}`.
- Training problem IDs removed against XLCoST validation/test: **39**.
- Match records by rule: `{"ast": 91, "exact": 7, "identifier_normalized": 8, "lexical": 7, "near_duplicate": 8}`.
- Detailed IDs, similarities, protected split, and code hashes: `reports/data/xlcost_leakage.json`.

## External benchmark source-code protection

- Parse-valid external programs fingerprinted: `{"AVATAR-TC/Java": 1744, "AVATAR-TC/Python": 1548, "HumanEval-X/C++": 162, "HumanEval-X/Java": 164, "HumanEval-X/Python": 164, "TransCoder-Test/C++": 849, "TransCoder-Test/Java": 855, "TransCoder-Test/Python": 865, "xCodeEval/C++": 1633, "xCodeEval/Java": 1849, "xCodeEval/Python": 1942}`.
- Parser rejections retained as coverage limitations (benchmark/language): `{"AVATAR-TC/Java": 2, "AVATAR-TC/Python": 198, "HumanEval-X/C++": 2, "HumanEval-X/Java": 0, "HumanEval-X/Python": 0, "TransCoder-Test/C++": 19, "TransCoder-Test/Java": 13, "TransCoder-Test/Python": 3, "xCodeEval/C++": 346, "xCodeEval/Java": 0, "xCodeEval/Python": 0}`.
- XLCoST training problem IDs removed: **514**.
- Match records by rule: `{"ast": 107, "identifier_normalized": 173, "lexical": 175, "near_duplicate": 798}`.
- Sources: frozen xCodeEval candidate pool, HumanEval-X prompt+canonical solution, TransCoder-Test test functions (official Python detokenizer), and AVATAR-TC test split source programs.
- Only training rows were filtered. The original data and running pilot sample remain untouched; a new structure-enriched corpus is at `content/datasets/xlcost/structural_external_clean/`.
- Output shard SHA256/row counts and source-manifest SHA256s are in `reports/data/external_leakage_report.json` and `content/datasets/xlcost/structural_external_clean/manifest.json`.
- External benchmark task eligibility is frozen separately after canonical reference execution. This audit does not remove external tasks based on overlap and does not inspect model predictions.

### Filtered training rows by direction

| Direction | Kept rows | Removed rows |
|---|---:|---:|
| C++ → Java | 7730 | 494 |
| C++ → Python | 7224 | 472 |
| Java → C++ | 7730 | 494 |
| Java → Python | 7152 | 486 |
| Python → C++ | 7224 | 472 |
| Python → Java | 7152 | 486 |

## Limitations

Fingerprint coverage is limited to programs accepted by the pinned Tree-sitter parsers and deterministic benchmark reconstruction. Rejected source records are counted above and remain in the benchmark; they are not evidence of no leakage. Exact and lexical rules are robust to parser-independent source overlap only for the representations actually reconstructed. Structural similarity is a conservative node-shape fingerprint and may match unrelated code or miss structurally equivalent programs. Pretraining contamination cannot be ruled out.

## Provenance

- External input artifact manifest hashes: `{"AVATAR-TC": "68cf63950a392f2e8d7e9a27d8a930ded697a149eb46eb079923c2f2ddd4fdfb", "HumanEval-X": "b676dd179bcfaee7634aedaaf2a67f9cb718786487dcaf254d6ca3672eebde79", "TransCoder-Test": "3794a1e521027aa7dd00621fdc944b941cd661699422c6ae84af99d123ec2ffb", "XLCoST_structural_full": "0d05aefb122106bc288d89e0396de12e62b79f12c7990d8de6efd591b78a65be", "xCodeEval": "918695a00c060d3b8fa9a85371cecf1bef5e912a7a2709767310a83f76116440", "xCodeEval_selection": "adf3970efe8a6510a7c58a3b4eab01b140c6f726de9429bd7afe74d7f22af287"}`.
- Training output manifest status: `COMPLETE`.
