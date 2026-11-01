# XLCoST program-level raw data audit

Status: **raw audit plus completed syntax-filtered aligned-pair construction; tokenizer length, structural-region, and leakage audits remain open.** No external predictions were inspected.

Source archive SHA256: `2b822d82197c4ca159a383854f8a3d187f24f970e6209e77641e709220bf2f2b`  
Extracted 36 files from `generation/pair_data_tok_full` only.

## Pair counts and length distributions

Each undirected parallel pair provides its two directed translation tasks. Length counts below are whitespace-separated pretokenized XLCoST tokens; they are not the selected model tokenizer's subword lengths.

| Parallel pair | Split | Programs | First-language tokens | Second-language tokens |
|---|---|---:|---|---|
| Java ↔ Python | train | 8991 | mean 226.7, median 200, min 12, max 1700 | mean 195.1, median 169, min 16, max 1664 |
| Java ↔ Python | validation | 471 | mean 223.9, median 195, min 27, max 1185 | mean 195.6, median 171, min 24, max 1215 |
| Java ↔ Python | test | 882 | mean 217.3, median 186, min 38, max 1011 | mean 184.3, median 156, min 23, max 810 |
| Java ↔ C++ | train | 9450 | mean 233.3, median 204, min 12, max 1906 | mean 208.9, median 183, min 13, max 1685 |
| Java ↔ C++ | validation | 490 | mean 224.3, median 194.5, min 20, max 1185 | mean 203.0, median 177, min 22, max 1175 |
| Java ↔ C++ | test | 901 | mean 219.3, median 190, min 41, max 1011 | mean 194.7, median 170, min 32, max 867 |
| C++ ↔ Python | train | 9139 | mean 206.0, median 182, min 13, max 1166 | mean 197.7, median 171, min 22, max 1300 |
| C++ ↔ Python | validation | 468 | mean 202.0, median 177.5, min 26, max 1175 | mean 195.6, median 171, min 24, max 1215 |
| C++ ↔ Python | test | 878 | mean 193.4, median 169, min 32, max 867 | mean 184.8, median 156, min 23, max 810 |

## Problem-level split audit

- Distinct program IDs across all pair folders/splits: 11234.
- IDs assigned to more than one official split across languages/pairs: 0.
- Pair-folder train/validation/test overlaps: `{"C++-Python": {"train/test": 0, "train/validation": 0, "validation/test": 0}, "Java-C++": {"train/test": 0, "train/validation": 0, "validation/test": 0}, "Java-Python": {"train/test": 0, "train/validation": 0, "validation/test": 0}}`.
- A nonzero problem-level overlap requires creating and freezing a grouped clean split before training; never preserve cross-language sibling programs across partitions.

## Exact duplicate audit (pretokens only)

- Exact whitespace-canonical duplicates within partitions, counted over selected source files and pair folders: `{"C++": {"test": 5, "train": 71, "validation": 2}, "Java": {"test": 6, "train": 70, "validation": 5}, "Python": {"test": 6, "train": 66, "validation": 2}}`.
- Exact hash intersections across partitions: `{"C++": {"train_test": 3, "train_validation": 0, "validation_test": 2}, "Java": {"train_test": 1, "train_validation": 1, "validation_test": 2}, "Python": {"train_test": 1, "train_validation": 1, "validation_test": 2}}`.
- These are preliminary exact checks. Comment-stripped, identifier-normalized, AST, and near-duplicate checks still need implementation after detokenization/parser validation.

## Directional counts

| Direction | Train | Validation | Test |
|---|---:|---:|---:|
| Java → Python | 8991 | 471 | 882 |
| Python → Java | 8991 | 471 | 882 |
| Java → C++ | 9450 | 490 | 901 |
| C++ → Java | 9450 | 490 | 901 |
| C++ → Python | 9139 | 468 | 878 |
| Python → C++ | 9139 | 468 | 878 |

## Parsing and representation

The official program-level files are whitespace-tokenized text; Python rows include `NEW_LINE`, `INDENT`, and `DEDENT` markers. The current deterministic formatting adapter restores these markers and common lexical operators. Python 3 AST parse coverage is:

| Parallel pair | Split | Parse-valid Python programs | Parse errors | Parse rate |
|---|---|---:|---:|---:|
| Java-Python | train | 7994 | 997 | 88.91% |
| Java-Python | validation | 423 | 48 | 89.81% |
| Java-Python | test | 801 | 81 | 90.82% |
| C++-Python | train | 8132 | 1007 | 88.98% |
| C++-Python | validation | 420 | 48 | 89.74% |
| C++-Python | test | 797 | 81 | 90.77% |

The syntax-filtered build used Tree-sitter 0.23.2 with the pinned Python 0.23.6, Java 0.23.5, and C++ 0.23.4 grammars. It rejects an official aligned pair if either side contains a Tree-sitter syntax error and applies the resulting eligibility decision symmetrically to both directions. It did not repair source text or inspect model predictions.

| Parallel pair | Split | Aligned | Eligible | Parse exclusions | Retention |
|---|---|---:|---:|---:|---:|
| Java ↔ Python | train | 8991 | 7679 | 1312 | 85.41% |
| Java ↔ Python | validation | 471 | 393 | 78 | 83.44% |
| Java ↔ Python | test | 882 | 759 | 123 | 86.05% |
| Java ↔ C++ | train | 9450 | 8266 | 1184 | 87.47% |
| Java ↔ C++ | validation | 490 | 420 | 70 | 85.71% |
| Java ↔ C++ | test | 901 | 794 | 107 | 88.12% |
| C++ ↔ Python | train | 9139 | 7737 | 1402 | 84.66% |
| C++ ↔ Python | validation | 468 | 390 | 78 | 83.33% |
| C++ ↔ Python | test | 878 | 756 | 122 | 86.10% |

The atomic JSONL outputs and their SHA256 hashes are recorded in `content/datasets/xlcost/parsed_program_level/manifest.json`. Detailed exclusions retain problem IDs, affected languages, parser error counts, and root-node types.

The archive contains no explicit data license/terms file; the dataset license remains unverified and the archive must not be redistributed.
