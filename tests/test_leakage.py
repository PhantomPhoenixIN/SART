from __future__ import annotations

import numpy as np

from src.data.leakage import (
    exact_jaccard,
    fingerprints,
    minhash_signature,
    token_ngram_hashes,
)


def test_comment_and_local_identifier_normalization() -> None:
    first = "def add(a, b):\n    # note\n    return a + b\n"
    second = "def add(left, right):\n    return left + right\n"
    first_fp = fingerprints(first, "Python")
    second_fp = fingerprints(second, "Python")
    assert first_fp.exact != second_fp.exact
    assert first_fp.lexical != second_fp.lexical
    assert first_fp.identifier_normalized == second_fp.identifier_normalized
    assert first_fp.ast == second_fp.ast


def test_minhash_and_exact_jaccard_are_deterministic() -> None:
    tokens = tuple(f"t{i}" for i in range(20))
    changed = tokens[:-1] + ("different",)
    first = token_ngram_hashes(tokens)
    second = token_ngram_hashes(changed)
    signature = minhash_signature(first)
    assert np.array_equal(signature, minhash_signature(first))
    assert exact_jaccard(first, first) == 1.0
    assert 0.0 < exact_jaccard(first, second) < 1.0
