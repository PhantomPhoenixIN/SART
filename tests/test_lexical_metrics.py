"""Smoke tests for the non-executable validation metric bundle."""

import pytest

from src.evaluation.lexical_metrics import code_tokens, compute_lexical_metrics


@pytest.mark.parametrize(
    ("language", "code"),
    [
        ("Python", "def f(x):\n    # comment\n    return x + 1\n"),
        ("Java", "class A { // comment\n int f(int x) { return x + 1; } }"),
        ("C++", "int f(int x) { /* comment */ return x + 1; }"),
    ],
)
def test_language_lexers_strip_comments_and_whitespace(language: str, code: str) -> None:
    tokens = code_tokens(code, language)
    assert tokens
    assert "comment" not in tokens
    assert all(token.strip() == token for token in tokens)


@pytest.mark.parametrize(
    ("language", "reference"),
    [
        ("Python", "def f(x): return x + 1"),
        ("Java", "class A { int f(int x) { return x + 1; } }"),
        ("C++", "int f(int x) { return x + 1; }"),
    ],
)
def test_exact_program_gets_perfect_lexical_bundle_when_no_ngrams_ignored(
    language: str, reference: str
) -> None:
    scores = compute_lexical_metrics([reference], [reference], language, crystal_k=0)
    assert scores["bleu4"] == pytest.approx(1.0)
    assert scores["codebleu"] == pytest.approx(1.0)
    assert scores["exact_match"] == pytest.approx(1.0)
    assert scores["crystalbleu"] == pytest.approx(1.0)


def test_crystalbleu_uses_full_direction_reference_corpus_for_common_ngrams() -> None:
    reference = "def f():\n    return " + " + ".join(f"value_{index}" for index in range(600))
    scores = compute_lexical_metrics(
        [reference],
        [reference],
        "Python",
        crystal_k=500,
        crystal_reference_corpus=[reference],
    )
    assert scores["crystalbleu"] == pytest.approx(1.0)
