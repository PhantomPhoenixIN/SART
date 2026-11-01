"""Reference-based code metrics used only when execution tests are unavailable."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from codebleu import calc_codebleu
from crystalbleu import corpus_bleu as crystal_corpus_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from pygments.lexers import CppLexer, JavaLexer, PythonLexer
from pygments.token import Comment, Whitespace

_LEXERS = {
    "Python": PythonLexer,
    "Java": JavaLexer,
    "C++": CppLexer,
}


def code_tokens(code: str, language: str) -> list[str]:
    """Tokenize code with Pygments, excluding comments and whitespace."""
    try:
        lexer = _LEXERS[language]()
    except KeyError as exc:
        raise ValueError(f"Unsupported target language: {language}") from exc
    return [
        value
        for token_type, value in lexer.get_tokens(code)
        if value
        and not value.isspace()
        and token_type not in Whitespace
        and token_type not in Comment
    ]


def compute_lexical_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    language: str,
    *,
    crystal_k: int = 500,
    crystal_reference_corpus: Sequence[str] | None = None,
) -> dict[str, float | int]:
    """Compute BLEU-4, CodeBLEU, exact match and CrystalBLEU for one direction.

    CrystalBLEU's ignored n-grams are estimated from references only, never from
    model hypotheses. The reference corpus is the frozen evaluation split.
    """
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must have the same non-zero length")
    if crystal_k < 0:
        raise ValueError("crystal_k must be non-negative")
    tokenized_references = [code_tokens(code, language) for code in references]
    tokenized_hypotheses = [code_tokens(code, language) for code in hypotheses]
    common_reference_tokens = [
        code_tokens(code, language)
        for code in (
            crystal_reference_corpus if crystal_reference_corpus is not None else references
        )
    ]
    reference_sets = [[tokens] for tokens in tokenized_references]
    bleu4 = corpus_bleu(
        reference_sets,
        tokenized_hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    common_ngrams: Counter[tuple[str, ...]] = Counter()
    for tokens in common_reference_tokens:
        for order in range(1, 5):
            common_ngrams.update(ngrams(tokens, order))
    ignored = dict(common_ngrams.most_common(crystal_k))
    crystal_bleu = crystal_corpus_bleu(
        reference_sets,
        tokenized_hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        ignoring=ignored,
    )
    codebleu = calc_codebleu(
        list(references),
        list(hypotheses),
        {"Python": "python", "Java": "java", "C++": "cpp"}[language],
        tokenizer=lambda code: code_tokens(code, language),
    )
    exact_match = sum(
        reference.replace("\r\n", "\n").replace("\r", "\n")
        == hypothesis.replace("\r\n", "\n").replace("\r", "\n")
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ) / len(references)
    return {
        "bleu4": float(bleu4),
        "codebleu": float(codebleu["codebleu"]),
        "codebleu_ngram": float(codebleu["ngram_match_score"]),
        "codebleu_weighted_ngram": float(codebleu["weighted_ngram_match_score"]),
        "codebleu_syntax": float(codebleu["syntax_match_score"]),
        "codebleu_dataflow": float(codebleu["dataflow_match_score"]),
        "exact_match": float(exact_match),
        "crystalbleu": float(crystal_bleu),
        "examples": len(references),
        "crystal_k": crystal_k,
    }
