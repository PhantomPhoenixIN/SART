"""Deterministic code normalizations and fingerprints for leakage analysis."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.parsing.tree_sitter_adapter import parse_tree

COMMENT_TYPES = {"comment", "line_comment", "block_comment"}
IDENTIFIER_TYPES = {"identifier"}
PRESERVE_IDENTIFIER_PARENTS = {
    "import_statement",
    "import_from_statement",
    "package_declaration",
    "import_declaration",
    "scoped_identifier",
    "field_access",
    "attribute",
}
LITERAL_SUFFIXES = ("string", "string_literal", "character_literal", "integer", "float")
_MERSENNE_61 = np.uint64((1 << 61) - 1)


@dataclass(frozen=True)
class Fingerprints:
    """Frozen exact, lexical, identifier-normalized, and AST fingerprints."""

    exact: str
    lexical: str
    identifier_normalized: str
    ast: str
    lexical_tokens: tuple[str, ...]


def sha256_text(text: str) -> str:
    """Hash UTF-8 text with SHA256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_line_endings(source: str) -> str:
    """Apply the frozen exact-code normalization: normalize line endings only."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _leaf_nodes(root) -> Iterator:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.child_count == 0:
            yield node
        else:
            stack.extend(reversed(node.children))


def _node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="strict")


def lexical_tokens(source: str, language: str) -> tuple[str, ...]:
    """Return comment-free Tree-sitter leaf tokens while preserving literal contents."""
    source_bytes = source.encode("utf-8")
    root = parse_tree(source, language).root_node
    return tuple(
        _node_text(source_bytes, node)
        for node in _leaf_nodes(root)
        if node.type not in COMMENT_TYPES and _node_text(source_bytes, node).strip()
    )


def identifier_normalized_tokens(source: str, language: str) -> tuple[str, ...]:
    """Rename local-like identifiers by first occurrence and preserve APIs/types heuristically.

    Tree-sitter exposes dedicated type/field node kinds in Java and C++. Plain identifiers
    attached to imports, packages, attributes, and member access are also preserved. The rule is
    deterministic and frozen before any model evaluation.
    """
    source_bytes = source.encode("utf-8")
    root = parse_tree(source, language).root_node
    names: dict[str, str] = {}
    result: list[str] = []
    for node in _leaf_nodes(root):
        text = _node_text(source_bytes, node)
        if node.type in COMMENT_TYPES or not text.strip():
            continue
        parent_type = node.parent.type if node.parent is not None else ""
        preserve = (
            node.type not in IDENTIFIER_TYPES
            or parent_type in PRESERVE_IDENTIFIER_PARENTS
            or (text[:1].isupper() and text.replace("_", "").isalnum())
        )
        if preserve:
            result.append(text)
        else:
            result.append(names.setdefault(text, f"ID_{len(names)}"))
    return tuple(result)


def normalized_ast_tokens(source: str, language: str) -> tuple[str, ...]:
    """Return a structure-only preorder AST representation without comments or leaf values."""
    root = parse_tree(source, language).root_node
    output: list[str] = []
    stack = [(root, False)]
    while stack:
        node, closing = stack.pop()
        if node.type in COMMENT_TYPES:
            continue
        if closing:
            output.append(")")
            continue
        output.extend(("(", node.type))
        stack.append((node, True))
        for child in reversed(node.children):
            stack.append((child, False))
    return tuple(output)


def fingerprints(source: str, language: str) -> Fingerprints:
    """Compute every frozen exact fingerprint for one valid program."""
    exact_source = normalize_line_endings(source)
    lexical = lexical_tokens(exact_source, language)
    identifiers = identifier_normalized_tokens(exact_source, language)
    ast = normalized_ast_tokens(exact_source, language)
    return Fingerprints(
        exact=sha256_text(exact_source),
        lexical=sha256_text("\0".join(lexical)),
        identifier_normalized=sha256_text("\0".join(identifiers)),
        ast=sha256_text("\0".join(ast)),
        lexical_tokens=lexical,
    )


def token_ngram_hashes(tokens: tuple[str, ...], n: int = 5) -> np.ndarray:
    """Return stable unique 61-bit hashes for token n-grams."""
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    values = {
        int.from_bytes(
            hashlib.blake2b("\0".join(tokens[i : i + n]).encode(), digest_size=8).digest(),
            "little",
        )
        & ((1 << 61) - 1)
        for i in range(len(tokens) - n + 1)
    }
    return np.fromiter(sorted(values), dtype=np.uint64)


def minhash_signature(hashes: np.ndarray, permutations: int = 256, seed: int = 42) -> np.ndarray:
    """Compute a deterministic universal-hash MinHash signature."""
    if hashes.size == 0:
        return np.full(permutations, _MERSENNE_61, dtype=np.uint64)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_MERSENNE_61), size=permutations, dtype=np.uint64)
    b = rng.integers(0, int(_MERSENNE_61), size=permutations, dtype=np.uint64)
    signature = np.full(permutations, _MERSENNE_61, dtype=np.uint64)
    # Small chunks bound temporary memory for long programs.
    for start in range(0, hashes.size, 256):
        values = hashes[start : start + 256]
        transformed = (a[:, None] * values[None, :] + b[:, None]) % _MERSENNE_61
        signature = np.minimum(signature, transformed.min(axis=1))
    return signature


def exact_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate exact Jaccard similarity between sorted unique n-gram hashes."""
    if first.size == 0 and second.size == 0:
        return 1.0
    intersection = np.intersect1d(first, second, assume_unique=True).size
    union = first.size + second.size - intersection
    return float(intersection / union) if union else 1.0
