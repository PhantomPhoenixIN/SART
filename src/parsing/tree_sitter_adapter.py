"""Small Tree-sitter syntax adapter for the three XLCoST languages."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ParseResult:
    """Syntax status and a compact parser diagnostic."""

    valid: bool
    error_count: int
    root_type: str


@lru_cache(maxsize=3)
def _parser(language: str):
    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise RuntimeError("Install the pinned SART Tree-sitter dependencies in env/") from exc

    if language == "Python":
        import tree_sitter_python as grammar
    elif language == "Java":
        import tree_sitter_java as grammar
    elif language == "C++":
        import tree_sitter_cpp as grammar
    else:
        raise ValueError(f"Unsupported language: {language}")

    parser = Parser(Language(grammar.language()))
    return parser


def parse_program(source: str, language: str) -> ParseResult:
    """Parse source and count error or missing syntax nodes."""
    tree = _parser(language).parse(source.encode("utf-8"))
    errors = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            errors += 1
        stack.extend(node.children)
    return ParseResult(not tree.root_node.has_error and errors == 0, errors, tree.root_node.type)


def parse_tree(source: str, language: str):
    """Return a Tree-sitter tree using the project's pinned language grammar."""
    return _parser(language).parse(source.encode("utf-8"))
