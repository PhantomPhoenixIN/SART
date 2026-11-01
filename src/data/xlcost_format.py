"""Convert XLCoST's whitespace-tokenized program rows into source text."""

from __future__ import annotations

import re

_MULTI_OPERATORS = {
    ("<", "<", "="): "<<=",
    (">", ">", "="): ">>=",
    ("*", "*", "="): "**=",
    ("+", "+"): "++",
    ("-", "-"): "--",
    ("-", ">"): "->",
    (":", ":"): "::",
    ("=", "="): "==",
    ("!", "="): "!=",
    ("<", "="): "<=",
    (">", "="): ">=",
    ("+", "="): "+=",
    ("-", "="): "-=",
    ("*", "="): "*=",
    ("/", "="): "/=",
    ("%", "="): "%=",
    ("&", "="): "&=",
    ("|", "="): "|=",
    ("^", "="): "^=",
    ("&", "&"): "&&",
    ("|", "|"): "||",
    ("<", "<"): "<<",
    (">", ">"): ">>",
    ("*", "*"): "**",
    ("/", "/"): "//",
}
_NO_SPACE_BEFORE = {".", ",", ";", ")", "]", "}", "::", "->", "++", "--"}
_NO_SPACE_AFTER = {".", "(", "[", "{", "::", "->", "#include"}


def _merge_operators(tokens: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 3 <= len(tokens) and tuple(tokens[index : index + 3]) in _MULTI_OPERATORS:
            merged.append(_MULTI_OPERATORS[tuple(tokens[index : index + 3])])
            index += 3
        elif index + 2 <= len(tokens) and tuple(tokens[index : index + 2]) in _MULTI_OPERATORS:
            merged.append(_MULTI_OPERATORS[tuple(tokens[index : index + 2])])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def _format_code_line(tokens: list[str], language: str) -> str:
    if not tokens:
        return ""
    tokens = _merge_operators(tokens)
    output = tokens[0]
    for token in tokens[1:]:
        if token in _NO_SPACE_BEFORE or output.endswith(tuple(_NO_SPACE_AFTER)):
            output += token
        else:
            output += " " + token
    output = re.sub(r"#\s+include\b", "#include", output)
    output = re.sub(r"(?<!\w)@\s+", "@", output) if language == "Python" else output
    if language in {"C++", "Java"}:
        output = re.sub(r"\s*::\s*", "::", output)
        output = re.sub(r"\s*->\s*", "->", output)
    if language == "C++":
        include = re.match(r'^(#include\s*[<"])(.*?)([>"])\s*$', output)
        if include:
            header = re.sub(r"\s*([/+.])\s*", r"\1", include.group(2).strip())
            output = include.group(1) + header + include.group(3)
    return output


def detokenize_xlcost(code: str, language: str) -> str:
    """Restore line/indent markers and common operator tokens from one XLCoST row.

    XLCoST's release is already tokenized; whitespace in string literals is
    represented partly with ``▁``. The transformation restores explicit marker
    tokens and lexical operators. Ambiguous original formatting cannot be
    recovered exactly and is audited via parser/round-trip coverage.
    """
    if language not in {"Python", "Java", "C++"}:
        raise ValueError(f"Unsupported XLCoST language: {language}")
    tokens = code.strip().split()
    lines: list[str] = []
    current: list[str] = []
    indent_level = 0
    quote: str | None = None
    quoted: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if quote is None and language == "Python" and token == "INDENT":
            indent_level += 1
            index += 1
            continue
        if quote is None and language == "Python" and token == "DEDENT":
            indent_level = max(0, indent_level - 1)
            index += 1
            continue
        if quote is None and token == "NEW_LINE":
            lines.append("    " * indent_level + _format_code_line(current, language))
            current = []
            index += 1
            continue

        if quote is not None:
            if token == quote:
                quoted_text = "".join(quoted)
                trailing_slashes = len(quoted_text) - len(quoted_text.rstrip("\\"))
                if trailing_slashes % 2:
                    quoted.append(token)
                else:
                    quoted.append(token)
                    current.append("".join(quoted))
                    quoted = []
                    quote = None
            elif token == "STRNEWLINE":
                quoted.append(r"\n")
            elif token == "TABSYMBOL":
                quoted.append(r"\t")
            elif token == "▁":
                quoted.append(" ")
            else:
                quoted.append(token)
            index += 1
            continue

        if token in {"'", '"', "`"}:
            quote = token
            quoted = [token]
            index += 1
            continue
        if token in {"NEW_LINE", "STRNEWLINE"}:
            # String-newline markers outside a literal are not structural lines.
            current.append(r"\n")
        elif token == "TABSYMBOL":
            current.append(r"\t")
        else:
            current.append(token)
        index += 1

    if quote is not None:
        current.append("".join(quoted))
    if current:
        lines.append("    " * indent_level + _format_code_line(current, language))
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")
