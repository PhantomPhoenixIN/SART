"""Frozen SART code-translation prompt and target serialization."""

from __future__ import annotations

import re

LANGUAGES = {"Python", "Java", "C++"}
_FENCED_BLOCK = re.compile(
    r"\A```[^\r\n]*\r?\n(?P<content>.*)(?:\r?\n)```[ \t]*\Z",
    flags=re.DOTALL,
)


def normalize_line_endings(text: str) -> str:
    """Normalize CRLF and CR to LF without any other cleanup."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def serialize_prompt(
    source_language: str,
    target_language: str,
    source_code: str,
    target_interface: str = "",
) -> str:
    """Serialize one translation request using frozen protocol v1."""
    if source_language not in LANGUAGES or target_language not in LANGUAGES:
        raise ValueError("source and target languages must be Python, Java, or C++")
    if source_language == target_language:
        raise ValueError("translation languages must differ")
    source_code = normalize_line_endings(source_code)
    source_block = source_code if source_code.endswith("\n") else source_code + "\n"
    target_interface = normalize_line_endings(target_interface)
    return (
        f"Translate the complete source program from {source_language} to {target_language}.\n"
        "Preserve its behavior. Return only the translated program.\n\n"
        f"### TARGET INTERFACE ({target_language})\n"
        f"{target_interface}\n\n"
        f"### SOURCE PROGRAM ({source_language})\n"
        f"{source_block}"
        f"### END SOURCE PROGRAM\n\n"
        f"### TARGET PROGRAM ({target_language})\n"
    )


def serialize_target(target_code: str) -> str:
    """Serialize a training target as code only with normalized line endings."""
    return normalize_line_endings(target_code)


def extract_generated_code(generated: str) -> str:
    """Apply the frozen architecture-independent output cleanup policy."""
    normalized = normalize_line_endings(generated)
    match = _FENCED_BLOCK.fullmatch(normalized)
    return match.group("content") if match else normalized
