from __future__ import annotations

import pytest

from src.data.translation_protocol import extract_generated_code, serialize_prompt


def test_prompt_preserves_source_and_uses_shared_markers() -> None:
    source = "print('x')\r\n"
    prompt = serialize_prompt("Python", "Java", source)
    assert "from Python to Java" in prompt
    assert "### TARGET INTERFACE (Java)\n\n\n" in prompt
    assert "print('x')\n### END SOURCE PROGRAM" in prompt
    assert prompt.endswith("### TARGET PROGRAM (Java)\n")


def test_invalid_same_language_translation_is_rejected() -> None:
    with pytest.raises(ValueError):
        serialize_prompt("Python", "Python", "pass\n")


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("before\n```python\nprint(1)\n```", "before\n```python\nprint(1)\n```"),
        ("```\nline1\r\nline2\n```", "line1\nline2"),
        ("plain\rreturn 1", "plain\nreturn 1"),
    ],
)
def test_output_cleanup_is_frozen_and_minimal(generated: str, expected: str) -> None:
    assert extract_generated_code(generated) == expected
