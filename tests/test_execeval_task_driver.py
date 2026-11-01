from src.evaluation.execeval_task_driver import (
    _java_public_class_name,
    _match_avatar_output,
    _match_output,
)
from src.evaluation.xcodeeval_preflight import (
    _cpp_interface_header,
    run_complete_program_reference,
)


def test_exec_eval_line_comparison_ignores_only_outer_and_line_whitespace() -> None:
    assert _match_output(" 2 \n 4 \n", ["2\n4"])
    assert not _match_output("2 4", ["4 2"])


def test_exec_eval_numeric_token_tolerance_matches_reference_rule() -> None:
    assert _match_output("1.0000000000005", ["1"])
    assert not _match_output("1.00000000001", ["1"])


def test_exec_eval_boolean_tokens_are_case_insensitive() -> None:
    assert _match_output("TRUE No", ["true no"])
    assert _match_output("yes maybe", ["YES maybe"])
    assert not _match_output("maybe", ["MAYBE"])


def test_exec_eval_accepts_any_reference_output_alternative() -> None:
    assert _match_output("second", ["first", "second"])


def test_avatar_comparison_ignores_case_and_whitespace_but_keeps_punctuation() -> None:
    assert _match_avatar_output(" Hello\n WORLD ", ["hello world"])
    assert not _match_avatar_output("hello!", ["hello"])


def test_java_filename_uses_first_public_class_in_one_line_source() -> None:
    source = "public class Main { public class Nested { } }"
    assert _java_public_class_name(source) == "Main"


def test_java_filename_supports_package_private_harness_class() -> None:
    source = "class Harness { public static void main(String[] a) {} }"
    assert _java_public_class_name(source) == "Harness"


def test_python_full_program_harness_keeps_assertions_enabled() -> None:
    source = "def f():\n    return False\nassert f() is True\n"
    result, sandbox = run_complete_program_reference(source, "Python", "Python 3.11", repeats=1)
    assert sandbox.returncode == 0
    assert result is not None
    assert result["status"] == "failed"


def test_cpp_harness_header_exposes_prompt_helpers_and_target() -> None:
    prompt = (
        "#include <string>\nusing namespace std;\n"
        'string helper(){ return "x"; }\nstring target(){\n'
    )
    header = _cpp_interface_header(prompt)
    assert "string helper();" in header
    assert "string target();" in header
    assert ">>>" not in header
