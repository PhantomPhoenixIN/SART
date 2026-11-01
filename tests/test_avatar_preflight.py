from scripts.evaluate.preflight_avatar_tc_references import _output_options


def test_avatar_single_expected_output_becomes_one_option() -> None:
    assert _output_options("answer") == ["answer"]


def test_avatar_accepted_output_alternatives_are_flattened() -> None:
    assert _output_options(["first", "second"]) == ["first", "second"]
