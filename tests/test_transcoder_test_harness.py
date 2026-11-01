from scripts.evaluate.preflight_transcoder_test_references import (
    _extract_f_gold,
    _insert_candidate,
)
from scripts.preprocess.audit_transcoder_test import _case_count


def test_python_reference_harness_inserts_renamed_candidate() -> None:
    source = "def f_gold(x):\n    return x + 1\n#TOFILL\nassert f_filled(2) == f_gold(2)\n"
    candidate = _extract_f_gold(source, "Python")
    filled = _insert_candidate(source, candidate, "Python")
    assert "def f_filled(x):" in filled
    assert "def f_gold(x):" in filled


def test_java_reference_harness_inserts_candidate_inside_class() -> None:
    source = (
        "class Test {\nstatic int f_gold(int x) { return x + 1; }\n"
        "//TOFILL\npublic static void main(String[] a) {}\n}\n"
    )
    candidate = _extract_f_gold(source, "Java")
    filled = _insert_candidate(source, candidate, "Java")
    assert "static int f_filled(int x)" in filled
    assert "static int f_gold(int x)" in filled


def test_cpp_reference_harness_inserts_candidate_before_test_main() -> None:
    source = "int f_gold(int x) { return x + 1; }\n//TOFILL\nint main() {}\n"
    candidate = _extract_f_gold(source, "C++")
    filled = _insert_candidate(source, candidate, "C++")
    assert "int f_filled(int x)" in filled
    assert "int f_gold(int x)" in filled


def test_insert_candidate_preserves_backslashes_in_generated_code() -> None:
    source = "class Test {\n//TOFILL\n}\n"
    candidate = r'static String f_filled() { return "\\u1234" + "\\1"; }'
    filled = _insert_candidate(source, candidate, "Java")
    assert candidate in filled


def test_case_count_supports_java_arrays_as_list(tmp_path) -> None:
    path = tmp_path / "case.java"
    path.write_text('List<String> param0 = Arrays.asList("a,b", "c", "d");')
    assert _case_count(path, "java") == 3


def test_case_count_supports_nested_cpp_vector_initializers(tmp_path) -> None:
    path = tmp_path / "case.cpp"
    path.write_text(
        "vector<vector<vector<int>>> param0 {{{1,2},{3}},{{4},{5,6}}};"
    )
    assert _case_count(path, "cpp") == 2
