from __future__ import annotations

import pytest

from src.regions.alignment import align_token_offsets
from src.regions.extract import extract_region_graph

EXAMPLES = {
    "Python": """class Counter:
    def run(self, values):
        total = 0
        for i in values:
            while total < 10:
                if i % 2 == 0:
                    total += values[i]
                    continue
                else:
                    break
        try:
            return helper(total)
        except ValueError:
            raise
""",
    "Java": """class Counter {
    int run(int[] values) {
        int total = 0;
        for (int i = 0; i < values.length; i++) {
            while (total < 10) {
                if (i % 2 == 0) { total += values[i]; continue; }
                else { break; }
            }
        }
        try { return helper(total); } catch (RuntimeException e) { throw e; }
    }
    int helper(int x) { return x; }
}
""",
    "C++": """class Counter {
public:
    int run(int *values) {
        int total = 0;
        for (int i = 0; i < 5; ++i) {
            while (total < 10) {
                if (i % 2 == 0) { total += values[i]; continue; }
                else { break; }
            }
        }
        try { return helper(total); } catch (...) { throw; }
    }
    int helper(int x) { return x; }
};
""",
}


@pytest.mark.parametrize("language", ["Python", "Java", "C++"])
def test_core_region_ontology_and_ast_edges(language: str) -> None:
    graph = extract_region_graph(EXAMPLES[language], language)
    types = {region.region_type for region in graph.regions}
    expected = {
        "PROGRAM",
        "FUNCTION",
        "PARAMETER",
        "CONDITIONAL",
        "LOOP",
        "ASSIGNMENT",
        "MUTATION",
        "CALL",
        "INDEX_ACCESS",
        "ARITHMETIC",
        "COMPARISON",
        "RETURN",
        "BREAK",
        "CONTINUE",
        "EXCEPTION",
    }
    if language != "Python":
        expected.add("DECLARATION")
    assert expected <= types
    assert graph.ast_available
    assert not graph.cfg_available
    assert not graph.dfg_available
    assert len(graph.edges) == 2 * (len(graph.regions) - 1)
    assert all(region.start_byte < region.end_byte for region in graph.regions)
    assert all(region.region_type in types for region in graph.regions)


def test_smallest_enclosing_region_alignment_and_unicode_offsets() -> None:
    source = "def f(x):\n    café = x + 1\n    return café\n"
    graph = extract_region_graph(source, "Python")
    token = "café"
    first = source.index(token)
    second = source.rindex(token)
    assignments = align_token_offsets(
        source,
        [(first, first + len(token)), (second, second + len(token)), (0, 0)],
        graph.regions,
    )
    assert assignments[0] is not None
    assert assignments[1] is not None
    assert assignments[2] is None
    region_by_id = {region.region_id: region for region in graph.regions}
    assert region_by_id[assignments[0]].region_type == "ASSIGNMENT"
    assert region_by_id[assignments[1]].region_type == "RETURN"


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        ("Python", "def f(x):\n    return obj.make(x).value\n", {"CALL", "MEMBER_ACCESS"}),
        (
            "Java",
            "class A { Object f() { return new StringBuilder().append(\"x\"); } }",
            {"CALL", "CONSTRUCTOR"},
        ),
        (
            "C++",
            "struct A { int x; }; int f() { A *a = new A(); return a->x; }",
            {"CONSTRUCTOR", "MEMBER_ACCESS"},
        ),
    ],
)
def test_calls_constructors_and_member_access(
    language: str, source: str, expected: set[str]
) -> None:
    types = {region.region_type for region in extract_region_graph(source, language).regions}
    assert expected <= types
