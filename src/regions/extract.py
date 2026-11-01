"""Extract language-normalized semantic regions from pinned Tree-sitter trees."""

from __future__ import annotations

from dataclasses import dataclass

from src.parsing.tree_sitter_adapter import parse_tree

REGION_TYPES = {
    "PROGRAM",
    "FUNCTION",
    "PARAMETER",
    "DECLARATION",
    "CONDITIONAL",
    "LOOP",
    "SWITCH",
    "BRANCH",
    "ASSIGNMENT",
    "MUTATION",
    "CALL",
    "CONSTRUCTOR",
    "MEMBER_ACCESS",
    "INDEX_ACCESS",
    "ARITHMETIC",
    "COMPARISON",
    "BOOLEAN_EXPRESSION",
    "RETURN",
    "BREAK",
    "CONTINUE",
    "EXCEPTION",
    "OTHER_STRUCTURAL",
}

COMMON_TYPES = {
    "if_statement": "CONDITIONAL",
    "conditional_expression": "CONDITIONAL",
    "while_statement": "LOOP",
    "for_statement": "LOOP",
    "do_statement": "LOOP",
    "switch_statement": "SWITCH",
    "switch_expression": "SWITCH",
    "case_statement": "BRANCH",
    "return_statement": "RETURN",
    "break_statement": "BREAK",
    "continue_statement": "CONTINUE",
    "try_statement": "EXCEPTION",
    "throw_statement": "EXCEPTION",
    "catch_clause": "EXCEPTION",
    "call": "CALL",
    "call_expression": "CALL",
    "subscript": "INDEX_ACCESS",
    "array_access": "INDEX_ACCESS",
    "assignment": "ASSIGNMENT",
    "assignment_expression": "ASSIGNMENT",
    "update_expression": "MUTATION",
}

LANGUAGE_TYPES = {
    "Python": {
        "module": "PROGRAM",
        "function_definition": "FUNCTION",
        "lambda": "FUNCTION",
        "parameters": "PARAMETER",
        "default_parameter": "PARAMETER",
        "typed_parameter": "PARAMETER",
        "typed_default_parameter": "PARAMETER",
        "for_statement": "LOOP",
        "match_statement": "SWITCH",
        "case_clause": "BRANCH",
        "augmented_assignment": "MUTATION",
        "attribute": "MEMBER_ACCESS",
        "list_comprehension": "LOOP",
        "set_comprehension": "LOOP",
        "dictionary_comprehension": "LOOP",
        "generator_expression": "LOOP",
        "raise_statement": "EXCEPTION",
        "with_statement": "OTHER_STRUCTURAL",
        "class_definition": "OTHER_STRUCTURAL",
    },
    "Java": {
        "program": "PROGRAM",
        "method_declaration": "FUNCTION",
        "constructor_declaration": "FUNCTION",
        "formal_parameter": "PARAMETER",
        "spread_parameter": "PARAMETER",
        "local_variable_declaration": "DECLARATION",
        "field_declaration": "DECLARATION",
        "enhanced_for_statement": "LOOP",
        "switch_label": "BRANCH",
        "method_invocation": "CALL",
        "object_creation_expression": "CONSTRUCTOR",
        "field_access": "MEMBER_ACCESS",
        "array_access": "INDEX_ACCESS",
        "class_declaration": "OTHER_STRUCTURAL",
        "interface_declaration": "OTHER_STRUCTURAL",
        "synchronized_statement": "OTHER_STRUCTURAL",
    },
    "C++": {
        "translation_unit": "PROGRAM",
        "function_definition": "FUNCTION",
        "parameter_declaration": "PARAMETER",
        "declaration": "DECLARATION",
        "init_declarator": "DECLARATION",
        "new_expression": "CONSTRUCTOR",
        "field_expression": "MEMBER_ACCESS",
        "subscript_expression": "INDEX_ACCESS",
        "lambda_expression": "FUNCTION",
        "class_specifier": "OTHER_STRUCTURAL",
        "struct_specifier": "OTHER_STRUCTURAL",
        "range_based_for_statement": "LOOP",
    },
}

ARITHMETIC_OPERATORS = {"+", "-", "*", "/", "%", "**", "<<", ">>"}
COMPARISON_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}
BOOLEAN_OPERATORS = {"and", "or", "not", "&&", "||", "!"}


@dataclass(frozen=True)
class Region:
    """One meaningful language-normalized source region."""

    region_id: int
    region_type: str
    raw_node_type: str
    start_byte: int
    end_byte: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    parent_id: int | None
    ast_depth: int
    child_count: int


@dataclass(frozen=True)
class RegionEdge:
    """A typed directed relation between two semantic regions."""

    source: int
    target: int
    edge_type: str


@dataclass(frozen=True)
class RegionGraph:
    """A semantic region graph and extraction availability masks."""

    language: str
    regions: tuple[Region, ...]
    edges: tuple[RegionEdge, ...]
    ast_available: bool
    cfg_available: bool
    dfg_available: bool


def _immediate_operator(node, source_bytes: bytes) -> str | None:
    for child in node.children:
        if child.child_count == 0:
            text = source_bytes[child.start_byte : child.end_byte].decode("utf-8")
            if text in ARITHMETIC_OPERATORS | COMPARISON_OPERATORS | BOOLEAN_OPERATORS:
                return text
    return None


def _classify(node, language: str, source_bytes: bytes) -> str | None:
    mapped = LANGUAGE_TYPES[language].get(node.type) or COMMON_TYPES.get(node.type)
    if mapped is not None:
        return mapped
    if node.type in {
        "binary_expression",
        "binary_operator",
        "comparison_operator",
        "boolean_operator",
    }:
        operator = _immediate_operator(node, source_bytes)
        if operator in COMPARISON_OPERATORS:
            return "COMPARISON"
        if operator in BOOLEAN_OPERATORS or node.type == "boolean_operator":
            return "BOOLEAN_EXPRESSION"
        if operator in ARITHMETIC_OPERATORS:
            return "ARITHMETIC"
    if node.type in {"unary_expression", "unary_operator", "not_operator"}:
        operator = _immediate_operator(node, source_bytes)
        if operator in BOOLEAN_OPERATORS or node.type == "not_operator":
            return "BOOLEAN_EXPRESSION"
    return None


def extract_region_graph(source: str, language: str) -> RegionGraph:
    """Extract meaningful nested regions and bidirectional AST containment edges."""
    if language not in LANGUAGE_TYPES:
        raise ValueError(f"Unsupported language: {language}")
    tree = parse_tree(source, language)
    if tree.root_node.has_error:
        raise ValueError("Region extraction requires syntax-valid source code")
    source_bytes = source.encode("utf-8")
    mutable_regions: list[dict] = []
    stack = [(tree.root_node, None, 0)]
    while stack:
        node, enclosing_region, syntax_depth = stack.pop()
        region_type = _classify(node, language, source_bytes)
        current_parent = enclosing_region
        if region_type is not None:
            region_id = len(mutable_regions)
            mutable_regions.append(
                {
                    "region_id": region_id,
                    "region_type": region_type,
                    "raw_node_type": node.type,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "start_point": tuple(node.start_point),
                    "end_point": tuple(node.end_point),
                    "parent_id": enclosing_region,
                    "ast_depth": syntax_depth,
                    "child_count": 0,
                }
            )
            if enclosing_region is not None:
                mutable_regions[enclosing_region]["child_count"] += 1
            current_parent = region_id
        for child in reversed(node.named_children):
            stack.append((child, current_parent, syntax_depth + 1))

    regions = tuple(Region(**region) for region in mutable_regions)
    edges: list[RegionEdge] = []
    for region in regions:
        if region.parent_id is not None:
            edges.append(RegionEdge(region.parent_id, region.region_id, "AST_CHILD"))
            edges.append(RegionEdge(region.region_id, region.parent_id, "AST_PARENT"))
    return RegionGraph(
        language=language,
        regions=regions,
        edges=tuple(edges),
        ast_available=True,
        cfg_available=False,
        dfg_available=False,
    )
