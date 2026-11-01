"""Tests that graph-family ablations mask unrelated graph statistics."""

from src.training.data import _region_type_id, _structural_scalars

REGION = {
    "ast_depth": 8,
    "child_count": 3,
    "cfg_degree": 4,
    "dfg_degree": 5,
}


def test_ast_ablation_masks_cfg_and_dfg_statistics() -> None:
    features = _structural_scalars(REGION, "ast_sart")
    assert features == {
        "ast_depth": 0.25,
        "child_count": 3.0,
        "cfg_degree": 0.0,
        "dfg_degree": 0.0,
    }


def test_cfg_ablation_masks_ast_and_dfg_statistics() -> None:
    features = _structural_scalars(REGION, "cfg_sart")
    assert features == {
        "ast_depth": 0.0,
        "child_count": 0.0,
        "cfg_degree": 4.0,
        "dfg_degree": 0.0,
    }


def test_full_graph_keeps_all_structural_statistics() -> None:
    features = _structural_scalars(REGION, "full_sart")
    assert features == {
        "ast_depth": 0.25,
        "child_count": 3.0,
        "cfg_degree": 4.0,
        "dfg_degree": 5.0,
    }


def test_no_scalar_ablation_masks_every_graph_statistic() -> None:
    features = _structural_scalars(REGION, "sart_no_scalar")
    assert set(features.values()) == {0.0}


def test_coarse_region_ontology_merges_related_types() -> None:
    assert _region_type_id("LOOP", "sart_coarse_ontology") == _region_type_id(
        "CONDITIONAL", "sart_coarse_ontology"
    )
    assert _region_type_id("CONSTRUCTOR", "sart_coarse_ontology") == _region_type_id(
        "CALL", "sart_coarse_ontology"
    )
    assert _region_type_id("LOOP", "full_sart") != _region_type_id("CONDITIONAL", "full_sart")
