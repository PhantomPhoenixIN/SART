from scripts.preprocess.merge_joern_edges import merge_row


def test_merge_adds_only_available_relations_and_updates_region_degrees():
    row = {
        "source_language": "Java",
        "source_sha256": "abc",
        "structure": {
            "regions": [
                {"region_id": 0, "region_type": "PROGRAM", "cfg_degree": 0, "dfg_degree": 0},
                {"region_id": 1, "region_type": "FUNCTION", "cfg_degree": 0, "dfg_degree": 0},
                {"region_id": 2, "region_type": "RETURN", "cfg_degree": 0, "dfg_degree": 0},
            ],
            "edges": [{"source": 0, "target": 1, "edge_type": "AST_PARENT"}],
            "availability": {"ast": True, "cfg": False, "dfg": False},
        },
    }
    result = merge_row(
        row,
        {
            ("Java", "abc"): {
                "available": True,
                "cfg_available": True,
                "dfg_available": True,
                "cfg_edges": [{"source": 1, "target": 2, "edge_type": "CFG_NEXT"}],
                "dfg_edges": [{"source": 1, "target": 2, "edge_type": "DFG_DEF_USE"}],
                "cpg_nodes": 4,
                "mapped_cpg_nodes": 3,
                "node_mapping_coverage": 0.75,
            }
        },
    )
    structure = result["structure"]
    assert structure["availability"] == {"ast": True, "cfg": True, "dfg": True}
    assert len(structure["edges"]) == 3
    assert structure["regions"][1]["cfg_degree"] == 1
    assert structure["regions"][2]["dfg_degree"] == 1
    assert structure["joern"]["node_mapping_coverage"] == 0.75
