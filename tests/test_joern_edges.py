import csv

from src.regions.joern_edges import map_joern_edges


def _write_csv(directory, stem, header, rows):
    (directory / f"{stem}_header.csv").write_text(",".join(header) + "\n")
    with (directory / f"{stem}_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)


def test_maps_joern_cfg_and_reaching_def_to_smallest_regions(tmp_path):
    _write_csv(
        tmp_path,
        "nodes_FILE",
        [":ID", ":LABEL", "NAME:string"],
        [["1", "FILE", "source.py"]],
    )
    _write_csv(
        tmp_path,
        "nodes_METHOD",
        [":ID", ":LABEL", "FILENAME:string", "OFFSET:int", "OFFSET_END:int"],
        [["2", "METHOD", "source.py", "0", "20"]],
    )
    _write_csv(
        tmp_path,
        "nodes_CALL",
        [":ID", ":LABEL", "OFFSET:int", "OFFSET_END:int"],
        [["3", "CALL", "2", "5"], ["4", "CALL", "10", "15"]],
    )
    _write_csv(
        tmp_path,
        "edges_CONTAINS",
        [":START_ID", ":END_ID", ":TYPE"],
        [["1", "2", "CONTAINS"], ["2", "3", "CONTAINS"], ["2", "4", "CONTAINS"]],
    )
    for relation in ("CFG", "REACHING_DEF"):
        _write_csv(
            tmp_path,
            f"edges_{relation}",
            [":START_ID", ":END_ID", ":TYPE"],
            [["3", "4", relation]],
        )

    result = map_joern_edges(
        tmp_path,
        "source.py",
        [
            {"region_id": 0, "region_type": "PROGRAM", "start_byte": 0, "end_byte": 20},
            {"region_id": 1, "region_type": "FUNCTION", "start_byte": 0, "end_byte": 20},
            {"region_id": 2, "region_type": "CALL", "start_byte": 2, "end_byte": 5},
            {"region_id": 3, "region_type": "ASSIGNMENT", "start_byte": 10, "end_byte": 15},
        ],
    )
    assert result["available"]
    assert result["cfg_edges"] == [{"source": 2, "target": 3, "edge_type": "CFG_NEXT"}]
    assert result["dfg_edges"] == [{"source": 2, "target": 3, "edge_type": "DFG_DEF_USE"}]
    assert result["node_mapping_coverage"] == 1.0


def test_unicode_offsets_are_aligned_to_utf8_region_spans(tmp_path):
    source = "# café 🧪 before every AST span\ndef add_one(value):\n    return value + 1\n"
    _write_csv(
        tmp_path,
        "nodes_FILE",
        [":ID", ":LABEL", "NAME:string"],
        [["1", "FILE", "unicode.py"]],
    )
    _write_csv(
        tmp_path,
        "nodes_METHOD",
        [":ID", ":LABEL", "FILENAME:string", "OFFSET:int", "OFFSET_END:int"],
        [["2", "METHOD", "unicode.py", "32", "73"]],
    )
    result = map_joern_edges(
        tmp_path,
        "unicode.py",
        [
            {"region_id": 0, "region_type": "PROGRAM", "start_byte": 0, "end_byte": 76},
            {"region_id": 1, "region_type": "FUNCTION", "start_byte": 35, "end_byte": 75},
        ],
        source,
    )
    assert result["available"]
    assert result["mapped_cpg_nodes"] == 1
    assert result["node_mapping_coverage"] == 1.0
