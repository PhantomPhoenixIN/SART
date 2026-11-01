"""Tests for aligned all-direction XLCoST filtering and emission."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preprocess.build_xlcost_pairs import PAIRS, build
from src.parsing.tree_sitter_adapter import ParseResult


class BuildXLCoSTPairsTests(unittest.TestCase):
    def test_pair_filter_is_symmetric_and_emits_six_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "extraction_manifest.json").write_text(
                json.dumps({"status": "COMPLETE", "source_archive_sha256": "fixture"}),
                encoding="utf-8",
            )
            extensions = {"Python": ".py", "Java": ".java", "C++": ".cpp"}
            for pair_name, languages in PAIRS.items():
                folder = source / pair_name
                folder.mkdir()
                for file_split in ("train", "val", "test"):
                    for language in languages:
                        (folder / f"{file_split}-{language}-map.jsonl").write_text(
                            "problem-ok\nproblem-bad\n", encoding="utf-8"
                        )
                        codes = "valid\ninvalid\n" if language != "Java" else "valid\nvalid\n"
                        (folder / f"{file_split}-{pair_name}-tok{extensions[language]}").write_text(
                            codes, encoding="utf-8"
                        )

            def fake_parser(code: str, language: str) -> ParseResult:
                valid = code.strip() != "invalid"
                return ParseResult(valid, 0 if valid else 1, "program")

            manifest = build(source, output, fake_parser)
            self.assertEqual(len(manifest["splits"]["train"]["directions"]), 6)
            for direction, metadata in manifest["splits"]["train"]["directions"].items():
                source_language, target_language = direction.split("-to-")
                expected = 1
                self.assertEqual(metadata["records"], expected, direction)
                records = [
                    json.loads(line)
                    for line in (output / "train" / f"{direction}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(records[0]["problem_id"], "problem-ok")
                if expected == 2:
                    self.assertEqual(records[1]["problem_id"], "problem-bad")
            self.assertEqual(
                manifest["splits"]["train"]["pairs"]["Java-Python"]["excluded_parse"], 1
            )


if __name__ == "__main__":
    unittest.main()
