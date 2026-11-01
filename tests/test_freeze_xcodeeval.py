import hashlib
import json
from pathlib import Path

from scripts.evaluate.freeze_xcodeeval_manifests import freeze


def test_freeze_uses_problem_level_common_and_directional_sets(tmp_path: Path) -> None:
    uid = "problem-1"
    languages = ("Python", "Java", "C++")
    candidate_pool = {language: {uid: []} for language in languages}
    selected = {
        language: {
            uid: {
                "test_count": index + 2,
                "raw_test_sha256": f"test-{language}",
            }
        }
        for index, language in enumerate(languages)
    }
    selection = {
        "dataset_revision": "pinned-revision",
        "selection_rule": "median then uid",
        "candidate_pool": candidate_pool,
        "selected": selected,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    results_path = tmp_path / "results"
    for language in languages:
        directory = language.replace("+", "p")
        path = results_path / directory / f"{uid}.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "manifest_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    "complete": True,
                    "valid": True,
                    "status": "passed",
                    "selected_code_uid": f"code-{language}",
                    "selected_source_sha256": f"source-{language}",
                    "selected_runtime_label": language,
                }
            )
        )

    output_dir = tmp_path / "manifests"
    summary = freeze(selection_path, results_path, output_dir)
    common = json.loads((output_dir / "xcodeeval_common_core_manifest.json").read_text())
    full = json.loads((output_dir / "xcodeeval_full_direction_manifest.json").read_text())

    assert summary["common_core_problems"] == 1
    assert summary["common_core_tasks"] == 6
    assert len(common["directions"]) == 6
    assert common["common_core_coverage"]["hidden_test_executions_one_pass"] == 18
    assert full["directions"]["Python-to-Java"]["translation_tasks"] == 1
    task = common["directions"]["Python-to-Java"]["tasks"][0]
    assert task["target_reference_code_uid"] == "code-Java"
