"""Non-executing integrity tests for external scoring records."""

import json

import pytest

from scripts.evaluate.score_external_predictions import (
    _aggregate_pass_at_n,
    _manifest_tasks,
    _prediction_files,
    _program_runtime_status,
    _score_task,
    _unique_passed_tests,
)


def test_repeated_case_executions_count_unique_inputs() -> None:
    assert _unique_passed_tests({"passed": 6}, repeats=3) == 2
    assert _unique_passed_tests({"passed": 5}, repeats=3) == 1


def test_runtime_status_distinguishes_compile_and_execution_failures() -> None:
    assert _program_runtime_status("failed", []) == "compile_error"
    assert _program_runtime_status("passed", [{"timed_out": True}]) == "timeout"
    assert _program_runtime_status("passed", [{"returncode": 1}]) == "runtime_error"
    assert _program_runtime_status("passed", [{"returncode": 0}]) == "passed"


def test_manifest_task_map_keeps_problem_direction_as_unit() -> None:
    task = {
        "problem_id": "p-1",
        "source_language": "Python",
        "target_language": "Java",
        "source_code": "def f(): pass",
    }
    rows = _manifest_tasks(
        "transcoder_test",
        {"full_directional": {"Python-to-Java": {"tasks": [task]}}},
        "full_directional",
    )
    assert list(rows) == ["Python-to-Java/p-1"]


def test_non_generated_prediction_is_a_failed_task_without_execution() -> None:
    task = {
        "problem_id": "p-1",
        "source_language": "Python",
        "target_language": "Java",
        "target_test_sha256": "test-digest",
        "target_reference_sha256": "reference-digest",
        "test_count": 5,
        "_source_sha256": "source-digest",
        "_target_interface_sha256": "interface-digest",
    }
    prediction = {
        "task_id": "Python-to-Java/p-1",
        "status": "context_overflow",
        "target_test_sha256": "test-digest",
        "target_reference_sha256": "reference-digest",
        "source_sha256": "source-digest",
        "target_interface_sha256": "interface-digest",
    }
    result = _score_task("transcoder_test", task, prediction, None, repeats=3)
    assert result["pass_at_1"] is False
    assert result["tests_total"] == 5
    assert result["execution_status"] == "not_run"


def test_pass_at_5_and_squared_pass_ratio_use_exactly_five_samples() -> None:
    greedy = {"pass_at_1": False}
    samples = [
        {
            "pass_at_1": passed == 4,
            "tests_passed": passed,
            "tests_total": 4,
        }
        for passed in (0, 1, 2, 3, 4)
    ]
    result = _aggregate_pass_at_n(greedy, samples, tests_total=4)
    assert result["pass_at_1"] is False
    assert result["pass_at_5"] is True
    assert result["sample_passes"] == 1
    assert result["pass_ratio_at_5"] == (0 + 0.0625 + 0.25 + 0.5625 + 1) / 5
    assert result["mean_test_pass_ratio_at_5"] == 0.5


def test_pass_ratio_is_unavailable_for_binary_only_harnesses() -> None:
    samples = [
        {
            "pass_at_1": i == 4,
            "tests_passed": 1 if i == 4 else 0,
            "tests_total": 1,
            "pass_ratio_available": False,
        }
        for i in range(5)
    ]
    result = _aggregate_pass_at_n({"pass_at_1": False}, samples, tests_total=1)
    assert result["pass_at_5"] is True
    assert result["pass_ratio_at_5"] is None
    assert result["pass_ratio_available"] is False


def test_pass_at_5_rejects_wrong_candidate_count() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        _aggregate_pass_at_n({"pass_at_1": False}, [], tests_total=2)


def test_pass_ratio_rejects_a_candidate_test_denominator_mismatch() -> None:
    samples = [{"pass_at_1": False, "tests_passed": 0, "tests_total": 3} for _ in range(5)]
    with pytest.raises(ValueError, match="denominators differ"):
        _aggregate_pass_at_n({"pass_at_1": False}, samples, tests_total=4)


def test_prediction_scorer_requires_frozen_passk_sampling_protocol(tmp_path) -> None:
    protocol = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 768,
        "pass_at_5": {
            "do_sample": True,
            "num_samples": 5,
            "temperature": 0.8,
            "top_p": 0.95,
            "num_beams": 1,
            "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32",
        },
    }

    def write_manifest() -> None:
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "frozen_protocol": {"status": "FROZEN_PREMODEL"},
                    "generation_config": protocol,
                }
            ),
            encoding="utf-8",
        )

    write_manifest()
    assert _prediction_files(tmp_path) == []
    protocol["pass_at_5"]["num_samples"] = 4
    write_manifest()
    with pytest.raises(ValueError, match="frozen Pass@1/Pass@5"):
        _prediction_files(tmp_path)
