#!/usr/bin/env python3
"""Execute frozen external benchmark tasks for saved predictions in the sandbox."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.preflight_avatar_tc_references import (
    PAIR_SHIM as AVATAR_PAIR_SHIM,
)
from scripts.evaluate.preflight_avatar_tc_references import (
    load_data as load_avatar_data,
)
from scripts.evaluate.preflight_transcoder_test_references import (
    EXTENSIONS as TRANSCODER_EXTENSIONS,
)
from scripts.evaluate.preflight_transcoder_test_references import (
    PAIR_SHIM as TRANSCODER_PAIR_SHIM,
)
from scripts.evaluate.preflight_transcoder_test_references import (
    TESTS as TRANSCODER_TESTS,
)
from scripts.evaluate.preflight_transcoder_test_references import (
    _insert_candidate,
)
from src.evaluation.xcodeeval_preflight import (
    run_complete_program_reference,
    run_function_harness_reference,
    run_reference,
)

HUMANEVAL_ROOT = ROOT / "content/datasets/humanevalx/62c78627f3072a1454fa0cb0184737cafe5e4198/data"
TRANSCODER_RUNTIME = {"Python": "Python 3.11", "Java": "Java 8", "C++": "GNU C++11"}
AVATAR_RUNTIME = {"Python": "Python 3.11", "Java": "Java 14"}
_PRINTED_PROGRESS = threading.Lock()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_passed_tests(outcomes: dict[str, int], repeats: int) -> int:
    """Convert repeated per-case execution counts to unique test inputs."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return int(outcomes.get("passed", 0)) // repeats


def _program_runtime_status(compile_status: str, outcomes: list[dict[str, Any]]) -> str:
    """Classify compilation and execution independently from answer correctness."""
    if compile_status != "passed":
        return "compile_error"
    if any(item.get("timed_out") for item in outcomes):
        return "timeout"
    if any(item.get("returncode") != 0 or item.get("output_limited") for item in outcomes):
        return "runtime_error"
    return "passed"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _task_key(task: dict[str, Any], benchmark: str) -> str:
    problem = task.get("src_uid", task.get("problem_id"))
    return f"{task['source_language']}-to-{task['target_language']}/{problem}"


def _manifest_tasks(
    benchmark: str, manifest: dict[str, Any], split: str
) -> dict[str, dict[str, Any]]:
    if benchmark == "xcodeeval":
        directions = manifest["directions"]
    else:
        directions = manifest[split]
        if "directions" in directions:
            directions = directions["directions"]
    return {
        _task_key(task, benchmark): task
        for direction in directions.values()
        for task in direction["tasks"]
    }


def _load_humaneval_rows() -> dict[tuple[str, str], dict[str, Any]]:
    directories = {"Python": "python", "Java": "java", "C++": "cpp"}
    rows = {}
    for language, folder in directories.items():
        path = HUMANEVAL_ROOT / folder / "data" / "humaneval.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows[(language, row["task_id"].split("/")[-1])] = row
    return rows


def _prediction_files(
    prediction_root: Path,
    *,
    benchmark: str,
    split: str,
    manifest_path: Path,
) -> list[Path]:
    run_manifest = json.loads((prediction_root / "run_manifest.json").read_text())
    if run_manifest.get("status") != "COMPLETE" or run_manifest.get("generation_errors", 0):
        raise ValueError("Prediction run must be complete before benchmark scoring")
    frozen = run_manifest.get("frozen_protocol", {})
    if frozen.get("status") != "FROZEN_PREMODEL":
        raise ValueError("Prediction run lacks its pre-model frozen protocol record")
    frozen_path = Path(str(run_manifest.get("frozen_protocol_path", "")))
    frozen_digest = run_manifest.get("frozen_protocol_sha256")
    if not frozen_path.is_file() or not frozen_digest or _sha256_file(frozen_path) != frozen_digest:
        raise ValueError("Prediction run's frozen protocol file is missing or has changed")
    manifest_key = f"{benchmark}:{split}"
    if frozen.get("external_task_manifests", {}).get(manifest_key) != _sha256_file(manifest_path):
        raise ValueError("Scoring manifest does not match the prediction run's frozen task manifest")
    generation = run_manifest.get("generation_config", {})
    model = run_manifest.get("model", {})
    revision = Path(str(model.get("backbone_path", ""))).name
    architecture = model.get("architecture")
    policy_generation = frozen.get("generation", {})
    policy_eos = policy_generation.get("eos_token_id_by_revision", {}).get(
        revision, policy_generation.get("eos_token_id")
    )
    if policy_eos != generation.get("eos_token_id"):
        raise ValueError("Prediction EOS token differs from the frozen backbone policy")
    for field in (
        "max_new_tokens", "temperature", "top_p", "repetition_penalty", "postprocess",
        "batch_size",
    ):
        if policy_generation.get(field) != generation.get(field):
            raise ValueError(f"Prediction generation field differs from frozen policy: {field}")
    if (
        generation.get("do_sample") is not True
        or generation.get("num_beams") != 1
        or not isinstance(generation.get("max_new_tokens"), int)
        or generation["max_new_tokens"] < 1
        or "pass_at_5" in generation
    ):
        raise ValueError("Prediction run differs from the frozen Pass@1 protocol")
    pass1 = generation.get("pass_at_1", {})
    expected_pass1 = {
        "do_sample": True,
        "uses_sample_index": 0,
        "num_samples": 1,
        "temperature": 0.8,
        "top_p": 0.95,
        "num_beams": 1,
        "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32",
    }
    if (
        policy_generation.get("do_sample") is not generation.get("do_sample")
        or policy_generation.get("pass_at_1") != pass1
        or
        pass1 != expected_pass1
        or generation.get("temperature") != 0.8
        or generation.get("top_p") != 0.95
    ):
        raise ValueError("Prediction Pass@1 must use exactly one sample at index 0")
    if "pass_at_5" in policy_generation:
        raise ValueError("Frozen protocol must specify Pass@1 only")
    policy = frozen.get("routing_policies", {}).get(revision, {}).get(architecture)
    if not isinstance(policy, dict):
        raise ValueError(f"Frozen routing policy is absent for {revision}/{architecture}")
    config = model.get("config", {})
    if (
        policy.get("hard_threshold") != generation.get("hard_threshold")
        or policy.get("top_k_fraction") != generation.get("top_k_fraction")
        or policy.get("max_recurrence") != config.get("max_recurrence")
        or policy.get("insertion_fraction") != config.get("insertion_fraction")
        or frozen.get("training_horizon_epochs", {}).get(revision) != config.get("epochs")
    ):
        raise ValueError("Prediction routing/horizon settings differ from the frozen policy")
    return sorted(
        path for path in prediction_root.rglob("*.json") if path.name != "run_manifest.json"
    )


def _score_task(
    benchmark: str,
    task: dict[str, Any],
    prediction: dict[str, Any],
    test_payload: Any,
    repeats: int,
) -> dict[str, Any]:
    code = prediction.get("generated_code", "")
    target_language = task["target_language"]
    expected_test_hash = task.get("target_test_sha256", task.get("raw_test_sha256"))
    if prediction.get("target_test_sha256") != expected_test_hash:
        raise ValueError(
            f"Prediction test digest differs from frozen task: {prediction['task_id']}"
        )
    if prediction.get("target_reference_sha256") != task.get("target_reference_sha256"):
        raise ValueError(
            f"Prediction reference digest differs from frozen task: {prediction['task_id']}"
        )
    if prediction.get("source_sha256") != task.get("_source_sha256"):
        raise ValueError(
            f"Prediction source digest differs from frozen task: {prediction['task_id']}"
        )
    interface_hashes = {task.get("_target_interface_sha256")}
    if benchmark == "humanevalx":
        interface_hashes.add(task.get("_target_declaration_sha256"))
    if prediction.get("target_interface_sha256") not in interface_hashes:
        raise ValueError(
            f"Prediction interface digest differs from frozen task: {prediction['task_id']}"
        )
    expected_tests = int(task.get("target_test_count", task.get("test_count", 0)))
    if prediction.get("status") != "generated":
        return {
            "status": prediction.get("status", "missing_generation"),
            "execution_status": "not_run",
            "compile_status": "not_run",
            "runtime_status": "not_run",
            "tests_passed": 0,
            "tests_total": expected_tests,
            "pass_at_1": False,
            "diagnostic": prediction.get("error", ""),
        }

    if benchmark == "xcodeeval":
        response, sandbox = run_reference(
            code,
            target_language,
            str(task["target_runtime_label"]),
            test_payload,
            repeats=repeats,
            case_timeout_seconds=6.0,
        )
        if response is None:
            return {
                "status": "runner_error",
                "execution_status": "runner_error",
                "compile_status": "unknown",
                "runtime_status": "unknown",
                "tests_passed": 0,
                "tests_total": expected_tests,
                "pass_at_1": False,
                "pass_ratio_available": False,
                "diagnostic": sandbox.stderr.decode(errors="replace")[:1000],
                "timed_out": sandbox.timed_out,
            }
        outcomes = response.get("outcomes", {})
        runtime_status = (
            "compile_error"
            if response.get("compile_status") != "passed"
            else "timeout"
            if outcomes.get("timeout", 0)
            else "runtime_error"
            if outcomes.get("runtime_error", 0)
            else "passed"
        )
        return {
            "status": response.get("status", "unknown"),
            "execution_status": response.get("status", "unknown"),
            "compile_status": response.get("compile_status", "unknown"),
            "runtime_status": runtime_status,
            "tests_passed": _unique_passed_tests(outcomes, repeats),
            "tests_total": expected_tests,
            "tests_checked_until_failure": response.get("tests_checked"),
            "case_outcomes": response.get("outcomes", {}),
            "repeats": response.get("repeats", repeats),
            "flaky": response.get("flaky", False),
            "pass_at_1": bool(response.get("all_passed") and not response.get("flaky")),
            "pass_ratio_available": not bool(response.get("flaky", False)),
            "diagnostic": response.get("diagnostic", ""),
        }

    if benchmark == "humanevalx":
        target_row = test_payload
        entry_point = None
        if target_language == "Python":
            match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", target_row["prompt"])
            if match is None:
                raise ValueError(
                    f"HumanEval-X target prompt has no Python function: {prediction['task_id']}"
                )
            entry_point = match.group(1)
        preamble = (
            "".join(
                line + "\n"
                for line in target_row["prompt"].splitlines()
                if line.lstrip().startswith("import ")
            )
            if target_language == "Java"
            else ""
        )
        response, sandbox = run_function_harness_reference(
            code,
            target_row["test"],
            target_language,
            str(task["runtime"]),
            interface_source=target_row["prompt"],
            entry_point=entry_point,
            test_preamble=preamble,
            repeats=repeats,
            timeout_seconds=30.0,
        )
        if response is None:
            return {
                "status": "runner_error",
                "execution_status": "runner_error",
                "compile_status": "unknown",
                "runtime_status": "unknown",
                "tests_passed": 0,
                "tests_total": expected_tests,
                "pass_at_1": False,
                "pass_ratio_available": False,
                "diagnostic": sandbox.stderr.decode(errors="replace")[:1000],
                "timed_out": sandbox.timed_out,
            }
        passed = bool(response.get("all_passed") and not response.get("flaky"))
        program_outcomes = response.get("program_outcomes", [])
        return {
            "status": response.get("status", "unknown"),
            "execution_status": response.get("status", "unknown"),
            "compile_status": response.get("compile_status", "unknown"),
            "runtime_status": _program_runtime_status(
                str(response.get("compile_status", "unknown")), program_outcomes
            ),
            "tests_passed": expected_tests if passed else 0,
            "tests_total": expected_tests,
            "repeats": response.get("repeats", repeats),
            "flaky": response.get("flaky", False),
            "pass_at_1": passed,
            "pass_ratio_available": False,
            "pass_ratio_unavailable_reason": "humanevalx_harness_reports_task_level_outcome_only",
            "diagnostic": response.get("diagnostic", ""),
        }

    if benchmark == "transcoder_test":
        language = target_language
        uid = str(task["problem_id"])
        extension = TRANSCODER_EXTENSIONS[language]
        short = {"Python": "python", "Java": "java", "C++": "cpp"}[language]
        test_path = TRANSCODER_TESTS / short / f"{uid}{extension}"
        test_source = test_path.read_text(encoding="utf-8")
        candidate = _insert_candidate(test_source, code, language)
        auxiliary = (
            {"javafx/util/Pair.java": TRANSCODER_PAIR_SHIM}
            if language == "Java" and "javafx.util.Pair" in test_source
            else {}
        )
        response, sandbox = run_complete_program_reference(
            candidate,
            language,
            str(task["target_runtime"]),
            repeats=repeats,
            timeout_seconds=30.0,
            auxiliary_sources=auxiliary,
        )
        if response is None:
            return {
                "status": "runner_error",
                "execution_status": "runner_error",
                "compile_status": "unknown",
                "runtime_status": "unknown",
                "tests_passed": 0,
                "tests_total": expected_tests,
                "pass_at_1": False,
                "pass_ratio_available": False,
                "diagnostic": sandbox.stderr.decode(errors="replace")[:1000],
                "timed_out": sandbox.timed_out,
            }
        samples = [
            outcome.get("stdout_sample", "") for outcome in response.get("program_outcomes", [])
        ]
        parsed = [
            re.search(r"#\s*Results\s*:\s*(\d+)\s*,\s*(\d+)", sample, re.IGNORECASE)
            for sample in samples
        ]
        passed_counts = [int(match.group(1)) for match in parsed if match]
        totals = [int(match.group(2)) for match in parsed if match]
        stable_counts = len(parsed) == repeats and all(match is not None for match in parsed)
        passed = (
            response.get("all_passed") is True
            and response.get("flaky") is False
            and stable_counts
            and all(count == expected_tests for count in totals)
            and all(count == expected_tests for count in passed_counts)
        )
        program_outcomes = response.get("program_outcomes", [])
        return {
            "status": "passed" if passed else response.get("status", "failed"),
            "execution_status": "passed" if passed else "failed",
            "compile_status": response.get("compile_status", "unknown"),
            "runtime_status": _program_runtime_status(
                str(response.get("compile_status", "unknown")), program_outcomes
            ),
            "tests_passed": min(passed_counts)
            if stable_counts and not response.get("flaky")
            else 0,
            "tests_total": expected_tests,
            "repeats": response.get("repeats", repeats),
            "flaky": response.get("flaky", False),
            "reported_results": [
                {"passed": p, "total": t} for p, t in zip(passed_counts, totals, strict=False)
            ],
            "pass_at_1": passed,
            "pass_ratio_available": stable_counts and not bool(response.get("flaky")),
            "diagnostic": response.get("diagnostic", ""),
        }

    if benchmark == "avatar_tc":
        language = target_language
        auxiliary = (
            {"javafx/util/Pair.java": AVATAR_PAIR_SHIM}
            if task.get("_target_requires_pair_shim", False)
            else {}
        )
        response, sandbox = run_reference(
            code,
            language,
            str(task["target_runtime"]),
            test_payload,
            repeats=repeats,
            case_timeout_seconds=8.0,
            auxiliary_sources=auxiliary,
            comparison_policy="avatar_casefold_whitespace",
        )
        if response is None:
            return {
                "status": "runner_error",
                "execution_status": "runner_error",
                "compile_status": "unknown",
                "runtime_status": "unknown",
                "tests_passed": 0,
                "tests_total": expected_tests,
                "pass_at_1": False,
                "pass_ratio_available": False,
                "diagnostic": sandbox.stderr.decode(errors="replace")[:1000],
                "timed_out": sandbox.timed_out,
            }
        outcomes = response.get("outcomes", {})
        runtime_status = (
            "compile_error"
            if response.get("compile_status") != "passed"
            else "timeout"
            if outcomes.get("timeout", 0)
            else "runtime_error"
            if outcomes.get("runtime_error", 0)
            else "passed"
        )
        return {
            "status": response.get("status", "unknown"),
            "execution_status": response.get("status", "unknown"),
            "compile_status": response.get("compile_status", "unknown"),
            "runtime_status": runtime_status,
            "tests_passed": _unique_passed_tests(outcomes, repeats),
            "tests_total": expected_tests,
            "case_outcomes": outcomes,
            "repeats": response.get("repeats", repeats),
            "flaky": response.get("flaky", False),
            "pass_at_1": bool(response.get("all_passed") and not response.get("flaky")),
            "pass_ratio_available": not bool(response.get("flaky", False)),
            "diagnostic": response.get("diagnostic", ""),
        }
    raise ValueError(f"Unsupported benchmark {benchmark}")


def score(
    benchmark: str,
    split: str,
    manifest_path: Path,
    prediction_root: Path,
    output_root: Path,
    xcode_selection_path: Path | None = None,
    repeats: int = 3,
    workers: int = 4,
) -> dict[str, Any]:
    """Score every saved prediction, retaining task failures and resume records."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = _manifest_tasks(benchmark, manifest, split)
    human_rows = _load_humaneval_rows() if benchmark == "humanevalx" else None
    for key, task in tasks.items():
        if benchmark in {"transcoder_test", "avatar_tc"}:
            source = task["source_code"]
        elif benchmark == "humanevalx":
            assert human_rows is not None
            source_row = human_rows[(task["source_language"], str(task["problem_id"]))]
            source = source_row["prompt"] + source_row["canonical_solution"]
        else:
            source = None
        task["_source_sha256"] = task.get(
            "source_sha256", task.get("source_code_sha256", _sha256(source) if source else None)
        )
        if benchmark == "humanevalx":
            target_row = human_rows[(task["target_language"], str(task["problem_id"]))]
            # The two frozen HumanEval-X structural caches use the full
            # prompt and shorter declaration respectively. Validate either
            # pinned dataset string against the saved generation record.
            interface = target_row["prompt"]
            reference = target_row["prompt"] + target_row["canonical_solution"]
            if task["target_reference_sha256"] != _sha256(reference):
                raise ValueError(f"HumanEval-X target reference hash mismatch: {key}")
            task["_target_interface_sha256"] = _sha256(interface)
            task["_target_declaration_sha256"] = _sha256(target_row["declaration"])
        else:
            interface = str(task.get("target_interface", ""))
            task["_target_interface_sha256"] = _sha256(interface)

    xcode_selection = None
    if benchmark == "xcodeeval":
        if xcode_selection_path is None:
            raise ValueError("xcode_selection_path is required for xCodeEval scoring")
        selection_hash = _sha256_file(xcode_selection_path)
        if selection_hash != manifest["selection_manifest_sha256"]:
            raise ValueError("xCodeEval selection manifest hash differs from frozen task manifest")
        xcode_selection = json.loads(xcode_selection_path.read_text(encoding="utf-8"))

    avatar_reference_code = load_avatar_data()[1] if benchmark == "avatar_tc" else None

    predictions = _prediction_files(
        prediction_root,
        benchmark=benchmark,
        split=split,
        manifest_path=manifest_path,
    )
    if len(predictions) != len(tasks):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match frozen task count {len(tasks)}"
        )
    prediction_by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in predictions:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["benchmark"] != benchmark or row["split"] != split:
            raise ValueError(f"Unexpected prediction record {path}")
        prediction_by_key[row["task_id"]] = (path, row)
    missing = set(tasks) - prediction_by_key.keys()
    extra = prediction_by_key.keys() - set(tasks)
    if missing or extra:
        raise ValueError(
            f"Prediction/task identity mismatch; missing={len(missing)}, extra={len(extra)}"
        )

    input_manifest_hash = _sha256_file(manifest_path)
    prediction_run = json.loads((prediction_root / "run_manifest.json").read_text())
    model_metadata = prediction_run["model"]
    score_manifest_path = output_root / "score_manifest.json"
    score_manifest = {
        "schema_version": 2,
        "candidate_scoring_version": "pass_at_1_single_sample_v1",
        "status": "RUNNING",
        "benchmark": benchmark,
        "split": split,
        "eligibility_manifest_sha256": input_manifest_hash,
        "prediction_run_manifest_sha256": _sha256_file(prediction_root / "run_manifest.json"),
        "repeats": repeats,
        "workers": workers,
        "tasks_total": len(tasks),
    }
    if score_manifest_path.exists():
        old = json.loads(score_manifest_path.read_text(encoding="utf-8"))
        for field in (
            "benchmark",
            "split",
            "eligibility_manifest_sha256",
            "prediction_run_manifest_sha256",
            "repeats",
            "candidate_scoring_version",
        ):
            if old.get(field) != score_manifest[field]:
                raise ValueError(f"Existing score run differs at {field}; use a new output path")
    _atomic_json(score_manifest_path, score_manifest)

    done = 0
    counts: Counter[str] = Counter()
    lock = threading.Lock()

    def run_one(task_key: str) -> tuple[str, str]:
        task = tasks[task_key]
        expected_test_hash = task.get("target_test_sha256", task.get("raw_test_sha256"))
        expected_tests = int(task.get("target_test_count", task.get("test_count", 0)))
        prediction_path, prediction = prediction_by_key[task_key]
        result_id = hashlib.sha256(task_key.encode()).hexdigest()[:20]
        direction = task_key.split("/", 1)[0]
        result_path = output_root / direction / f"{result_id}.json"
        prediction_hash = _sha256_file(prediction_path)
        if result_path.exists():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                old.get("prediction_record_sha256") == prediction_hash
                and old.get("eligibility_manifest_sha256") == input_manifest_hash
                and old.get("repeats") == repeats
                and old.get("status") not in {"runner_or_setup_error", "runner_error"}
            ):
                return task_key, old["status"]

        test_payload: Any
        if benchmark == "xcodeeval":
            assert xcode_selection is not None
            selected = xcode_selection["selected"][task["target_language"]][task["src_uid"]]
            target_reference = next(
                candidate
                for candidate in xcode_selection["candidate_pool"][task["target_language"]][
                    task["src_uid"]
                ]
                if candidate["code_uid"] == task["target_reference_code_uid"]
            )
            if selected["raw_test_sha256"] != expected_test_hash:
                raise ValueError(f"xCodeEval target test hash mismatch for {task_key}")
            if target_reference["source_sha256"] != task["target_reference_sha256"]:
                raise ValueError(f"xCodeEval target reference hash mismatch for {task_key}")
            if len(selected["tests"]) != expected_tests:
                raise ValueError(f"xCodeEval test count mismatch for {task_key}")
            test_payload = selected["tests"]
        elif benchmark == "humanevalx":
            assert human_rows is not None
            test_payload = human_rows[(task["target_language"], str(task["problem_id"]))]
            if _sha256(test_payload["test"]) != expected_test_hash:
                raise ValueError(f"HumanEval-X test hash mismatch for {task_key}")
        elif benchmark == "transcoder_test":
            language = task["target_language"]
            short = {"Python": "python", "Java": "java", "C++": "cpp"}[language]
            extension = TRANSCODER_EXTENSIONS[language]
            test_path = TRANSCODER_TESTS / short / f"{task['problem_id']}{extension}"
            test_payload = test_path.read_text(encoding="utf-8")
            if _sha256(test_payload) != expected_test_hash:
                raise ValueError(f"TransCoder-Test test hash mismatch for {task_key}")
        else:
            test_payload = task["tests"]
            serialized_tests = json.dumps(test_payload, ensure_ascii=False, separators=(",", ":"))
            if _sha256(serialized_tests) != expected_test_hash:
                raise ValueError(f"AVATAR-TC test hash mismatch for {task_key}")
            assert avatar_reference_code is not None
            task["_target_requires_pair_shim"] = (
                task["target_language"] == "Java"
                and "javafx.util.Pair" in avatar_reference_code["Java"][task["src_uid"]]
            )

        started = time.perf_counter()
        fatal_error: Exception | None = None
        try:
            sampled = prediction.get("sampled_candidates")
            if not isinstance(sampled, list) or len(sampled) != 1:
                raise ValueError("Prediction record must contain exactly one Pass@1 candidate")
            if sampled[0].get("sample_index") != 0:
                raise ValueError("Pass@1 candidate must have sample index 0")
            if sampled[0].get("generated_code") != prediction.get("generated_code"):
                raise ValueError("Pass@1 candidate differs from the prediction record")
            pass_at_1_candidate = _score_task(benchmark, task, prediction, test_payload, repeats)
            scored = {
                **pass_at_1_candidate,
                "pass_at_1_result": pass_at_1_candidate,
                "pass_at_1": bool(pass_at_1_candidate.get("pass_at_1", False)),
                "sample_count": 1,
            }
            if scored.get("status") == "runner_error":
                fatal_error = RuntimeError(str(scored.get("diagnostic", "sandbox runner error")))
        except Exception as exc:
            fatal_error = exc
            scored = {
                "status": "runner_or_setup_error",
                "execution_status": "not_scored",
                "compile_status": "unknown",
                "runtime_status": "unknown",
                "tests_passed": 0,
                "tests_total": expected_tests,
                "pass_at_1": False,
                "error_type": type(exc).__name__,
                "diagnostic": str(exc),
            }
        record = {
            "schema_version": 1,
            "benchmark": benchmark,
            "split": split,
            "experiment_id": model_metadata["experiment_id"],
            "architecture": model_metadata["architecture"],
            "seed": int(model_metadata["config"]["seed"]),
            "backbone": model_metadata["backbone_path"],
            "task_id": task_key,
            "problem_id": task.get("src_uid", task.get("problem_id")),
            "source_language": task["source_language"],
            "target_language": task["target_language"],
            "direction": f"{task['source_language']}-to-{task['target_language']}",
            "source_sha256": task["_source_sha256"],
            "target_reference_sha256": task.get("target_reference_sha256"),
            "target_test_sha256": expected_test_hash,
            "prediction_record_sha256": prediction_hash,
            "generated_code_sha256": _sha256(prediction.get("generated_code", "")),
            "sampled_code_sha256": [
                _sha256(str(sample.get("generated_code", "")))
                for sample in prediction.get("sampled_candidates", [])
            ],
            "eligibility_manifest_sha256": input_manifest_hash,
            "repeats": repeats,
            "latency_seconds": time.perf_counter() - started,
            **scored,
        }
        _atomic_json(result_path, record)
        if fatal_error is not None:
            raise RuntimeError(f"Scorer setup failed for {task_key}: {fatal_error}") from fatal_error
        return task_key, str(record["status"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, key) for key in sorted(tasks)]
        for future in concurrent.futures.as_completed(futures):
            _task_id, status = future.result()
            with lock:
                done += 1
                counts[status] += 1
                if done % 20 == 0:
                    with _PRINTED_PROGRESS:
                        print(
                            f"scored {done}/{len(tasks)} tasks; statuses={dict(counts)}", flush=True
                        )
    score_manifest["status"] = "COMPLETE"
    score_manifest["tasks_completed"] = done
    score_manifest["status_counts"] = dict(counts)
    score_manifest["pass_at_1"] = sum(
        json.loads(path.read_text(encoding="utf-8"))["pass_at_1"]
        for path in output_root.rglob("*.json")
        if path.name != "score_manifest.json"
    )
    scored_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_root.rglob("*.json")
        if path.name != "score_manifest.json"
    ]
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        direction = f"{row['source_language']}-to-{row['target_language']}"
        by_direction.setdefault(direction, []).append(row)
    directional_metrics = {}
    for direction, rows in by_direction.items():
        directional_metrics[direction] = {
            "tasks": len(rows),
            "pass_at_1": sum(bool(row.get("pass_at_1")) for row in rows) / len(rows),
        }
    score_manifest["directional_metrics"] = directional_metrics
    for metric in ("pass_at_1",):
        directional_values = [
            float(row[metric])
            for row in directional_metrics.values()
            if isinstance(row.get(metric), int | float)
        ]
        score_manifest[f"{metric}_macro"] = (
            sum(directional_values) / len(directional_values) if directional_values else None
        )
    _atomic_json(score_manifest_path, score_manifest)
    return score_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["xcodeeval", "humanevalx", "transcoder_test", "avatar_tc"],
    )
    parser.add_argument(
        "--split", choices=["common_core", "full_directional"], default="common_core"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--xcode-selection", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = score(
        benchmark=args.benchmark,
        split=args.split,
        manifest_path=args.manifest,
        prediction_root=args.predictions,
        output_root=args.output_root,
        xcode_selection_path=args.xcode_selection,
        repeats=args.repeats,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
