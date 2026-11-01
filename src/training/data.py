"""Tokenization, balanced example loading, and source-region batch collation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.translation_protocol import serialize_prompt, serialize_target
from src.models.recurrent.region_module import LANGUAGES, REGION_TYPES, RELATIONS, RegionBatch

COARSE_REGION_TYPE = {
    "PARAMETER": "DECLARATION",
    "CONDITIONAL": "BRANCH",
    "LOOP": "BRANCH",
    "SWITCH": "BRANCH",
    "ASSIGNMENT": "DECLARATION",
    "MUTATION": "DECLARATION",
    "CONSTRUCTOR": "CALL",
    "INDEX_ACCESS": "MEMBER_ACCESS",
    "BOOLEAN_EXPRESSION": "ARITHMETIC",
    "COMPARISON": "ARITHMETIC",
    "RETURN": "BREAK",
    "CONTINUE": "BREAK",
    "EXCEPTION": "BREAK",
}

STRUCTURAL_SCALAR_ALLOWLIST = {
    "ast_sart": {"ast_depth", "child_count"},
    "cfg_sart": {"cfg_degree"},
    "dfg_sart": {"dfg_degree"},
    "ast_cfg_sart": {"ast_depth", "child_count", "cfg_degree"},
    "ast_dfg_sart": {"ast_depth", "child_count", "dfg_degree"},
    "sart_no_scalar": set(),
}


def _structural_scalars(region: dict[str, Any], architecture: str) -> dict[str, float]:
    """Return edge-family-consistent scalar features for a region ablation."""
    allowlist = STRUCTURAL_SCALAR_ALLOWLIST.get(architecture)
    return {
        "ast_depth": float(region.get("ast_depth", 0)) / 32.0
        if allowlist is None or "ast_depth" in allowlist
        else 0.0,
        "child_count": float(region.get("child_count", 0))
        if allowlist is None or "child_count" in allowlist
        else 0.0,
        "cfg_degree": float(region.get("cfg_degree", 0))
        if allowlist is None or "cfg_degree" in allowlist
        else 0.0,
        "dfg_degree": float(region.get("dfg_degree", 0))
        if allowlist is None or "dfg_degree" in allowlist
        else 0.0,
    }


def _region_type_id(region_type: str, architecture: str) -> int:
    """Map fine labels to a frozen coarse ontology for the ontology ablation."""
    if architecture == "sart_coarse_ontology":
        region_type = COARSE_REGION_TYPE.get(region_type, region_type)
    return REGION_TYPES[region_type]


class ManifestDataset(Dataset):
    """Dataset backed by immutable selected directional JSONL shards."""

    def __init__(
        self,
        manifest_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_sequence_length: int | None = 2048,
    ) -> None:
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.index: list[tuple[Path, int]] = []
        self._handles: dict[Path, Any] = {}
        if split == "train":
            manifest = json.loads(manifest_path.read_text())
            for direction in manifest["direction_order"]:
                file_path = Path(manifest["files"][direction]["path"])
                self._index_jsonl(file_path)
        else:
            structure_root = (
                manifest_path / split
                if manifest_path.is_dir()
                else Path("content/datasets/xlcost/structural_ast_qwen") / split
            )
            for path in sorted(structure_root.glob("*/*.jsonl")):
                self._index_jsonl(path)

    def _index_jsonl(self, path: Path) -> None:
        """Index record offsets so large structural shards are not all kept in RAM."""
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                self.index.append((path, offset))

    def __getstate__(self):
        """Do not share open file handles across DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, offset = self.index[index]
        if path not in self._handles:
            self._handles[path] = path.open("rb")
        stream = self._handles[path]
        stream.seek(offset)
        row = json.loads(stream.readline())
        prompt = serialize_prompt(row["source_language"], row["target_language"], row["source"])
        target = serialize_target(row["target"])
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        if (
            self.max_sequence_length is not None
            and len(prompt_ids) + len(target_ids) > self.max_sequence_length
        ):
            raise ValueError(f"Frozen context filter missed {row['problem_id']}")
        source_mapping = [None] * len(prompt_ids)
        for token_index, region_id in row["structure"]["source_token_to_region"]:
            if token_index < len(prompt_ids):
                source_mapping[token_index] = region_id
        structure = row["structure"]
        regions = structure["regions"]
        edges = structure["edges"]
        return {
            "row": row,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "source_token_to_region": source_mapping,
            "regions": regions,
            "edges": edges,
        }

    def row_metadata(self, index: int) -> dict[str, Any]:
        """Read task identifiers/direction without tokenizing code."""
        path, offset = self.index[index]
        if path not in self._handles:
            self._handles[path] = path.open("rb")
        stream = self._handles[path]
        stream.seek(offset)
        row = json.loads(stream.readline())
        return {
            "problem_id": row["problem_id"],
            "source_language": row["source_language"],
            "target_language": row["target_language"],
        }

    def target_reference(self, index: int) -> tuple[str, str, str, str]:
        """Read target reference text and its direction without tokenizer work."""
        path, offset = self.index[index]
        if path not in self._handles:
            self._handles[path] = path.open("rb")
        stream = self._handles[path]
        stream.seek(offset)
        row = json.loads(stream.readline())
        direction = f"{row['source_language']}-to-{row['target_language']}"
        return direction, row["target_language"], row["target"], row["problem_id"]


class TranslationCollator:
    """Pad causal-LM examples and construct the sparse region graph batch."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        architecture: str,
        random_seed: int = 42,
        generation_only: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.architecture = architecture
        self.random_seed = random_seed
        self.generation_only = generation_only

    def _random_regions(self, item: dict[str, Any], index: int = 0) -> None:
        """Create deterministic contiguous groups matching semantic group sizes."""
        mapping = item["source_token_to_region"]
        source_indexes = [i for i, region in enumerate(mapping) if region is not None]
        if not source_indexes:
            return
        old_sizes: dict[int, int] = {}
        for region_id in mapping:
            if region_id is not None:
                old_sizes[region_id] = old_sizes.get(region_id, 0) + 1
        sizes = list(old_sizes.values())
        del index  # Partition membership must not depend on batch order.
        stable_key = item["row"].get("source_sha256", item["row"]["problem_id"])
        rng = random.Random(self.random_seed ^ sum(ord(character) for character in stable_key))
        rng.shuffle(sizes)
        cursor = 0
        random_mapping = [None] * len(mapping)
        random_regions = []
        for region_id, size in enumerate(sizes):
            assigned = source_indexes[cursor : cursor + size]
            if not assigned:
                continue
            cursor += len(assigned)
            for token_index in assigned:
                random_mapping[token_index] = region_id
            random_regions.append(
                {
                    "region_id": region_id,
                    "region_type": "OTHER_STRUCTURAL",
                    "ast_depth": 0,
                    "child_count": 0,
                    "cfg_degree": 0,
                    "dfg_degree": 0,
                    "region_length": len(assigned),
                }
            )
        for token_index in source_indexes[cursor:]:
            region_id = len(random_regions)
            random_mapping[token_index] = region_id
            random_regions.append(
                {
                    "region_id": region_id,
                    "region_type": "OTHER_STRUCTURAL",
                    "ast_depth": 0,
                    "child_count": 0,
                    "cfg_degree": 0,
                    "dfg_degree": 0,
                    "region_length": 1,
                }
            )
        item["source_token_to_region"] = random_mapping
        item["regions"] = random_regions
        item["edges"] = []

    def _shuffle_structure(self, item: dict[str, Any], index: int = 0) -> None:
        """Deterministically permute typed graph links and region labels per example."""
        problem_id = str(item["row"]["problem_id"])
        del index
        source_hash = item["row"].get("source_sha256", problem_id)
        seed = self.random_seed ^ sum(ord(character) for character in source_hash)
        rng = random.Random(seed)
        for field in (
            "region_type",
            "ast_depth",
            "child_count",
            "cfg_degree",
            "dfg_degree",
        ):
            values = [region.get(field, 0) for region in item["regions"]]
            rng.shuffle(values)
            for region, value in zip(item["regions"], values, strict=False):
                region[field] = value
        sources = [edge["source"] for edge in item["edges"]]
        targets = [edge["target"] for edge in item["edges"]]
        rng.shuffle(sources)
        rng.shuffle(targets)
        edge_types = [edge["edge_type"] for edge in item["edges"]]
        rng.shuffle(edge_types)
        item["edges"] = [
            {"source": source, "target": target, "edge_type": edge_type}
            for source, target, edge_type in zip(sources, targets, edge_types, strict=False)
        ]

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if self.architecture == "random_region":
            for index, item in enumerate(items):
                self._random_regions(item, index)
        elif self.architecture == "shuffled_structure":
            for index, item in enumerate(items):
                item["regions"] = [dict(region) for region in item["regions"]]
                item["edges"] = [dict(edge) for edge in item["edges"]]
                self._shuffle_structure(item, index)
        batch_size = len(items)
        sequence_ids = [
            item["prompt_ids"] if self.generation_only else item["input_ids"] for item in items
        ]
        max_tokens = max(len(token_ids) for token_ids in sequence_ids)
        max_regions = max(len(item["regions"]) for item in items)
        pad_token = self.tokenizer.pad_token_id
        if pad_token is None:
            pad_token = self.tokenizer.eos_token_id
        input_ids = torch.full((batch_size, max_tokens), pad_token, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
        labels = torch.full((batch_size, max_tokens), -100, dtype=torch.long)
        token_to_region = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
        region_mask = torch.zeros((batch_size, max_regions), dtype=torch.bool)
        region_type = torch.zeros((batch_size, max_regions), dtype=torch.long)
        ast_depth = torch.zeros((batch_size, max_regions), dtype=torch.float32)
        child_count = torch.zeros_like(ast_depth)
        cfg_degree = torch.zeros_like(ast_depth)
        dfg_degree = torch.zeros_like(ast_depth)
        region_length = torch.zeros_like(ast_depth)
        edge_indexes: list[list[int]] = []
        edge_types: list[int] = []
        source_languages: list[int] = []
        target_languages: list[int] = []
        relation_allowlist = {
            "ast_sart": {"AST_PARENT", "AST_CHILD"},
            "cfg_sart": {"CFG_NEXT", "CFG_TRUE", "CFG_FALSE", "CFG_BACKEDGE"},
            "dfg_sart": {"DFG_DEF_USE", "DFG_USE_DEF"},
            "ast_cfg_sart": {
                "AST_PARENT",
                "AST_CHILD",
                "CFG_NEXT",
                "CFG_TRUE",
                "CFG_FALSE",
                "CFG_BACKEDGE",
            },
            "ast_dfg_sart": {"AST_PARENT", "AST_CHILD", "DFG_DEF_USE", "DFG_USE_DEF"},
            "sart_no_message": set(),
        }
        allowed_relations = relation_allowlist.get(self.architecture)
        for batch_index, item in enumerate(items):
            row = item["row"]
            current_ids = sequence_ids[batch_index]
            length = len(current_ids)
            prompt_length = len(item["prompt_ids"])
            input_ids[batch_index, :length] = torch.tensor(current_ids)
            attention_mask[batch_index, :length] = 1
            if not self.generation_only:
                labels[batch_index, :length] = torch.tensor(item["labels"])
            token_to_region[batch_index, :prompt_length] = torch.tensor(
                [
                    -1 if region_id is None else region_id
                    for region_id in item["source_token_to_region"]
                ]
            )
            for region in item["regions"]:
                region_id = int(region["region_id"])
                region_mask[batch_index, region_id] = True
                region_type[batch_index, region_id] = _region_type_id(
                    region["region_type"], self.architecture
                )
                scalar_features = _structural_scalars(region, self.architecture)
                ast_depth[batch_index, region_id] = scalar_features["ast_depth"]
                child_count[batch_index, region_id] = scalar_features["child_count"]
                cfg_degree[batch_index, region_id] = scalar_features["cfg_degree"]
                dfg_degree[batch_index, region_id] = scalar_features["dfg_degree"]
                region_length[batch_index, region_id] = float(region.get("region_length", 1))
            for edge in item["edges"]:
                if allowed_relations is not None and edge["edge_type"] not in allowed_relations:
                    continue
                edge_type = RELATIONS.get(edge["edge_type"])
                if edge_type is None:
                    continue
                edge_indexes.append([batch_index, int(edge["source"]), int(edge["target"])])
                edge_types.append(edge_type)
            source_languages.append(LANGUAGES[row["source_language"]])
            target_languages.append(LANGUAGES[row["target_language"]])
        region_batch = RegionBatch(
            token_to_region=token_to_region,
            region_mask=region_mask,
            region_type=region_type,
            ast_depth=ast_depth,
            child_count=child_count,
            cfg_degree=cfg_degree,
            dfg_degree=dfg_degree,
            region_length=region_length,
            edge_index=torch.tensor(edge_indexes, dtype=torch.long).reshape(-1, 3),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            source_language=torch.tensor(source_languages, dtype=torch.long),
            target_language=torch.tensor(target_languages, dtype=torch.long),
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "region_batch": region_batch,
            "task_ids": [item["row"]["problem_id"] for item in items],
            "target_texts": [item["row"]["target"] for item in items],
        }
