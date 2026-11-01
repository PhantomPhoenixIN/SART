"""Resume-safe PEFT training for the fixed-horizon translation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.translation_protocol import extract_generated_code, normalize_line_endings
from src.evaluation.lexical_metrics import compute_lexical_metrics
from src.models.recurrent.backbone_hook import MidLayerSourceAdapter, decoder_layers
from src.models.recurrent.region_module import DenseCapacityAdapter, RecurrentRegionCore
from src.training.data import ManifestDataset, TranslationCollator

_INTERRUPT_REQUESTED = False


def _request_interrupt(signum: int, frame: Any) -> None:
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True


@dataclass(frozen=True)
class RunConfig:
    """All settings that affect a single translation training run."""

    experiment_id: str
    architecture: str
    backbone_path: str
    train_manifest: str
    validation_manifest: str
    output_dir: str
    seed: int = 42
    epochs: int = 3
    max_sequence_length: int = 2048
    insertion_fraction: float = 0.5
    width: int = 1024
    max_recurrence: int = 6
    expected_depth_budget: float = 3.0
    budget_lambda: float = 0.1
    router_temperature: float = 1.0
    learning_rate: float = 1.0e-5
    adapter_learning_rate: float = 5.0e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    micro_batch_size: int = 1
    gradient_accumulation: int = 16
    warmup_fraction: float = 0.04
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    checkpoint_every_optimizer_steps: int = 100
    validation_batch_size: int = 1
    generation_examples_per_direction: int = 16
    generation_max_new_tokens: int = 768
    num_workers: int = 0
    hard_threshold: float | None = None
    unfreeze_last_layers: int = 0


def _config_hash(config: RunConfig) -> str:
    canonical = asdict(config)
    canonical.pop("experiment_id")
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


_COMPATIBLE_ADDED_DEFAULTS = {
    "insertion_fraction": 0.5,
    "router_temperature": 1.0,
    "unfreeze_last_layers": 0,
}


def _config_matches_saved_run(config: RunConfig, saved: dict[str, Any]) -> bool:
    """Match old resolved configs only when newly added settings retain known defaults."""
    current = asdict(config)
    for name, saved_value in saved.items():
        if name == "experiment_id":
            continue
        if name not in current or current[name] != saved_value:
            return False
    for name, current_value in current.items():
        if name == "experiment_id" or name in saved:
            continue
        if _COMPATIBLE_ADDED_DEFAULTS.get(name) != current_value:
            return False
    return True


def _saved_config_hash(saved: dict[str, Any]) -> str:
    """Reproduce the hash written by the code version that serialized `saved`."""
    canonical = dict(saved)
    canonical.pop("experiment_id", None)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _discover_resume_experiment_id(config: RunConfig) -> str | None:
    """Find one resumable run whose saved config matches this config plus default additions."""
    root = Path(config.output_dir)
    matches = []
    for resolved_path in root.glob("*/resolved_config.json"):
        if not (resolved_path.parent / "last.pt").is_file():
            continue
        try:
            saved = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _config_matches_saved_run(config, saved):
            matches.append(str(saved.get("experiment_id", resolved_path.parent.name)))
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple resumable runs match this configuration under {root}: {matches}"
        )
    return matches[0] if matches else None


def _canonical_experiment_id(config: RunConfig) -> str:
    """Build a deterministic registry key from all run settings except the key itself."""
    return f"{config.architecture}-seed{config.seed}-{_config_hash(config)[:12]}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _effective_num_workers(config: RunConfig) -> int:
    """Resolve a throughput-only DataLoader worker override from the job environment."""
    raw_value = os.environ.get("SART_NUM_WORKERS")
    if raw_value is None:
        if Path(config.output_dir).name == "confirmatory_7b":
            return max(config.num_workers, 2)
        return config.num_workers
    try:
        worker_count = int(raw_value)
    except ValueError as exc:
        raise ValueError("SART_NUM_WORKERS must be a non-negative integer") from exc
    if worker_count < 0:
        raise ValueError("SART_NUM_WORKERS must be a non-negative integer")
    return worker_count


def _unfreeze_last_decoder_layers(model: torch.nn.Module, count: int) -> None:
    """Unfreeze only the requested top decoder blocks after PEFT wrapping."""
    if count < 0:
        raise ValueError("unfreeze_last_layers cannot be negative")
    layers = decoder_layers(model)
    if count > len(layers):
        raise ValueError(f"Cannot unfreeze {count} layers from a {len(layers)}-layer backbone")
    if count:
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)


def _architecture_adapter(config: RunConfig, hidden_size: int):
    if config.architecture == "dense":
        return None
    if config.architecture == "dense_param_match":
        with torch.random.fork_rng(devices=[]):
            reference = RecurrentRegionCore(
                hidden_size=hidden_size,
                width=config.width,
                max_steps=config.max_recurrence,
                mode="sart",
            )
            target_parameters = sum(parameter.numel() for parameter in reference.parameters())
        bottleneck = round((target_parameters - hidden_size - 1) / (2 * hidden_size + 1))
        return DenseCapacityAdapter(hidden_size, bottleneck)
    modes = {
        "uniform_token_loop": "uniform_token",
        "uniform_region_loop": "uniform_region",
        "mor_style": "mor_token",
        "random_region": "random_region",
        "region_adaptive_no_graph": "region_adaptive_no_graph",
        "structure_only": "structure_only",
        "ast_sart": "sart",
        "full_sart": "sart",
        "shuffled_structure": "sart",
        "cfg_sart": "sart",
        "dfg_sart": "sart",
        "ast_cfg_sart": "sart",
        "ast_dfg_sart": "sart",
        "sart_no_scalar": "sart",
        "sart_no_message": "sart_no_message",
        "sart_attention_pool": "sart",
        "sart_no_language_pair": "sart",
        "sart_coarse_ontology": "sart",
    }
    if config.architecture not in modes:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    return RecurrentRegionCore(
        hidden_size=hidden_size,
        width=config.width,
        max_steps=config.max_recurrence,
        mode=modes[config.architecture],
        pooling_mode="attention" if config.architecture == "sart_attention_pool" else "mean",
        use_language_pair=config.architecture != "sart_no_language_pair",
        router_temperature=config.router_temperature,
    )


def _save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        saved_cuda_states = state["torch_cuda"]
        visible_devices = torch.cuda.device_count()
        if len(saved_cuda_states) == visible_devices:
            torch.cuda.set_rng_state_all(saved_cuda_states)
        else:
            # A checkpoint may be resumed with a different CUDA_VISIBLE_DEVICES
            # mapping. Its saved cuda:0 state maps to the process's first visible
            # device; preserve it and leave any additional device states as seeded.
            for device_index in range(min(len(saved_cuda_states), visible_devices)):
                torch.cuda.set_rng_state(saved_cuda_states[device_index], device_index)


def _checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: RunConfig,
    epoch: int,
    examples_seen: int,
    global_step: int,
    best_validation_loss: float,
    validation_loss: float | None,
) -> None:
    _save_atomic(
        path,
        {
            "config_hash": _config_hash(config),
            "config": asdict(config),
            "trainable_state": _trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
            "epoch": epoch,
            "examples_seen": examples_seen,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "validation_loss": validation_loss,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_model(config: RunConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    causal_lm = AutoModelForCausalLM.from_pretrained(
        config.backbone_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    causal_lm.config.use_cache = False
    causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(causal_lm, "enable_input_require_grads"):
        causal_lm.enable_input_require_grads()
    causal_lm = get_peft_model(
        causal_lm,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    _unfreeze_last_decoder_layers(causal_lm, config.unfreeze_last_layers)
    hidden_size = int(causal_lm.config.hidden_size)
    adapter = _architecture_adapter(config, hidden_size)
    if adapter is None:
        model: torch.nn.Module = causal_lm
    else:
        model = MidLayerSourceAdapter(
            causal_lm,
            adapter,
            insertion_fraction=config.insertion_fraction,
            max_source_length=config.max_sequence_length,
        )
    model.to(device)
    return model, tokenizer


def _validate(
    model: torch.nn.Module,
    dataset: ManifestDataset,
    tokenizer,
    config: RunConfig,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=TranslationCollator(tokenizer, config.architecture, config.seed),
    )
    total_loss = 0.0
    total_tokens = 0
    depth_values: list[float] = []
    gate_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch.pop("task_ids")
            batch.pop("target_texts")
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["region_batch"] = batch["region_batch"].to(device)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if isinstance(model, MidLayerSourceAdapter):
                        output = model(
                            **batch,
                            labels=labels,
                            budget=config.expected_depth_budget,
                            hard_threshold=config.hard_threshold,
                            use_cache=False,
                        )
                        depth_values.append(
                            float(
                                model._last_diagnostics.get("mean_depth", torch.tensor(0.0)).item()
                            )
                        )
                        diagnostics = model._last_diagnostics
                        gate_key = (
                            "token_gate_probabilities"
                            if "token_gate_probabilities" in diagnostics
                            else "gate_probabilities"
                        )
                        probabilities = diagnostics.get(gate_key)
                        if probabilities is not None:
                            is_token = gate_key == "token_gate_probabilities"
                            valid = (
                                batch["region_batch"].token_to_region >= 0
                                if is_token
                                else batch["region_batch"].region_mask
                            )
                            gate_rows.append(
                                probabilities.permute(0, 2, 1)[valid].detach().float().cpu()
                            )
                    else:
                        output = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=labels,
                            use_cache=False,
                        )
            finally:
                if isinstance(model, MidLayerSourceAdapter):
                    model.clear_route()
            count = int((labels != -100).sum())
            total_loss += float(output.loss.item()) * count
            total_tokens += count
    if total_tokens == 0:
        raise RuntimeError("Validation split contains no target tokens")
    generation_stats = _validation_generation_metrics(model, dataset, tokenizer, config, device)
    router_stats = _router_validation_stats(
        gate_rows, config.max_recurrence, architecture=config.architecture
    )
    model.train()
    return total_loss / total_tokens, {
        "target_tokens": float(total_tokens),
        "mean_recurrent_depth": float(np.mean(depth_values)) if depth_values else 0.0,
        **generation_stats,
        **router_stats,
    }


def _router_validation_stats(
    gate_rows: list[torch.Tensor], max_steps: int, *, architecture: str | None = None
) -> dict[str, float]:
    """Summarize soft halting, depth mass, entropy, and collapse indicators."""
    one_pass_diagnostics = architecture == "structure_only" or (
        max_steps > 1
        and gate_rows
        and all(row.ndim == 2 and row.shape[1] == 1 for row in gate_rows)
    )
    if one_pass_diagnostics:
        # This control performs exactly one structural pass and has no adaptive
        # router. Its one-column diagnostic tensor is not a six-step halting distribution.
        return {
            "mean_continue_probability": 0.0,
            "router_entropy": 0.0,
            "stop_after_first_fraction": 0.0,
            "reach_rmax_fraction": 0.0,
            "depth_mass_r1": 1.0,
            **{f"depth_mass_r{step}": 0.0 for step in range(2, max_steps + 1)},
        }
    if not gate_rows:
        return {
            "mean_continue_probability": 0.0,
            "router_entropy": 0.0,
            "stop_after_first_fraction": 0.0,
            "reach_rmax_fraction": 0.0,
            **{f"depth_mass_r{step}": 0.0 for step in range(1, max_steps + 1)},
        }
    gates = torch.cat(gate_rows, dim=0).clamp(1e-7, 1.0 - 1e-7)
    entropy = -(gates * gates.log() + (1.0 - gates) * (1.0 - gates).log())
    depth_mass = []
    for depth in range(1, max_steps):
        survive = gates[:, : depth - 1].prod(-1) if depth > 1 else torch.ones_like(gates[:, 0])
        depth_mass.append(survive * (1.0 - gates[:, depth - 1]))
    final_survival = gates[:, : max_steps - 1].prod(-1)
    depth_mass.append(final_survival)
    return {
        "mean_continue_probability": float(gates.mean()),
        "router_entropy": float(entropy.mean()),
        "stop_after_first_fraction": float((1.0 - gates[:, 0]).mean()),
        "reach_rmax_fraction": float(final_survival.mean()),
        **{
            f"depth_mass_r{step}": float(mass.mean())
            for step, mass in enumerate(depth_mass, start=1)
        },
    }


def _validation_generation_metrics(
    model: torch.nn.Module,
    dataset: ManifestDataset,
    tokenizer,
    config: RunConfig,
    device: torch.device,
) -> dict[str, float]:
    """Compute a fixed validation lexical-metric bundle on a balanced subset."""
    if config.generation_examples_per_direction <= 0:
        names = (
            "bleu4",
            "codebleu",
            "codebleu_ngram",
            "codebleu_weighted_ngram",
            "codebleu_syntax",
            "codebleu_dataflow",
            "exact_match",
            "crystalbleu",
        )
        return {
            **{f"validation_{name}_macro": 0.0 for name in names},
            "generation_examples": 0.0,
            "generation_directions": 0.0,
            "crystal_k": 500.0,
        }
    by_direction: dict[str, list[tuple[str, int]]] = {}
    crystal_references_by_direction: dict[str, list[str]] = {}
    for index in range(len(dataset)):
        direction, _language, target_reference, problem_id = dataset.target_reference(index)
        crystal_references_by_direction.setdefault(direction, []).append(
            normalize_line_endings(target_reference)
        )
        key = hashlib.sha256(f"{config.seed}|{direction}|{problem_id}".encode()).hexdigest()
        by_direction.setdefault(direction, []).append((key, index))
    selected = [
        index
        for values in by_direction.values()
        for _, index in sorted(values)[: config.generation_examples_per_direction]
    ]
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=TranslationCollator(
            tokenizer,
            config.architecture,
            config.seed,
            generation_only=True,
        ),
    )
    by_direction: dict[str, dict[str, list[str]]] = {}
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            target_text = batch.pop("target_texts")[0]
            batch.pop("labels")
            batch.pop("task_ids")
            input_ids = batch.pop("input_ids").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            region_batch = batch.pop("region_batch").to(device)
            if isinstance(model, MidLayerSourceAdapter):
                _, token_ids = model.generate_greedy(
                    input_ids,
                    attention_mask,
                    region_batch,
                    max_new_tokens=config.generation_max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    budget=config.expected_depth_budget,
                    hard_threshold=config.hard_threshold,
                )
                prediction = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.generation_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prediction = tokenizer.decode(
                    generated[0, input_ids.shape[1] :], skip_special_tokens=True
                )
            row = dataset.row_metadata(selected[batch_index])
            direction = f"{row['source_language']}-to-{row['target_language']}"
            bucket = by_direction.setdefault(
                direction,
                {"references": [], "hypotheses": []},
            )
            bucket["references"].append(normalize_line_endings(target_text))
            bucket["hypotheses"].append(normalize_line_endings(extract_generated_code(prediction)))
            total += 1
    if isinstance(model, MidLayerSourceAdapter):
        model.clear_route()
    direction_metrics = {
        direction: compute_lexical_metrics(
            values["references"],
            values["hypotheses"],
            direction.split("-to-", 1)[1],
            crystal_reference_corpus=crystal_references_by_direction[direction],
        )
        for direction, values in by_direction.items()
    }
    metric_names = (
        "bleu4",
        "codebleu",
        "codebleu_ngram",
        "codebleu_weighted_ngram",
        "codebleu_syntax",
        "codebleu_dataflow",
        "exact_match",
        "crystalbleu",
    )
    aggregate = {
        f"validation_{name}_macro": float(
            np.mean([float(row[name]) for row in direction_metrics.values()])
        )
        for name in metric_names
    }
    aggregate.update(
        {
            "generation_examples": float(total),
            "generation_directions": float(len(direction_metrics)),
            "crystal_k": 500.0,
        }
    )
    return aggregate


def train(config: RunConfig) -> None:
    """Train/resume a run and choose the best checkpoint within its fixed horizon."""
    global _INTERRUPT_REQUESTED
    signal.signal(signal.SIGTERM, _request_interrupt)
    signal.signal(signal.SIGINT, _request_interrupt)
    if config.experiment_id != _canonical_experiment_id(config):
        raise ValueError("experiment_id must be the canonical hash-derived ID (or 'auto' in YAML)")
    _seed_everything(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires a GPU PBS allocation; this process has no CUDA device"
        )
    device = torch.device("cuda:0")
    output = Path(config.output_dir) / config.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    num_workers = _effective_num_workers(config)
    resolved_path = output / "resolved_config.json"
    resolved_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    model, tokenizer = _load_model(config, device)
    train_data = ManifestDataset(
        Path(config.train_manifest), tokenizer, "train", config.max_sequence_length
    )
    validation_data = ManifestDataset(
        Path(config.validation_manifest), tokenizer, "validation", config.max_sequence_length
    )
    parameter_summary = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "train_examples_per_epoch": len(train_data),
        "validation_examples": len(validation_data),
        "effective_train_dataloader_workers": num_workers,
        "train_manifest_sha256": hashlib.sha256(
            Path(config.train_manifest).read_bytes()
        ).hexdigest(),
        "validation_root": str(Path(config.validation_manifest).resolve()),
    }
    (output / "model_summary.json").write_text(
        json.dumps(parameter_summary, indent=2, sort_keys=True) + "\n"
    )
    adapter_parameters = []
    backbone_parameters = []
    backbone_parameter_ids = {
        id(parameter)
        for parameter in (
            model.causal_lm.parameters()
            if isinstance(model, MidLayerSourceAdapter)
            else model.parameters()
        )
        if parameter.requires_grad
    }
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            backbone_parameters if id(parameter) in backbone_parameter_ids else adapter_parameters
        )
        destination.append(parameter)
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.learning_rate})
    if adapter_parameters:
        groups.append({"params": adapter_parameters, "lr": config.adapter_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = int(
        np.ceil(len(train_data) / (config.micro_batch_size * config.gradient_accumulation))
    )
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * config.warmup_fraction),
        num_training_steps=total_steps,
    )
    last_path = output / "last.pt"
    start_epoch = 0
    examples_seen = 0
    global_step = 0
    best_loss = float("inf")
    resume_offset = 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        saved_config = state.get("config", {})
        if not _config_matches_saved_run(config, saved_config) or state.get(
            "config_hash"
        ) != _saved_config_hash(saved_config):
            raise ValueError("Checkpoint configuration hash differs; refusing unsafe resume")
        missing, unexpected = model.load_state_dict(state["trainable_state"], strict=False)
        if unexpected or any(name in state["trainable_state"] for name in missing):
            raise RuntimeError(
                f"Trainable checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng"])
        start_epoch = state["epoch"]
        resume_offset = state["examples_seen"] if start_epoch < config.epochs else 0
        examples_seen = resume_offset
        global_step = state["global_step"]
        best_loss = state["best_validation_loss"]
    model.train()
    collator = TranslationCollator(tokenizer, config.architecture, config.seed)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.epochs):
        epoch_seed = torch.Generator().manual_seed(config.seed + epoch)
        permutation = torch.randperm(len(train_data), generator=epoch_seed).tolist()
        offset = resume_offset if epoch == start_epoch else 0
        loader = DataLoader(
            Subset(train_data, permutation[offset:]),
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=num_workers,
            multiprocessing_context="spawn" if num_workers else None,
            collate_fn=collator,
        )
        epoch_loss = 0.0
        target_tokens = 0
        accumulation_count = 0
        router_gradient_norms: list[float] = []
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch.pop("task_ids")
            batch.pop("target_texts")
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["region_batch"] = batch["region_batch"].to(device)
            if isinstance(model, MidLayerSourceAdapter):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_values = model(
                        **batch,
                        labels=labels,
                        budget=config.expected_depth_budget,
                        hard_threshold=config.hard_threshold,
                        use_cache=False,
                    )
                    budget_loss = model._last_diagnostics.get(
                        "budget_loss", labels.new_zeros((), dtype=torch.float32)
                    )
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_values = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=labels,
                        use_cache=False,
                    )
                budget_loss = labels.new_zeros((), dtype=torch.float32)
            loss = output_values.loss + config.budget_lambda * budget_loss
            (loss / config.gradient_accumulation).backward()
            if isinstance(model, MidLayerSourceAdapter):
                model.clear_route()
            accumulation_count += 1
            token_count = int((labels != -100).sum())
            epoch_loss += float(output_values.loss.detach().item()) * token_count
            target_tokens += token_count
            examples_seen += len(batch["input_ids"])
            offset += len(batch["input_ids"])
            if accumulation_count == config.gradient_accumulation or offset == len(train_data):
                if accumulation_count < config.gradient_accumulation:
                    correction = config.gradient_accumulation / accumulation_count
                    for group in groups:
                        for parameter in group["params"]:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                if isinstance(model, MidLayerSourceAdapter):
                    router_parameters = [
                        parameter
                        for name, parameter in model.named_parameters()
                        if "router" in name and parameter.grad is not None
                    ]
                    if router_parameters:
                        squared_norm = sum(
                            float(parameter.grad.detach().float().norm().item()) ** 2
                            for parameter in router_parameters
                        )
                        router_gradient_norms.append(squared_norm**0.5)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in groups for parameter in group["params"]],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                global_step += 1
                if global_step % config.checkpoint_every_optimizer_steps == 0:
                    _checkpoint(
                        last_path,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch,
                        offset,
                        global_step,
                        best_loss,
                        None,
                    )
                if _INTERRUPT_REQUESTED:
                    _checkpoint(
                        last_path,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        epoch,
                        offset,
                        global_step,
                        best_loss,
                        None,
                    )
                    _atomic_jsonl(
                        output / "status.jsonl",
                        {"status": "PREEMPTED", "epoch": epoch, "step": global_step},
                    )
                    return
        validation_loss, validation_stats = _validate(
            model, validation_data, tokenizer, config, device
        )
        metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_target_loss": epoch_loss / max(1, target_tokens),
            "validation_target_loss": validation_loss,
            "train_target_tokens": target_tokens,
            "router_gradient_norm": (
                float(np.mean(router_gradient_norms)) if router_gradient_norms else 0.0
            ),
            **validation_stats,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_jsonl(output / "metrics.jsonl", metrics)
        _checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            config,
            epoch + 1,
            0,
            global_step,
            min(best_loss, validation_loss),
            validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            _checkpoint(
                output / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch + 1,
                0,
                global_step,
                best_loss,
                validation_loss,
            )
        resume_offset = 0
        if _INTERRUPT_REQUESTED:
            _atomic_jsonl(
                output / "status.jsonl",
                {"status": "PREEMPTED", "epoch": epoch + 1, "step": global_step},
            )
            return
    _atomic_jsonl(
        output / "status.jsonl", {"status": "TRAINED", "epoch": config.epochs, "step": global_step}
    )
    if isinstance(model, MidLayerSourceAdapter):
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_values = yaml.safe_load(args.config.read_text())
    config = RunConfig(**config_values)
    if config.experiment_id == "auto":
        experiment_id = _canonical_experiment_id(config)
        expected_checkpoint = Path(config.output_dir) / experiment_id / "last.pt"
        if not expected_checkpoint.is_file():
            experiment_id = _discover_resume_experiment_id(config) or experiment_id
        config = replace(config, experiment_id=experiment_id)
    print(json.dumps(asdict(config), sort_keys=True), flush=True)
    train(config)


if __name__ == "__main__":
    main()
