"""Strict, lightweight configuration for judge experiments.

Precedence: dataclass defaults < model YAML < experiment YAML < overrides.
Mappings merge recursively; sequences and scalars replace earlier values.
Unknown keys and invalid types fail rather than silently changing a run.
No model loading or hardware compatibility claims are made here.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

DType = Literal["bfloat16", "float16", "float32"]


def _decode(annotation: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return _decode(option, value, path)
            except ValueError:
                pass
    elif origin is Literal:
        if any(type(value) is type(choice) and value == choice for choice in args):
            return value
    elif origin is tuple:
        if isinstance(value, (list, tuple)):
            return tuple(_decode(args[0], item, f"{path}[{i}]") for i, item in enumerate(value))
    elif is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            known = {f.name for f in fields(annotation)}
            unknown = value.keys() - known
            if unknown:
                raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
            hints = get_type_hints(annotation)
            return annotation(**{
                key: _decode(hints[key], item, f"{path}.{key}")
                for key, item in value.items()
            })
    elif annotation is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif type(value) is annotation:
        return value
    raise ValueError(f"{path}: expected {annotation}, got {value!r}")


class _Section:
    def __post_init__(self) -> None:
        for name, annotation in get_type_hints(type(self)).items():
            object.__setattr__(self, name, _decode(annotation, getattr(self, name), name))
        self._validate()

    def _validate(self) -> None:
        pass


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class BackboneConfig(_Section):
    checkpoint: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "main"
    tokenizer_checkpoint: str | None = None  # None follows the backbone
    tokenizer_revision: str | None = None   # None follows backbone revision
    use_chat_template: bool = True
    chat_template: str | None = None        # None uses the checkpoint template
    enable_thinking: Literal[False] = False
    compute_dtype: DType = "bfloat16"
    max_input_length: int = 4096
    attention_implementation: Literal["eager", "sdpa"] = "sdpa"
    gradient_checkpointing: bool = False

    def _validate(self) -> None:
        for name in ("checkpoint", "revision"):
            _nonempty(name, getattr(self, name))
        for name in ("tokenizer_checkpoint", "tokenizer_revision", "chat_template"):
            if getattr(self, name) is not None:
                _nonempty(name, getattr(self, name))
        _positive("max_input_length", self.max_input_length)
        if not self.use_chat_template and self.chat_template is not None:
            raise ValueError("chat_template requires use_chat_template")


@dataclass(frozen=True)
class JudgeConfig(_Section):
    kind: Literal["pairwise", "shared_scorer"] = "pairwise"
    pooling: Literal["last_non_padding"] = "last_non_padding"
    head_init: Literal["normal"] = "normal"
    head_init_std: float = 0.02
    head_bias: bool = True
    tie_policy: Literal["abstain"] = "abstain"

    def _validate(self) -> None:
        _positive("head_init_std", self.head_init_std)

    @property
    def output_dim(self) -> int:
        return 2 if self.kind == "pairwise" else 1


@dataclass(frozen=True)
class LoRAConfig(_Section):
    enabled: bool = True
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_preset: Literal["attention", "all_language_linear"] | None = "all_language_linear"
    target_modules: tuple[str, ...] = ()
    exclude_modules: tuple[str, ...] = ()
    bias: Literal["none"] = "none"

    def _validate(self) -> None:
        _positive("lora.rank", self.rank)
        _positive("lora.alpha", self.alpha)
        if not 0 <= self.dropout < 1:
            raise ValueError("lora.dropout must be in [0, 1)")
        if (self.target_preset is None) == (not self.target_modules):
            raise ValueError("choose exactly one of lora.target_preset or lora.target_modules")
        for name in ("target_modules", "exclude_modules"):
            values = getattr(self, name)
            for value in values:
                _nonempty(f"lora.{name}", value)
            if len(values) != len(set(values)):
                raise ValueError(f"lora.{name} contains duplicates")
        if set(self.target_modules) & set(self.exclude_modules):
            raise ValueError("explicit LoRA targets and exclusions overlap")


@dataclass(frozen=True)
class QuantizationConfig(_Section):
    mode: Literal["none", "4bit"] = "none"
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: DType = "bfloat16"


@dataclass(frozen=True)
class DataConfig(_Section):
    augmentation: Literal["none", "both_orders"] = "none"
    truncation: Literal["independent_budgets"] = "independent_budgets"
    truncation_side: Literal["right"] = "right"
    prompt_max_tokens: int = 768
    answer_max_tokens: int = 1408  # Same cap for each candidate in all conditions

    def _validate(self) -> None:
        _positive("prompt_max_tokens", self.prompt_max_tokens)
        _positive("answer_max_tokens", self.answer_max_tokens)


@dataclass(frozen=True)
class TrainingConfig(_Section):
    seed: int = 42
    batch_size: int = 2  # Preference pairs per microbatch for either judge
    gradient_accumulation_steps: int = 8
    optimizer: Literal["adamw"] = "adamw"
    adapter_learning_rate: float = 0.0002
    head_learning_rate: float = 0.0002
    weight_decay: float = 0.01
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_ratio: float = 0.03
    max_steps: int = 1000  # Optimizer steps, not epochs; provisional run budget

    def _validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in ("batch_size", "gradient_accumulation_steps", "max_steps",
                     "adapter_learning_rate", "head_learning_rate"):
            _positive(name, getattr(self, name))
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig(_Section):
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def _validate(self) -> None:
        if self.judge.kind == "shared_scorer" and self.data.augmentation != "none":
            raise ValueError("shared_scorer uses independent scores; disable swap augmentation")
        if self.quantization.mode == "4bit":
            if self.quantization.compute_dtype != self.backbone.compute_dtype:
                raise ValueError("4-bit compute_dtype must match backbone.compute_dtype")
        candidates = 2 if self.judge.kind == "pairwise" else 1
        content_budget = self.data.prompt_max_tokens + candidates * self.data.answer_max_tokens
        if content_budget >= self.backbone.max_input_length:
            raise ValueError("token budgets must leave room for template tokens in max_input_length")

    def to_dict(self) -> dict:
        """Return JSON-compatible resolved settings, including defaults."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict) -> ExperimentConfig:
        return _decode(cls, value, "config")

    def save_json(self, path: str | Path) -> None:
        """Write settings only; the backbone report records checkpoint provenance."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _check_patch(section: type, patch: dict, path: str = "config") -> None:
    """Check supplied keys/types without applying defaults to partial sections."""
    hints = get_type_hints(section)
    for key, value in patch.items():
        if key not in hints:
            raise ValueError(f"{path}: unknown key {key!r}")
        annotation = hints[key]
        if is_dataclass(annotation) and isinstance(value, dict):
            _check_patch(annotation, value, f"{path}.{key}")
        else:
            _decode(annotation, value, f"{path}.{key}")


def load_config(
    model_path: str | Path,
    experiment_path: str | Path,
    *,
    overrides: dict | None = None,
) -> ExperimentConfig:
    """Load YAML/JSON files with nested mapping overrides (future CLI boundary).

    Overrides use section mappings, e.g. {"lora": {"rank": 16}}. Arrays replace
    earlier arrays. To use explicit targets, also set target_preset to null.
    """
    import yaml  # Lightweight optional-at-import dependency; never loads torch

    merged = ExperimentConfig().to_dict()
    for path in (model_path, experiment_path):
        contents = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(contents, dict):
            raise ValueError(f"{path}: expected a configuration mapping")
        # Validate each supplied section before merging so a later override
        # cannot silently conceal an unknown key in an earlier file.
        _check_patch(ExperimentConfig, contents)
        merged = _merge(merged, contents)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a mapping")
        _check_patch(ExperimentConfig, overrides)
        merged = _merge(merged, overrides)
    return ExperimentConfig.from_dict(merged)
