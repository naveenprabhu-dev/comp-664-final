"""Qwen3.5 text-only loading and auditable adapter attachment.

Heavy dependencies are imported only when a model operation is requested.
Configuration and report inspection therefore remain usable on data-only hosts.
No judge heads, pooling, tokenization, or training loop are implemented here.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ExperimentConfig, LoRAConfig

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class AdapterSpec:
    module: str  # Exact path relative to the text backbone
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BackboneReport:
    checkpoint: str
    requested_revision: str
    resolved_revision: str | None
    model_class: str
    hidden_size: int
    device: str
    compute_dtype: str
    quantization: str
    gradient_checkpointing: bool
    adapters: tuple[AdapterSpec, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int  # Logical elements, not packed 4-bit storage elements
    versions: dict[str, str]
    ignored_checkpoint_keys: tuple[str, ...]

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass
class LoadedBackbone:
    model: nn.Module  # Returns .last_hidden_state [batch, length, hidden_size]
    report: BackboneReport


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
        from transformers import Qwen3_5TextModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Backbone loading requires the training stack with native Qwen3.5 support. "
            "See requirements-training.txt (Transformers 5.3.0 is tested)."
        ) from exc
    return torch, transformers


def _assert_text_backbone(model: nn.Module) -> None:
    _, transformers = _dependencies()
    if not isinstance(model, transformers.Qwen3_5TextModel):
        raise ValueError("Expected a bare Qwen3_5TextModel, without vision, vocabulary, or task heads")


def resolve_lora_targets(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Resolve exact language-only targets before PEFT mutates the module tree.

    attention: all Linear projections in self_attn and linear_attn (including
    DeltaNet gates); all_language_linear: also includes decoder MLP projections.
    Convolutions, embeddings, norms, vision, and heads are never eligible.
    """
    torch, _ = _dependencies()
    _assert_text_backbone(model)
    if not config.enabled:
        return ()
    eligible = {
        name for name, module in model.named_modules()
        if re.fullmatch(r"layers\.\d+\.(self_attn|linear_attn|mlp)\.[^.]+", name)
        and isinstance(module, torch.nn.Linear)
    }
    for name in (*config.target_modules, *config.exclude_modules):
        if name not in eligible:
            raise ValueError(f"Unknown or ineligible language linear module: {name!r}")
    if config.target_preset == "attention":
        selected = {name for name in eligible if ".mlp." not in name}
    elif config.target_preset == "all_language_linear":
        selected = eligible
    else:
        selected = set(config.target_modules)
    selected = selected - set(config.exclude_modules)
    if not selected:
        raise ValueError("LoRA selection resolved to no language linear modules")
    return tuple(sorted(selected))


def _parameter_count(parameter: Any) -> int:
    # Params4bit.numel() counts packed storage after moving onto CUDA.
    if parameter.__class__.__name__ == "Params4bit":
        return parameter.numel() * 2 * parameter.element_size()
    return parameter.numel()


def prepare_backbone(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    resolved_revision: str | None = None,
    ignored_checkpoint_keys: tuple[str, ...] = (),
) -> LoadedBackbone:
    """Freeze a loaded text backbone and attach adapters; never add a task head.

    Public separately from load_backbone so tiny real Qwen models can exercise
    freezing, forward/backward, and adapter placement entirely offline.
    Quantized models must already have been loaded in 4-bit by Transformers.
    """
    torch, _ = _dependencies()
    _assert_text_backbone(model)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("Backbone already contains LoRA parameters")
    is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    if is_4bit != (config.quantization.mode == "4bit"):
        raise ValueError("Loaded backbone quantization does not match configuration")

    targets = resolve_lora_targets(model, config.lora)
    original_parameters = tuple(model.parameters())
    model.requires_grad_(False)
    model.config.use_cache = False
    if is_4bit:
        from peft import prepare_model_for_kbit_training

        # Non-reentrant checkpointing needs no gradients on frozen embeddings.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if config.backbone.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    adapters = []
    allowed_trainable: set[int] = set()
    if config.lora.enabled:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Adapter attachment requires PEFT; see requirements-training.txt") from exc
        # A regex with fullmatch semantics prevents PEFT's list-based suffix
        # matching from attaching to an unintended module with a similar name.
        target_regex = "(?:" + "|".join(re.escape(name) for name in targets) + ")"
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            target_modules=target_regex,
        ))
        base = model.get_base_model()
        for name in targets:
            layer = base.get_submodule(name)
            # Explicit policy: train adapters in FP32, including BF16/QLoRA runs.
            # PEFT casts inputs to adapter dtype and adds the correction back in
            # the base layer's output dtype. The future head chooses its dtype.
            layer.lora_A["default"].to(dtype=torch.float32)
            layer.lora_B["default"].to(dtype=torch.float32)
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            allowed_trainable.update((id(a), id(b)))
            adapters.append(AdapterSpec(name, tuple(a.shape), tuple(b.shape), str(a.dtype)))
    if any(parameter.requires_grad for parameter in original_parameters):
        raise RuntimeError("Original backbone parameters must remain frozen")
    actual_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if actual_trainable != allowed_trainable:
        raise RuntimeError("Trainable parameters differ from the resolved LoRA matrices")
    model.train()
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    parameters = tuple(model.named_parameters())
    base = model.get_base_model() if config.lora.enabled else model
    report = BackboneReport(
        checkpoint=config.backbone.checkpoint,
        requested_revision=config.backbone.revision,
        resolved_revision=resolved_revision,
        model_class=type(base).__name__,
        hidden_size=base.config.hidden_size,
        device=str(base.get_input_embeddings().weight.device),
        compute_dtype=config.backbone.compute_dtype,
        quantization=config.quantization.mode,
        gradient_checkpointing=config.backbone.gradient_checkpointing,
        adapters=tuple(adapters),
        trainable_parameter_names=tuple(name for name, p in parameters if p.requires_grad),
        trainable_parameters=sum(_parameter_count(p) for _, p in parameters if p.requires_grad),
        total_parameters=sum(_parameter_count(p) for _, p in parameters),
        versions=versions,
        ignored_checkpoint_keys=ignored_checkpoint_keys,
    )
    return LoadedBackbone(model, report)


def load_backbone(
    config: ExperimentConfig,
    *,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> LoadedBackbone:
    """Load only pretrained text weights, then freeze and attach adapters.

    Single-device training only: pass cuda:LOCAL_RANK from a future distributed
    launcher, never device_map='auto'. No remote repository code is executed.
    CPU use is intended for FP32 smoke checks. Full checkpoints are downloaded
    only when this function is explicitly called with local_files_only=False.
    """
    torch, transformers = _dependencies()
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit CPU or CUDA devices are supported")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if target_device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {target_device}")
        with torch.cuda.device(target_device):
            if config.backbone.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 is unsupported on the selected CUDA device; choose float16")
    if config.quantization.mode == "4bit" and target_device.type != "cuda":
        raise ValueError("This loader supports bitsandbytes 4-bit training on CUDA only")

    from orderbias.config import CACHE_DIR

    source_options = dict(
        revision=config.backbone.revision,
        cache_dir=str(cache_dir if cache_dir is not None else CACHE_DIR),
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    raw_config = transformers.AutoConfig.from_pretrained(config.backbone.checkpoint, **source_options)
    # Resolve 'main' once so config and weights cannot come from different
    # commits if the upstream branch advances during this load.
    resolved_revision = getattr(raw_config, "_commit_hash", None)
    if resolved_revision is not None:
        source_options["revision"] = resolved_revision
    if raw_config.model_type == "qwen3_5":
        text_config = raw_config.text_config
        key_mapping = {r"^model\.language_model\.": ""}
    elif raw_config.model_type == "qwen3_5_text":
        text_config = raw_config
        key_mapping = None
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {raw_config.model_type!r}")
    if config.backbone.max_input_length > text_config.max_position_embeddings:
        raise ValueError("max_input_length exceeds the checkpoint's positional limit")
    text_config.use_cache = False
    load_options: dict[str, Any] = dict(
        config=text_config,
        dtype=getattr(torch, config.backbone.compute_dtype),
        attn_implementation=config.backbone.attention_implementation,
        device_map={"": str(target_device)},
        output_loading_info=True,
        key_mapping=key_mapping,
    )
    if config.quantization.mode == "4bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError("QLoRA requires a CUDA-compatible bitsandbytes installation") from exc
        load_options["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.compute_dtype),
        )
    model, info = transformers.Qwen3_5TextModel.from_pretrained(
        config.backbone.checkpoint, **source_options, **load_options,
    )
    # Never silently train on randomly initialized language weights. The full
    # checkpoint's vision/MTP/vocabulary weights are expected to be unused.
    unexpected = tuple(info.get("unexpected_keys", ()))
    bad_extra = [key for key in unexpected if not key.startswith(("model.visual.", "mtp.", "lm_head."))]
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or bad_extra:
        raise RuntimeError(f"Incomplete or incompatible text backbone checkpoint: {info}")
    return prepare_backbone(
        model, config,
        resolved_revision=resolved_revision,
        ignored_checkpoint_keys=unexpected,
    )
